"""Evaluate continuous knee-to-stand attempts and learn residual actions with PPO."""

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import traceback

import torch

from .mount_teacher import frozen_entry, knee_entry_config, source_inputs
from .reference import digest
from .runner import LegacyRslAdapter, runner_config
from .training_control import save_checkpoint_atomically


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def select_controller(env, name):
    if name == "standard":
        return env.mount_control
    if name == "knee":
        from .mount_control import KneeMountControl

        return KneeMountControl(env)
    if name == "crawl":
        from .mount_crawl import MountCrawl

        return MountCrawl(env)
    raise ValueError(f"Unknown mount controller: {name}")


def controller_digest(control):
    """Hash only executable classes and local helpers used by this controller."""
    from .mount_control import MountControl
    from .knee_transfer import point_jacobian

    sources = [type(control)]
    if isinstance(control, MountControl):
        if type(control) is not MountControl:
            sources.append(MountControl)
        sources.append(point_jacobian)
    return hashlib.sha256("".join(inspect.getsource(obj) for obj in sources).encode()).hexdigest()


def normalize_contract(contract):
    # v1 stored all of mount_control.py. These two snapshots contain the very
    # same standard controller and differ only in the unused KneeMountControl class.
    if contract.get("schema") == "mount_residual_v1" and contract.get("controller_sha256") in {
        "cbf919fc35b04523e5e268676be667bd4e70055d4f8d4ca1d2c14d9ade50ccb6",
        "c1a22405ac55102d7d0fd2fc4dc1ae6ea7201d588bf5aca6d927b0dd20e07376",
    }:
        contract = dict(
            contract,
            schema="mount_residual_v2",
            controller="standard",
            controller_sha256="d60e23896c8921968e7f798ff0d36f1831f5b34d337a0e4c6695339786cab654",
        )
    # Only this audited predecessor has the exact historical reward, termination
    # and hold behavior provided by this version's legacy branch.
    if (
        contract.get("schema") == "mount_residual_v2"
        and "stance" not in contract
        and contract.get("mount_task_sha256")
        == "02d8552360a2c640847d7129d4600ea91e2aac5b58e4944fc310380085abcac3"
    ):
        contract = dict(
            contract,
            stance="legacy",
            mount_task_sha256="4a064e9513fa6aaeb602ba58ea77cc34e23cff5e38e1c789d994db084086a80f",
        )
    return contract


def validate_mount_contract(previous, current, requested_stance):
    if requested_stance and previous.get("stance") in ("legacy", "flat-foot"):
        previous = dict(previous, stance=requested_stance)
    if previous != current:
        raise ValueError("Residual checkpoint has incompatible assets, source or task settings")


def controller_columns(control):
    columns = dict(zip(control.parameter_names, control.parameters.unbind(1), strict=True))
    if hasattr(control, "transfer_parameter_names"):
        columns.update(
            zip(
                control.transfer_parameter_names, control.transfer_parameters.unbind(1), strict=True
            )
        )
    return columns


def set_controller_parameters(control, requested):
    """Apply named scalar overrides or a saved full parallel parameter table."""
    columns = controller_columns(control)
    if not isinstance(requested, dict):
        raise ValueError("--control-parameters must be a JSON object")
    batch = set(requested) == {"names", "values"}
    if batch:
        names = requested["names"]
        values = requested["values"]
        if (
            names != list(columns)
            or not isinstance(values, list)
            or len(values) != control.parameters.shape[0]
        ):
            raise ValueError(
                "Sweep table must match selected controller parameter names and --num-envs"
            )
        if any(not isinstance(row, list) or len(row) != len(names) for row in values):
            raise ValueError("Sweep table contains a row with a different number of parameters")
        requested = dict(zip(names, zip(*values), strict=True))
    for name, value in requested.items():
        values = value if batch else (value,)
        if name not in columns or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values
        ):
            raise ValueError(f"Unknown controller parameter or invalid value: {name}={value}")
        if name in ("tempo", "pose_blend_frames", "swing_duration_s") and min(values) <= 0:
            raise ValueError(f"{name} must be positive")
        if (
            name
            in (
                "position_feedback",
                "arm_feedback",
                "foot_clearance_m",
                "knee_clearance_m",
                "pull_gain_N_m",
                "swing_height_m",
                "wrist_anchor_weight",
            )
            and min(values) < 0
        ):
            raise ValueError(f"{name} must be nonnegative")
        columns[name].copy_(columns[name].new_tensor(value))


# All observations below come from the time step before the vectorized environment resets.
PAIR_FIELDS = (
    "foot_top_N",
    "foot_floor_N",
    "knee_top_N",
    "foot_inside_edge_m",
    "knee_inside_edge_m",
    "foot_above_top_m",
    "foot_bearing",
    "knee_bearing",
)
SCALAR_FIELDS = (
    "entry_accepted",
    "mount_active",
    "success",
    "stage",
    "standing",
    "stand_hold_s",
    "com_inside_edge_m",
    "torso_up",
    "root_speed_m_s",
    "hip_above_top_m",
)
SOLE_FIELDS = (
    "foot_tilt_deg",
    "foot_patch_major_std_m",
    "foot_patch_minor_std_m",
    "foot_patch_supported",
)

COURSE_FIELDS = (
    "course_stage",
    "walk_active",
    "walk_loaded",
    "rise_active",
    "walk_quality",
    "walk_slip_m_s",
    "walk_tilt_deg",
    "rise_quality",
)


class GroundAssessment:
    """Count only the first unassisted ground-start episode in each clone."""

    def __init__(self, env):
        root = env.robot.data.root_link_pos_w - env.scene.env_origins
        self.start_xy = root[:, :2].clone()
        self.direction = env.advance_direction[:2]
        self.advance = torch.zeros_like(root[:, 0])
        self.walk_origin = self.start_xy.clone()
        self.walk_seen = torch.zeros_like(self.advance, dtype=torch.bool)
        self.walk_advance = torch.zeros_like(self.advance)
        self.flip_drift = torch.zeros_like(self.advance)
        self.switches = torch.zeros_like(self.advance, dtype=torch.long)
        self.braced = torch.zeros_like(self.advance, dtype=torch.bool)
        self.sums = {k: torch.zeros_like(self.advance) for k in COURSE_FIELDS[1:]}

    def update(self, m, active):
        approach = active & (m["course_stage"] == 0)
        advance = ((m["root_pose"][:, :2] - self.start_xy) * self.direction).sum(1)
        self.advance = torch.where(approach, advance, self.advance)
        self.braced |= active & (m["approach_accepted"] | (m["course_stage"] >= 1))
        walking = active & m["walk_active"].bool()
        standing = approach & (m["course_time_s"] < 1.5)
        xy = m["root_pose"][:, :2]
        drift = (xy - self.start_xy).norm(dim=1)
        self.flip_drift = torch.maximum(self.flip_drift, drift * standing)
        self.walk_origin = torch.where((walking & ~self.walk_seen)[:, None], xy, self.walk_origin)
        self.walk_seen |= walking
        self.walk_advance = torch.where(
            walking, ((xy - self.walk_origin) * self.direction).sum(1), self.walk_advance
        )
        self.switches += walking & m["walk_step_switch"]
        bearing = walking & m["walk_loaded"].bool()
        rising = active & m["rise_active"].bool()
        for name, values in self.sums.items():
            if name in ("walk_slip_m_s", "walk_tilt_deg", "walk_loaded"):
                mask = bearing
            else:
                mask = walking if name.startswith("walk_") else rising
            values += torch.where(mask, m[name], 0)

    def summary(self):
        s = self.sums

        def average(name, count):
            # No stage samples is distinct from a genuinely measured zero.
            total = s[count].sum().item()
            return s[name].sum().item() / total if total else None

        return {
            "ground_braced": int(self.braced.sum()),
            "ground_mean_approach_advance_m": float(self.advance.mean()),
            "ground_mean_peak_flip_drift_m": float(self.flip_drift.mean()),
            "ground_mean_walk_advance_m": float(self.walk_advance.mean()),
            "ground_with_alternating_steps": int((self.switches > 0).sum()),
            "ground_walk_samples": int(s["walk_active"].sum()),
            "ground_walk_quality": average("walk_quality", "walk_active"),
            "ground_walk_loaded_samples": int(s["walk_loaded"].sum()),
            "ground_walk_loaded_slip_m_s": average("walk_slip_m_s", "walk_loaded"),
            "ground_walk_loaded_tilt_deg": average("walk_tilt_deg", "walk_loaded"),
            "ground_rise_samples": int(s["rise_active"].sum()),
            "ground_rise_quality": average("rise_quality", "rise_active"),
        }


@torch.inference_mode()
def assess(runner, wrapped, record_dir=None, camera=False, record_env=0):
    env = wrapped.env
    runner.eval_mode()
    if hasattr(env, "stage_training"):
        env.stage_training = False
    wrapped.obs, _ = env.reset()
    ground = GroundAssessment(env) if hasattr(env, "ground_stage") else None
    n = env.num_envs
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    accepted = torch.zeros_like(active)
    successes = torch.zeros_like(active)
    peak = {}
    minimum = {}
    final = {}
    both_knees = torch.zeros_like(active)
    any_foot = torch.zeros_like(active)
    both_feet = torch.zeros_like(active)
    survived = torch.zeros_like(active)
    post_seconds = torch.zeros(n, device=env.device)
    duration = torch.zeros(n, device=env.device)
    dwell = torch.zeros(n, 4, device=env.device)
    max_dwell = torch.zeros_like(dwell)
    total_return = torch.zeros_like(duration)
    if camera:
        point = env.robot.data.root_link_pos_w[record_env].tolist()
        env.sim.set_camera_view([point[0] - 1.2, point[1] - 2.3, point[2] + 0.8], point)
    with ExitStack() as stack:
        trace = stack.enter_context((record_dir / "trace.jsonl").open("w")) if record_dir else None
        for step in range(env.max_episode_length + 2):
            obs = runner.obs_normalizer(wrapped.obs["policy"])
            action = runner.alg.policy.act_inference(obs)
            _, reward, done, _ = wrapped.step(action)
            m = env.last_metrics
            if ground is not None:
                ground.update(m, active)
            post = active & m["mount_active"].bool()
            accepted |= active & (m["entry_accepted"].bool() | m["mount_active"].bool())
            successes |= active & m["success"].bool()
            both_knees |= post & m["knee_bearing"].all(1)
            any_foot |= post & m["foot_bearing"].any(1)
            both_feet |= post & m["foot_bearing"].all(1)
            survived |= post & done.bool().view(-1) & ~m["terminated"].bool()
            post_seconds += post * env.step_dt
            duration += active * env.step_dt
            total_return += reward.view(-1) * active
            fields = {k: m[k] for k in (*PAIR_FIELDS, *SCALAR_FIELDS)}
            if "foot_tilt_deg" in m:
                fields.update({k: m[k] for k in SOLE_FIELDS})
            for name, value in fields.items():
                v = value.float()
                mask = post.reshape(-1, *([1] * (v.ndim - 1)))
                if name not in peak:
                    peak[name] = torch.full_like(v, -torch.inf)
                    minimum[name] = torch.full_like(v, torch.inf)
                    final[name] = torch.zeros_like(v)
                peak[name] = torch.where(mask, torch.maximum(peak[name], v), peak[name])
                minimum[name] = torch.where(mask, torch.minimum(minimum[name], v), minimum[name])
                terminal = active & done.bool().view(-1)
                final[name] = torch.where(
                    terminal.reshape(-1, *([1] * (v.ndim - 1))), v, final[name]
                )
            bearing = torch.cat((m["knee_top_N"] > 8.0, m["foot_top_N"] > 20.0), dim=1)
            dwell = torch.where(post[:, None] & bearing, dwell + env.step_dt, 0.0)
            max_dwell = torch.maximum(max_dwell, dwell)
            if trace is not None and bool(active[record_env]):
                row = {key: value[record_env].tolist() for key, value in fields.items()}
                if ground is not None:
                    row.update({key: m[key][record_env].item() for key in COURSE_FIELDS})
                    for key in (
                        "course_time_s",
                        "heading_error_deg",
                        "walk_step_switch",
                        "foot_ground_patch_N",
                        "foot_ground_slip_m_s",
                        "approach_accepted",
                        "forearm_top_load_N",
                        "arm_support_top_N",
                        "wrist_tabletop_load_N",
                        "wrist_forbidden",
                        "wrist_platform_force_N",
                        "root_angular_speed_rad_s",
                    ):
                        if key in m:
                            row[key] = m[key][record_env].tolist()
                    if "nonfoot_support_checked" in m:
                        checked = bool(m["nonfoot_support_checked"][record_env])
                        row["nonfoot_support_checked"] = checked
                        row["nonfoot_platform_force_N"] = (
                            m["nonfoot_platform_force_N"][record_env].item() if checked else None
                        )
                row.update(
                    environment=record_env,
                    t_s=(step + 1) * env.step_dt,
                    reference_frame=int(m["reference_frame"][record_env]),
                    root_pose=m["root_pose"][record_env].tolist(),
                    joint_pos=m["joint_pos"][record_env].tolist(),
                )
                trace.write(json.dumps(row, allow_nan=False) + "\n")
            active &= ~done.bool().view(-1)
            if not active.any():
                break
    if active.any():
        raise RuntimeError(
            f"{int(active.sum())} full-ascent attempts exceeded the configured environment horizon"
        )
    # Attempts ending before handoff have no post-handoff maximum/minimum.
    has_post = torch.isfinite(peak["stage"])
    for values in (peak, minimum):
        for name, v in values.items():
            values[name] = torch.nan_to_num(v, posinf=0, neginf=0)

    def export(values):
        return {k: v.cpu().tolist() for k, v in values.items()}

    peaks, minima, terminal = export(peak), export(minimum), export(final)
    dwells = max_dwell.cpu().tolist()
    post_ids = has_post.cpu().tolist()
    accepted, successes = accepted.tolist(), successes.tolist()
    durations, returns = duration.tolist(), total_return.tolist()
    episodes = []
    for i in range(n):
        episodes.append(
            {
                "environment": i,
                "entry_accepted": accepted[i],
                "success": successes[i],
                "duration_s": durations[i],
                "return": returns[i],
                "max_knee_then_foot_dwell_s_lr": dwells[i],
                "post_peak": {k: v[i] for k, v in peaks.items()} if post_ids[i] else None,
                "post_min": {k: v[i] for k, v in minima.items()} if post_ids[i] else None,
                "terminal": {k: v[i] for k, v in terminal.items()},
            }
        )
    summary = {
        "stance": "flat-foot" if "foot_tilt_deg" in peak else "legacy",
        "episodes": n,
        "entry_accepted": sum(accepted),
        "standing_successes": sum(successes),
        "both_feet_bearing": int(both_feet.sum()),
        "any_foot_bearing": int(any_foot.sum()),
        "both_knees_bearing": int(both_knees.sum()),
        "mean_peak_stand_hold_s": sum(peaks["stand_hold_s"]) / n,
        "survived_recovery_to_timeout_or_success": int(survived.sum()),
        "mean_post_duration_s": float(post_seconds.mean()),
        "mean_peak_hip_above_top_m_after_handoff": (
            sum(z for z, ok in zip(peaks["hip_above_top_m"], post_ids) if ok) / sum(post_ids)
            if any(post_ids)
            else None
        ),
        "mean_return": sum(returns) / n,
    }
    if ground is not None:
        summary.update(ground.summary())
        metrics = zip(
            ground.braced.tolist(),
            ground.flip_drift.tolist(),
            ground.walk_advance.tolist(),
            ground.switches.tolist(),
            strict=True,
        )
        for episode, (braced, drift, distance, switches) in zip(episodes, metrics, strict=True):
            episode.update(
                ground_braced=braced,
                peak_flip_drift_m=drift,
                walk_advance_m=distance,
                alternating_landings=switches,
            )
    return {"summary": summary, "episodes": episodes}


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    # Isaac inspects parse_known_args before installing its device and help options.
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("mode", choices=("evaluate", "play", "train"))
    parser.add_argument("--entry-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ground-trajectory",
        type=Path,
        help="Train/evaluate from standing on the ground, using trajectory.npz and sibling pose_trace.npz",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Previously saved whole-ascent residual policy"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for reports and checkpoints"
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--controller",
        choices=("standard", "knee", "crawl"),
        help="Restore checkpoint controller by default; fresh runs default to standard",
    )
    parser.add_argument(
        "--support-reward",
        action="store_true",
        help="Shape body raising after contact; restored from residual checkpoints",
    )
    parser.add_argument(
        "--stance",
        choices=("flat-foot", "legacy"),
        help="Standing criterion. Fresh runs use flat-foot; checkpoints restore their original "
        "criterion unless a value is explicitly supplied.",
    )
    settings_group = parser.add_mutually_exclusive_group()
    settings_group.add_argument(
        "--control-parameters",
        type=json.loads,
        help="JSON object of named recovery controller settings, applied to all environments",
    )
    settings_group.add_argument(
        "--control-parameters-file",
        type=Path,
        help="JSON of named settings or a saved controller_parameters_by_env.json table",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the selected environment from start through termination",
    )
    parser.add_argument("--record-env", type=int, help="Record this environment index (default 0)")
    args = parser.parse_args()
    if args.record_env is None:
        args.record_env = 0
    else:
        args.record = True
    if not 0 <= args.record_env < args.num_envs:
        parser.error("--record-env must be in [0, --num-envs)")
    if (
        min(args.num_envs, args.iterations, args.eval_interval, args.rollout_steps) < 1
        or not 0 < args.learning_rate < 1
    ):
        parser.error("Training and environment counts and learning rate must be positive")
    source = args.entry_checkpoint.expanduser().resolve()
    request, weights, files, hashes = source_inputs(source)
    saved = None
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    previous_contract = normalize_contract(
        (saved.get("infos") or {}).get("contract", {}) if saved else {}
    )
    if "ground_trajectory_sha256" in previous_contract and args.ground_trajectory is None:
        manifest = args.checkpoint.parent / "contract.json"
        if manifest.exists():
            location = json.loads(manifest.read_text()).get("ground_trajectory")
            if location:
                args.ground_trajectory = Path(location)
        if args.ground_trajectory is None:
            parser.error("Supply --ground-trajectory for this ground-start checkpoint")
    if args.ground_trajectory:
        if saved and "ground_trajectory_sha256" not in previous_contract:
            parser.error(
                "Ground-start training needs a fresh residual or a ground-start checkpoint"
            )
        if (
            args.controller
            or args.control_parameters is not None
            or args.control_parameters_file
            or args.support_reward
            or args.stance == "legacy"
        ):
            parser.error(
                "Ground-start mode uses its own controls, rewards and flat-foot standing criterion"
            )
        args.ground_trajectory = args.ground_trajectory.expanduser().resolve()
        hashes["ground_trajectory_sha256"] = digest(args.ground_trajectory)
        hashes["ground_pose_trace_sha256"] = digest(
            args.ground_trajectory.with_name("pose_trace.npz")
        )
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "request.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    app = AppLauncher(args).app
    env = None
    try:
        from .mount_env import MountEnv
        from rsl_rl.runners import OnPolicyRunner

        cfg = knee_entry_config(
            request, files, seed=args.seed, num_envs=args.num_envs, device=args.device
        )
        if args.ground_trajectory:
            from .ground_mount import GroundMountEnv

            env = GroundMountEnv(cfg, trajectory=args.ground_trajectory)
        else:
            env = MountEnv(cfg)
        stance = args.stance or previous_contract.get("stance", "flat-foot")
        env.mount_task.stance = stance
        controller = (
            "ground-trajectory"
            if args.ground_trajectory
            else (args.controller or previous_contract.get("controller", "standard"))
        )
        if not args.ground_trajectory:
            env.mount_control = select_controller(env, controller)
        support_reward = args.support_reward or "support_reward_sha256" in previous_contract
        if support_reward:
            from .mount_support import SupportReward

            env.mount_task = SupportReward(env.mount_task)
        settings = args.control_parameters
        parameter_file = args.control_parameters_file
        if (
            settings is None
            and parameter_file is None
            and "per_env_parameters_sha256" in previous_contract
        ):
            parameter_file = args.checkpoint.parent / "controller_parameters_by_env.json"
        if parameter_file is not None:
            settings = json.loads(parameter_file.read_text())
        if settings is None:
            settings = previous_contract.get("control_parameters", {})
        if not args.ground_trajectory:
            set_controller_parameters(env.mount_control, settings)
        env.entry_policy = frozen_entry(weights, source.parent / "ppo_config.json", env.device)
        wrapped = LegacyRslAdapter(env)
        config = runner_config(
            {
                "num_steps_per_env": args.rollout_steps,
                "policy": {
                    "actor_hidden_dims": [128, 128],
                    "critic_hidden_dims": [128, 128],
                    "init_noise_std": 0.10,
                },
                "algorithm": {
                    "learning_rate": args.learning_rate,
                    "num_mini_batches": min(4, args.num_envs * args.rollout_steps),
                },
            }
        )
        # These are externally assessed blocks. Only publish checkpoints paired with the evaluation.
        runner = OnPolicyRunner(wrapped, config, log_dir=None, device=env.device)
        runner.logger_type = "tensorboard"
        runner.disable_logs = True
        wrapped.initial_normalizers = (runner.obs_normalizer, runner.privileged_obs_normalizer)
        source_dir = Path(__file__).parent
        contract = dict(
            schema="ground_mount_residual_v1" if args.ground_trajectory else "mount_residual_v2",
            **hashes,
            controller=controller,
            mount_env_sha256=digest(source_dir / "mount_env.py"),
            mount_task_sha256=digest(source_dir / "mount_task.py"),
            stance=stance,
        )
        if support_reward:
            contract["support_reward_sha256"] = digest(source_dir / "mount_support.py")
        if args.ground_trajectory:
            from .ground_locomotion import GAIT_MODEL
            from .walk import Footsteps

            for module in (
                "ground_mount",
                "ground_mount_reward",
                "ground_trajectory",
                "ground_locomotion",
                "ground_support",
                "environment",
                "isaac_knee_entry",
                "robot_cfg",
            ):
                contract[module + "_source_sha256"] = digest(source_dir / (module + ".py"))
            contract["gait_weights_sha256"] = digest(GAIT_MODEL)
            contract["footsteps_source_sha256"] = hashlib.sha256(
                inspect.getsource(Footsteps).encode()
            ).hexdigest()
        else:
            contract["controller_sha256"] = controller_digest(env.mount_control)
            columns = controller_columns(env.mount_control)
            params = torch.stack(list(columns.values()), dim=1).detach().cpu()
            names = list(columns)
            if (params == params[0]).all():
                contract["control_parameters"] = dict(zip(names, params[0].tolist()))
            else:
                by_env = out / "controller_parameters_by_env.json"
                write_json(by_env, {"names": list(names), "values": params.tolist()})
                contract["per_env_parameters_sha256"] = digest(by_env)
        write_json(
            out / "contract.json",
            dict(
                **contract,
                **(
                    {"ground_trajectory": str(args.ground_trajectory)}
                    if args.ground_trajectory
                    else {}
                ),
                entry_checkpoint=str(source),
                entry_iteration=weights["infos"].get("completed_iterations"),
                frozen_entry=True,
                reference=str(files["reference"]),
                step_dt_s=env.step_dt,
                environment_horizon_steps=env.max_episode_length,
            ),
        )
        if args.checkpoint:
            validate_mount_contract(previous_contract, contract, args.stance)
            runner.alg.policy.load_state_dict(saved["model_state_dict"])
            runner.obs_normalizer.load_state_dict(saved["obs_norm_state_dict"])
            runner.privileged_obs_normalizer.load_state_dict(
                saved["privileged_obs_norm_state_dict"]
            )
            runner.current_learning_iteration = int(saved["iter"])
            if args.mode == "train":
                runner.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = args.learning_rate
        else:
            with torch.no_grad():
                runner.alg.policy.actor[-1].weight.zero_()
                runner.alg.policy.actor[-1].bias.zero_()
        if args.mode == "train":
            train(runner, wrapped, args, out, contract)
        else:
            report = assess(
                runner,
                wrapped,
                out if args.record or args.mode == "play" else None,
                camera=args.mode == "play",
                record_env=args.record_env,
            )
            write_json(out / "evaluation.json", report)
            print("MOUNT_EVAL", json.dumps(report["summary"]), flush=True)
        write_json(out / "completion.json", {"status": "COMPLETED", "mode": args.mode})
    except Exception:
        write_json(out / "error.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def checkpoint_score(summary):
    if "ground_braced" in summary:
        walk = summary["ground_walk_quality"] or 0.0
        advance = summary["ground_mean_walk_advance_m"]
        return (
            summary["standing_successes"],
            summary["mean_peak_stand_hold_s"],
            summary["entry_accepted"],
            summary["ground_rise_quality"] or 0.0,
            summary["ground_braced"],
            summary["ground_with_alternating_steps"],
            max(0.0, advance) * walk,
            advance - summary["ground_mean_peak_flip_drift_m"],
        )
    return (
        summary["standing_successes"],
        summary["both_feet_bearing"],
        summary["mean_peak_stand_hold_s"],
        summary["any_foot_bearing"],
        summary["both_knees_bearing"],
        summary["mean_return"],
    )


def train(runner, wrapped, args, out, contract):
    initial = int(runner.current_learning_iteration)
    completed = initial
    best = None
    while True:
        final = completed == initial + args.iterations
        report = assess(runner, wrapped, out if args.record else None, record_env=args.record_env)
        if args.record:
            trace_path = out / f"trace_{completed:05d}.jsonl"
            (out / "trace.jsonl").replace(trace_path)
            if final:
                shutil.copyfile(trace_path, out / "trace.jsonl")
        summary = report["summary"]
        info = {"contract": contract, "completed_iterations": completed, "evaluation": summary}
        runner.current_learning_iteration = completed
        write_json(out / f"evaluation_{completed:05d}.json", report)
        score = checkpoint_score(summary)
        if best is None or score > best:
            best = score
            save_checkpoint_atomically(runner, out / "best.pt", infos=info)
        save_checkpoint_atomically(runner, out / "latest.pt", infos=info)
        save_checkpoint_atomically(runner, out / f"checkpoint_{completed:05d}.pt", infos=info)
        with (out / "history.jsonl").open("a") as stream:
            print(json.dumps({"iteration": completed, **summary}), file=stream)
        print("MOUNT_CHECK", completed, json.dumps(summary), flush=True)
        if final:
            break
        next_interval = args.eval_interval - completed % args.eval_interval
        block = min(next_interval, initial + args.iterations - completed)
        with torch.inference_mode():
            if hasattr(wrapped.env, "stage_training"):
                wrapped.env.stage_training = True
            wrapped.obs, _ = wrapped.env.reset()
        end = completed + block
        while completed < end:
            steps = min(10, end - completed)
            runner.learn(steps, init_at_random_ep_len=False)
            completed += steps
            runner.current_learning_iteration = completed
            progress = {
                "completed_iterations": completed,
                "training_steps_this_run": (completed - initial)
                * args.num_envs
                * args.rollout_steps,
            }
            if hasattr(wrapped.env, "training_stage_steps"):
                progress["training_stage_steps"] = dict(
                    zip(
                        ("approach", "frozen_knee_entry", "recovery"),
                        wrapped.env.training_stage_steps.tolist(),
                        strict=True,
                    )
                )
            write_json(out / "progress.json", progress)
            print("MOUNT_TRAIN", json.dumps(progress), flush=True)
            if completed < end:
                save_checkpoint_atomically(
                    runner,
                    out / "progress.pt",
                    infos={"contract": contract, "completed_iterations": completed},
                )


if __name__ == "__main__":
    main()
