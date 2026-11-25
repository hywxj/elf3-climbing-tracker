"""Train and independently score the single ground-to-knee contact planner/tracker."""

import argparse
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from .paths import OUTPUT
from .reference import digest
from .runner import LegacyRslAdapter, runner_config
from .training_control import save_checkpoint_atomically

REFERENCE = OUTPUT / "platform_contacts_20261120_070/isaac_v4_070_surface_reference/reference.npz"


def write_json(path, value):
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@torch.inference_mode()
def evaluate(env, actor, seed, path, record=False, *, ground=True):
    """Count exactly the first episode in each lane; read pre-reset contact measurements."""
    env.stage_training = False
    obs, _ = env.reset(seed=seed)
    actor.eval()
    size = env.num_envs
    live = torch.ones(size, device=env.device, dtype=torch.bool)
    flags = {
        k: torch.zeros_like(live)
        for k in (
            "ground_start",
            "walked_to_edge",
            "bilateral_braced",
            "push_observed",
            "takeoff_observed",
            "right_knee_patch_touched",
            "knee_reached",
            "success",
        )
    }
    peaks = {
        k: torch.zeros(size, device=env.device)
        for k in (
            "walk_switches",
            "advance_m",
            "knee_hold_s",
            "right_knee_top_N",
            "right_knee_load_bodyweight",
        )
    }
    lengths = torch.zeros_like(live, dtype=torch.long)
    walked_push = torch.zeros_like(live)
    walked_takeoff = torch.zeros_like(live)
    walked_knee_touch = torch.zeros_like(live)
    ground_hold_peak = torch.zeros_like(peaks["knee_hold_s"])
    poses, joints, measurements = [], [], []
    trace_columns = (
        "walk_switches",
        "advance_m",
        "bilateral_braced",
        "right_knee_top_N",
        "right_knee_inside_edge_m",
        "knee_hold_s",
        "right_knee_load_bodyweight",
        "success",
    )
    for _ in range(int(env.max_episode_length) + 2):
        obs, _, term, trunc, _ = env.step(actor(obs["policy"]))
        m = env.last_metrics
        for name in flags:
            flags[name] |= live & m[name].bool()
        walked_push |= live & flags["walked_to_edge"] & m["bilateral_push"].bool()
        walked_takeoff |= live & walked_push & m["takeoff_observed"].bool()
        walked_knee_touch |= live & walked_takeoff & m["right_knee_patch_touched"].bool()
        ground_hold_peak = torch.where(
            live & flags["ground_start"] & walked_takeoff,
            torch.maximum(ground_hold_peak, m["knee_hold_s"]),
            ground_hold_peak,
        )
        for name, previous in peaks.items():
            peaks[name] = torch.where(live, torch.maximum(previous, m[name].float()), previous)
        lengths += live.long()
        if record:
            poses.append(m["root_pose"].detach().cpu().numpy().copy())
            joints.append(m["joint_pos"].detach().cpu().numpy().copy())
            measurements.append(
                torch.stack([m[k].float() for k in trace_columns], -1).cpu().numpy()
            )
        live &= ~(term | trunc)
        if not live.any():
            break
    if live.any():
        raise RuntimeError(f"{int(live.sum())} evaluation lanes exceeded the episode budget")
    counts = {name: int(value.sum()) for name, value in flags.items()}
    if counts["ground_start"] != (size if ground else 0):
        raise RuntimeError("Unexpected start in evaluation cohort")
    result = dict(
        seed=seed,
        episodes=size,
        counts=counts,
        after_walk={
            "push": int(walked_push.sum()),
            "takeoff": int(walked_takeoff.sum()),
            "knee_touch_after_takeoff": int(walked_knee_touch.sum()),
        },
        reached_0_20_s=int((peaks["knee_hold_s"] >= 0.20 - 1e-6).sum()),
        required_hold_s=round(
            getattr(getattr(env, "milestones", None), "hold_steps", 10) * env.step_dt, 6
        ),
        with_alternating_step=int((peaks["walk_switches"] > 0).sum()),
        peak={k: float(v.max()) for k, v in peaks.items()},
        median={k: float(v.median()) for k, v in peaks.items()},
        mean_episode_s=float(lengths.float().mean()) * env.step_dt,
        episode_limit_s=round(float(env.max_episode_length) * env.step_dt, 6),
        step_dt=float(env.step_dt),
        terrain_height_m=0.70,
        start="upright_ground" if ground else "archived_knee_start",
        minimum_knee_force_N=float(env.knee_contacts.total_mass[0]) * 9.81 * 0.15,
    )
    if ground:
        result["full_course_hold_counts"] = {
            f"{seconds:.2f}s": int((ground_hold_peak >= seconds - 1e-6).sum())
            for seconds in (0.20, 0.50, 1.0, 1.5, 2.0)
            if seconds <= result["required_hold_s"] + 1e-6
        }
    if record:
        # Pick a representative of the most advanced measured event, then report its lane ID.
        rank = [flags["success"], walked_knee_touch, walked_takeoff, walked_push]
        rank.extend(
            flags[k]
            for k in (
                "walked_to_edge",
                "knee_reached",
                "right_knee_patch_touched",
                "bilateral_braced",
            )
        )
        rank += [peaks["knee_hold_s"], peaks["walk_switches"], peaks["advance_m"]]
        keys = [x.cpu().tolist() for x in rank]
        idx = max(range(size), key=lambda i: tuple(k[i] for k in keys))
        n = int(lengths[idx])
        np.savez_compressed(
            path.with_suffix(".npz"),
            dt=float(env.step_dt),
            lane=idx,
            root_pose=np.stack(poses[:n])[:, idx],
            joint_pos=np.stack(joints[:n])[:, idx],
            metrics=np.stack(measurements[:n])[:, idx],
            metric_names=trace_columns,
            joint_names=env.reference_np["joint_names"],
            start=result["start"],
        )
        result["recorded_lane"] = idx
        result["recorded_seconds"] = n * float(env.step_dt)
    write_json(path, result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


class ContactRslAdapter(LegacyRslAdapter):
    @torch.no_grad()
    def step(self, actions):
        _, logits, points = self.actor.action_and_plan(self.obs["policy"])
        self.env.commit_plan(logits, points, yaw_frame=getattr(self.actor, "plan_yaw_frame", False))
        return super().step(actions)


def make_runner(env, args, actor):
    from rsl_rl.runners import OnPolicyRunner

    adapter = ContactRslAdapter(env)
    config = runner_config(
        {
            "empirical_normalization": False,
            "num_steps_per_env": args.rollout_steps,
            "policy": {"init_noise_std": args.noise, "critic_hidden_dims": [192, 128]},
            "algorithm": {"learning_rate": args.learning_rate, "gamma": 0.995},
        }
    )
    batch_size = env.num_envs * args.rollout_steps
    if hasattr(actor, "point_spec"):
        # Tensor size grows with minibatch × bodies × scan cells for the contact planner.
        batch_slices = math.ceil(batch_size / 1536)
        config["algorithm"]["num_mini_batches"] = max(4, 1 << (batch_slices - 1).bit_length())
    config["algorithm"]["num_mini_batches"] = min(
        batch_size, config["algorithm"]["num_mini_batches"]
    )
    runner = OnPolicyRunner(adapter, config, device=env.device)
    runner.disable_logs = True
    runner.logger_type = "tensorboard"
    runner.alg.policy.actor = actor.to(env.device)
    adapter.actor = runner.alg.policy.actor
    runner.alg.optimizer = torch.optim.Adam(runner.alg.policy.parameters(), lr=args.learning_rate)
    return runner


class DemonstrationReplay:
    """Rehearse physical drive targets and contact labels between PPO iterations."""

    def __init__(self, paths, device, seed, *, edge=False, point=False):
        from .contact_learning import load_examples

        examples = (
            load_examples(paths, edge=edge, point=True)
            if point
            else load_examples(paths, edge=edge)
        )
        self.train = [column.to(device) for column in examples["train"]]
        self.weights = examples["weights"].to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def update(self, runner):
        from .contact_learning import contact_imitation_loss

        actor = runner.alg.policy.actor
        idx = torch.multinomial(self.weights, 384, replacement=True, generator=self.generator)
        loss = contact_imitation_loss(actor, [column[idx] for column in self.train])
        runner.alg.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        runner.alg.optimizer.step()


def run(args, out):
    from .environment import TrackerCfg
    from .contact_tracker_env import ContactTrackerEnv
    from .contact_learning import ContactActor
    from .contact_gait_actor import GaitContactActor

    cfg = TrackerCfg()
    cfg.seed, cfg.sim.device, cfg.scene.num_envs = args.seed, args.device, args.num_envs
    cfg.reference_path, cfg.segment = str(args.reference), "knee_entry"
    cfg.knee_wrist_contact, cfg.knee_support_rule = "tabletop", "assisted-push"
    cfg.random_start = cfg.curriculum_enabled = False
    cfg.preload_example = cfg.knee_transfer_guide = None
    weights = torch.load(args.weights, weights_only=True, map_location="cpu")
    keys = weights.get("model_state_dict", weights)
    gait = "backbone.0.weight" in keys or "actor.backbone.0.weight" in keys
    edge = int(keys.get("edge_spec", keys.get("actor.edge_spec", 0)))
    point = int(keys.get("point_spec", keys.get("actor.point_spec", 0)))
    if point:
        from .contact_point_actor import PointContactActor

        actor = PointContactActor()
        env = ContactTrackerEnv(cfg, point_observation=True, edge_observation=edge)
    elif edge:
        from .contact_edge import EdgeContactActor, EdgeGaitContactActor

        actor = EdgeGaitContactActor() if gait else EdgeContactActor()
        env = ContactTrackerEnv(cfg, edge_observation=edge)
    else:
        actor = GaitContactActor() if gait else ContactActor()
        env = ContactTrackerEnv(cfg)
    try:
        env.cfg.episode_length_s = args.episode_seconds
        env.milestones.hold_steps = math.ceil(args.knee_hold_seconds / env.step_dt - 1e-8)
        original_examples = weights.get("infos", {}).get("contract", {}).get("cloning_examples")
        if original_examples is None and args.weights.with_suffix(".json").is_file():
            original_examples = json.loads(args.weights.with_suffix(".json").read_text()).get(
                "samples"
            )
        if args.examples and args.iterations:
            env.load_demonstration_starts(args.examples, cloning_split_paths=original_examples)
        runner = make_runner(env, args, actor)
        actor = runner.alg.policy.actor
        schema = {
            "task": "single_contact_tracker_ground_to_right_knee_v1",
            "cloning_examples": original_examples,
            "required_knee_hold_s": round(env.milestones.hold_steps * env.step_dt, 6),
            "reference_sha256": digest(args.reference),
            "implementation": {
                name: digest(Path(__file__).with_name(name + ".py"))
                for name in (
                    "contact_tracker_env",
                    "contact_tracker_task",
                    "contact_learning",
                    "contact_examples",
                    "isaac_knee_entry",
                    "contact_plan_reward",
                )
            },
        }
        if gait:
            schema["implementation"]["contact_gait_actor"] = digest(
                Path(__file__).with_name("contact_gait_actor.py")
            )
        if edge:
            schema["implementation"]["contact_edge"] = digest(
                Path(__file__).with_name("contact_edge.py")
            )
        if point:
            schema["implementation"]["contact_point_actor"] = digest(
                Path(__file__).with_name("contact_point_actor.py")
            )
        if "model_state_dict" in weights:
            previous = weights["infos"]["contract"]
            if any(previous.get(k) != schema[k] for k in ("task", "reference_sha256")):
                raise ValueError("Incompatible single-tracker checkpoint")
            changed = {
                key: value
                for key, value in schema["implementation"].items()
                if previous.get("implementation", {}).get(key) != value
            }
            if changed:
                write_json(
                    out / "implementation_changes.json",
                    {"source": previous.get("implementation", {}), "current": changed},
                )
            runner.alg.policy.load_state_dict(weights["model_state_dict"])
            runner.alg.optimizer.load_state_dict(weights["optimizer_state_dict"])
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate
            runner.current_learning_iteration = weights["iter"]
        else:
            actor.load_state_dict(weights)
        write_json(out / "contract.json", {**schema, "input_weights_sha256": digest(args.weights)})
        start = runner.current_learning_iteration
        stop = start + args.iterations
        best = None
        demos = (
            DemonstrationReplay(args.examples, env.device, args.seed, edge=edge, point=point)
            if args.iterations and args.examples
            else None
        )
        while True:
            it = runner.current_learning_iteration
            report = evaluate(
                env, actor, args.eval_seed, out / f"ground_{it:05d}.json", args.record
            )
            if args.knee_diagnostic:
                env.load_demonstration_starts([args.knee_diagnostic], initial_knee=True)
                evaluate(
                    env,
                    actor,
                    args.eval_seed,
                    out / f"knee_seed_{it:05d}.json",
                    args.record,
                    ground=False,
                )
                env.diagnostic_start = None
            if args.iterations:
                candidate = out / f"checkpoint_{it:05d}.pt"
                save_checkpoint_atomically(
                    runner, candidate, infos={"contract": schema, "ground": report["counts"]}
                )
                score = (
                    report["counts"]["success"],
                    report["after_walk"]["knee_touch_after_takeoff"],
                    report["after_walk"]["takeoff"],
                    report["after_walk"]["push"],
                    report["counts"]["walked_to_edge"],
                    report["with_alternating_step"],
                )
                if best is None or score > best:
                    best = score
                    temp = out / ".best.pt"
                    temp.symlink_to(candidate.name)
                    os.replace(temp, out / "best.pt")
            if it == stop:
                break
            env.stage_training = not args.ground_only
            with torch.inference_mode():
                runner.env.obs, _ = env.reset(seed=args.seed + it)
            count = min(args.eval_interval, stop - it)
            print(f"PPO {it + 1}..{it + count} ({args.num_envs} lanes)", flush=True)
            for _ in range(count):
                runner.learn(1, checkpoint_callback=lambda _: None)
                if demos is not None:
                    for _ in range(args.demo_updates):
                        demos.update(runner)
                completed = runner.current_learning_iteration
                if completed % 10 == 0 or completed == stop:
                    write_json(
                        out / "progress.json",
                        {
                            "iteration": completed,
                            "target_iteration": stop,
                            "last_evaluated_iteration": it,
                            "last_ground_successes": report["counts"]["success"],
                            "ground_episodes": report["episodes"],
                        },
                    )
        write_json(
            out / "completion.json",
            {
                "status": "COMPLETED",
                "iteration": stop,
                "ground_successes": report["counts"]["success"],
            },
        )
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="BC student state_dict or student PPO checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument(
        "--knee-diagnostic",
        type=Path,
        help="Also assess this student from the old teacher's physical start",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        nargs="+",
        help="Rehearse the training split of these physical examples during PPO",
    )
    parser.add_argument(
        "--iterations", type=int, default=0, help="Additional PPO iterations; 0 evaluates only"
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--knee-hold-seconds",
        type=float,
        default=0.20,
        help="Required continuous knee support before success and termination",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=12.0,
        help="Time allowed for each uninterrupted attempt",
    )
    parser.add_argument("--seed", type=int, default=427)
    parser.add_argument("--eval-seed", type=int, default=7241)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--demo-updates",
        type=int,
        default=8,
        help="Measured-action rehearsal minibatches per PPO iteration",
    )
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--noise", type=float, default=0.035, help="PPO exploration std, radians")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--ground-only", action="store_true", help="Use only ground resets even during PPO"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
        parser.error("Episode duration must be positive and finite")
    if (
        not math.isfinite(args.knee_hold_seconds)
        or not 0.20 <= args.knee_hold_seconds <= args.episode_seconds
    ):
        parser.error("Knee hold must be at least 0.20 s and within the episode duration")
    if args.iterations < 0 or min(args.num_envs, args.eval_interval, args.rollout_steps) <= 0:
        parser.error("Invalid episode or PPO counts")
    if args.demo_updates < 0:
        parser.error("Demo updates must be nonnegative")
    if not args.weights.is_file():
        parser.error(f"Missing student weights: {args.weights}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "request.json", json.loads(json.dumps(vars(args), default=str)))
    app = AppLauncher(args).app
    try:
        run(args, out)
    except BaseException:
        write_json(out / "error.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
