"""Train or evaluate flat walking without visual observations or Hiking imports."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .paths import OUTPUT
from .walk_data import (
    prepare_walk_manifest,
    WristMaskedPrior,
    IGNORED_AMP_COLUMNS,
    WRIST_NAME,
    WRIST_TARGET,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tienkung-root",
        type=Path,
        default=Path(os.environ.get("TIENKUNG_ROOT", Path.home() / "TienKung-Lab")),
    )
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate", "play"), default="smoke")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    repo = args.tienkung_root.expanduser().resolve()
    if not args.output.is_relative_to(OUTPUT.resolve()) or args.output == OUTPUT.resolve():
        parser.error("--output must be a NEW subdirectory of ELF3_RUNTIME_ROOT/runs")
    if args.output.exists() or args.num_envs < 1 or args.iterations < 1:
        parser.error("Use a new output directory and positive counts")
    if args.mode in ("evaluate", "play") and not args.checkpoint:
        parser.error("Evaluation/play requires a blind-walk checkpoint")
    if not (repo / "legged_lab/envs/elf3/elf3_env.py").is_file():
        parser.error("Local TienKung-Lab source is missing")
    # The existing project resolves some resources relative to its root.
    sys.path.insert(0, str(repo))
    args.output.mkdir(parents=True)
    write_json(
        args.output / "request.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    app = None
    env = None
    try:
        manifest = prepare_walk_manifest(
            repo / "logs/data_preparation/elf3_gmr_v1/manifest.json", args.output / "walk_data"
        )
        os.chdir(repo)
        app = AppLauncher(args).app
        import torch
        from rsl_rl.runners import AmpOnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from .walk_env import BlindWalkEnv, make_configs
        from .gripper import specification

        cfg, agent = make_configs(repo, args.num_envs, args.device, args.seed, manifest)
        if args.mode in ("evaluate", "play"):
            cfg.noise.add_noise = False
            cfg.scene.max_episode_length_s = 12.0
        env = BlindWalkEnv(cfg, headless=args.headless)
        obs, extras = env.get_observations()
        if obs.shape != (args.num_envs, 960) or extras["observations"]["critic"].shape != (
            args.num_envs,
            1010,
        ):
            raise ValueError("Unexpected observations: blind actor must have 960, critic 1010")
        if any(n in env.scene.sensors for n in ("height_scanner", "depth_camera", "lidar")):
            raise ValueError("A visual sensor was unexpectedly created")
        from pxr import UsdPhysics

        stage = env.sim.stage
        envelopes = [p for p in stage.Traverse() if p.GetName() == "ClimbingGripperEnvelope"]
        if len(envelopes) != args.num_envs or not all(
            stage.GetPrimAtPath(str(p.GetPath()) + "/geometry/mesh").HasAPI(UsdPhysics.CollisionAPI)
            for p in envelopes
        ):
            raise ValueError(f"Missing gripper collision geometry: {len(envelopes)} envelopes")
        joint_names = list(env.robot.joint_names)
        contract = {
            "schema": "elf3_blind_walk_flip_v1",
            "actor_dim": 960,
            "critic_dim": 1010,
            "history": 10,
            "action_joint_names": joint_names,
            "action_scales": env.action_scale.tolist(),
            "default_joint_pos": env.robot.data.default_joint_pos[0].tolist(),
            "soft_joint_pos_limits": env.robot.data.soft_joint_pos_limits[0].tolist(),
            "dt": env.step_dt,
            "gripper": specification(),
            "terrain": "plane",
            "sensors": list(env.scene.sensors),
            "gripper_collision_instances": len(envelopes),
            "amp_ignored_columns": list(IGNORED_AMP_COLUMNS),
            "source_files": {
                str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    repo / "legged_lab/envs/elf3/elf3_env.py",
                    repo / "legged_lab/envs/elf3/walk_cfg.py",
                )
            },
            "standing_checkpoint_loaded": False,
            "scope": "Starts already flipped; no flip-to-walk handover, platform or climbing acceptance",
        }
        write_json(args.output / "contract.json", contract)
        train_cfg = agent.to_dict()
        dump_yaml(str(args.output / "env.yaml"), cfg)
        dump_yaml(str(args.output / "agent.yaml"), agent)
        runner = AmpOnPolicyRunner(env, deepcopy(train_cfg), str(args.output), args.device)
        runner.alg.amp_data = WristMaskedPrior(runner.alg.amp_data)
        runner.motion_prior_provenance["ignored_amp_columns"] = list(IGNORED_AMP_COLUMNS)
        wrist = joint_names.index(WRIST_NAME)
        if args.checkpoint:
            previous = json.loads((args.checkpoint.resolve().parent / "contract.json").read_text())
            for key in (
                "schema",
                "action_joint_names",
                "action_scales",
                "default_joint_pos",
                "soft_joint_pos_limits",
                "actor_dim",
                "dt",
            ):
                if previous.get(key) != contract[key]:
                    raise ValueError(f"Incompatible walking checkpoint: {key}")
            runner.load(str(args.checkpoint.resolve()), load_optimizer=args.mode == "train")
        else:
            # Neutral position targets at initialization, all 29 outputs trainable.
            with torch.no_grad():
                runner.alg.policy.actor[-1].weight.mul_(0.01)
                runner.alg.policy.actor[-1].bias.zero_()
                runner.alg.policy.std[wrist] = 0.05
        initial = {k: v.clone() for k, v in runner.alg.policy.actor.state_dict().items()}
        if args.mode in ("smoke", "train"):
            iterations = 2 if args.mode == "smoke" else args.iterations
            runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
            if not all(torch.isfinite(v).all() for v in runner.alg.policy.state_dict().values()):
                raise ValueError("Nonfinite learned parameters")
            changed = sum(
                not torch.equal(v, initial[k])
                for k, v in runner.alg.policy.actor.state_dict().items()
            )
            if not changed:
                raise ValueError("Optimizer did not update actor")
            runner.save(str(args.output / "latest.pt"))
            result = {
                "integration": "PASS",
                "iterations": iterations,
                "changed_actor_tensors": changed,
                "walking_capability": "NOT RUN",
                "handover": "NOT RUN",
                "full_climb": "NOT RUN",
            }
        else:
            policy = runner.get_inference_policy(device=args.device)
            result = evaluate(env, policy, app, play=args.mode == "play")
        write_json(args.output / "result.json", result)
        write_json(args.output / "completion.json", {"status": "COMPLETED", "mode": args.mode})
        print("WALK_RESULT " + json.dumps(result), flush=True)
        if runner.writer:
            runner.writer.close()
    except BaseException as exc:
        write_json(
            args.output / "error.json",
            {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()},
        )
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


def evaluate(env, policy, app, play=False):
    """Deterministic stop 2s -> .2 m/s 4s -> stop 4s; failures never count as success."""
    import torch

    ids = torch.arange(env.num_envs, device=env.device)
    env.reset(ids)
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    speed_errors, wrist_errors, stop_speeds = [], [], []
    start_x = None
    distance = torch.zeros(env.num_envs, device=env.device)
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for step in range(round(10 / env.step_dt)):
            if not app.is_running():
                raise RuntimeError("Evaluation interrupted before the protocol completed")
            t = step * env.step_dt
            cmd = 0.2 if 2 <= t < 6 else 0.0
            env.eval_command = cmd
            env.command_generator.command[:] = 0
            env.command_generator.command[:, 0] = cmd
            # Replace only current command; do not advance history twice per step.
            obs = obs.clone()
            obs[:, -96 + 6 : -96 + 9] = env.command_generator.command
            if step == round(2 / env.step_dt):
                start_x = env.robot.data.root_pos_w[:, 0].clone()
            obs, _, done, _ = env.step(policy(obs))
            failed |= done.bool()
            speed = env.robot.data.root_lin_vel_b[:, :2].norm(dim=-1)
            if 3 <= t < 6:
                speed_errors.append(
                    (env.robot.data.root_lin_vel_b[:, :2] - env.command_generator.command[:, :2])
                    .norm(dim=-1)
                    .clone()
                )
            if step == round(6 / env.step_dt) - 1:
                distance = env.robot.data.root_pos_w[:, 0] - start_x
            if t >= 8:
                stop_speeds.append(speed.clone())
            wrist_errors.append(
                (
                    env.robot.data.joint_pos[:, env.robot.joint_names.index(WRIST_NAME)]
                    - WRIST_TARGET
                )
                .abs()
                .clone()
            )
            if play:
                root = env.robot.data.root_pos_w[0].cpu().tolist()
                env.sim.set_camera_view((root[0] - 2.5, root[1] - 2.5, 1.8), root)
                time.sleep(env.step_dt)
    velocity = torch.stack(speed_errors).mean(0)
    wrist = torch.stack(wrist_errors).amax(0)
    stopped = torch.stack(stop_speeds).mean(0)
    success = ~failed & (velocity < 0.10) & (wrist < 0.18) & (stopped < 0.08) & (distance > 0.4)
    return {
        "walking_capability": "PASS" if success.float().mean().item() >= 0.95 else "FAIL",
        "protocol": "stop2s_forward0.2m_s4s_stop4s",
        "num_envs": env.num_envs,
        "success_fraction": success.float().mean().item(),
        "failures": failed.sum().item(),
        "velocity_error_mean_m_s": velocity.mean().item(),
        "wrist_peak_error_rad": wrist.max().item(),
        "stop_speed_mean_m_s": stopped.mean().item(),
        "forward_distance_mean_m": distance.mean().item(),
        "per_env_success": success.cpu().tolist(),
        "handover": "NOT RUN",
        "full_climb": "NOT RUN",
    }


if __name__ == "__main__":
    main()
