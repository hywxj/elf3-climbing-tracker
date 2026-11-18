"""Explicit smoke/train entry point; no hardware access and no Hiking checkpoint loads."""

import hashlib
import argparse
import sys
import json
import math
from pathlib import Path

from .paths import DEFAULT_PACK, OUTPUT
from .runner import LegacyRslAdapter, runner_config
from .settings import load_settings
from .calibration import check_model_frames


def parser(settings=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="TOML settings; CLI options override [run]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["smoke", "train", "evaluate", "play"], default="smoke")
    p.add_argument("--reference", type=Path, default=DEFAULT_PACK)
    p.add_argument("--segment", choices=["flip", "stand", "prepare"], default="flip")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--reward-stage", choices=["prior", "refine"], default="prior")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--play-episodes", type=int, default=0, help="play: 0 loops until window closes")
    p.add_argument("--play-speed", type=float, default=1.0, help="play: 0.5 is half speed")
    p.add_argument("--view", choices=["rear-left", "front", "left", "right"], default="rear-left")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-pose-init", action="store_true")
    p.add_argument(
        "--output", type=Path, help="Required: a new directory under ELF3_RUNTIME_ROOT/runs"
    )
    p.add_argument("--smoke-controller", choices=["reference-pd", "zero"], default="zero")
    if settings:
        p.set_defaults(**settings["run"])
    return p


def main():
    from isaaclab.app import AppLauncher

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args()
    settings = load_settings(known.config)
    p = parser(settings)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.output is None:
        p.error("--output is required")
    if args.num_envs < 1 or args.steps < 1 or args.iterations < 1:
        p.error("Counts must be positive")
    if (
        args.mode in ("evaluate", "play") or args.reward_stage == "refine"
    ) and args.checkpoint is None:
        p.error("Evaluation, playback and reward refinement require an explicit checkpoint")
    if args.mode == "play" and (
        args.num_envs != 1
        or args.play_episodes < 0
        or not math.isfinite(args.play_speed)
        or args.play_speed <= 0
    ):
        p.error(
            "play requires --num-envs 1, nonnegative --play-episodes and positive finite --play-speed"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("Choose a new output directory")
    # Logs/checkpoints are kept in the data area, independently of code and Hiking.
    if not output.is_relative_to(OUTPUT.resolve()):
        p.error(f"Output must be inside {OUTPUT}")
    output.mkdir(parents=True)
    (output / "request.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
            default=str,
        )
    )
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    failed = False
    try:
        import torch
        import numpy as np
        from .environment import TrackerCfg, TrackerEnv
        from rsl_rl.runners import OnPolicyRunner

        cfg = TrackerCfg()
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.reference_path = str(args.reference)
        cfg.segment = args.segment
        cfg.random_start = False
        cfg.curriculum_enabled = args.mode == "train" and not args.no_curriculum
        cfg.reward_stage = args.reward_stage
        env = TrackerEnv(cfg)
        wrapped = LegacyRslAdapter(env)
        calibration = check_model_frames(env)
        wrapped.obs = env._get_observations()
        (output / "fk_calibration.json").write_text(json.dumps(calibration, indent=2))
        provenance = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.iterdir()
            if p.suffix in (".py", ".json")
        }
        asset = Path(cfg.robot_cfg.spawn.usd_path)
        provenance.update(
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in asset.parent.rglob("*.usd")
            }
        )
        (output / "code_and_usd_hashes.json").write_text(json.dumps(provenance, indent=2))
        i = env.frames()
        fk_error = torch.linalg.vector_norm(
            env.site_positions() - env.scene.env_origins[:, None, :] - env.ref["site_pos"][i],
            dim=-1,
        )
        mapping = {
            "gripper_dynamics": env.gripper_mass_report,
            "joint_names": env.reference_np["joint_names"].tolist(),
            "isaac_joint_indices": env.joint_ids,
            "initial_site_fk_max_error_m": float(fk_error.max()),
            "root_body": env.robot.body_names[0],
            "effort_limits_Nm": env.robot.data.joint_effort_limits[0, env.joint_ids].tolist(),
            "velocity_limits_rad_s": env.robot.data.joint_vel_limits[0, env.joint_ids].tolist(),
            "actor_size": wrapped.obs["policy"].shape[1],
            "critic_size": wrapped.obs["critic"].shape[1],
            "source_sha256": env.reference_manifest["source_sha256"],
        }
        (output / "model_mapping.json").write_text(json.dumps(mapping, indent=2))
        if float(fk_error.max()) > 0.02:
            raise RuntimeError("MuJoCo/Isaac initial FK difference > 2 cm; stop before rollout")
        # Verify that reference joint kinematics do not leak into actor observations.
        actor_before = env._get_observations()["policy"].clone()
        saved = env.ref["joint_pos"].clone()
        env.ref["joint_pos"] += 0.1
        actor_after = env._get_observations()["policy"]
        env.ref["joint_pos"].copy_(saved)
        if not torch.equal(actor_before, actor_after):
            raise RuntimeError("Reference joint positions leaked into actor")
        mapping["actor_reference_joint_isolation"] = "PASS"
        (output / "model_mapping.json").write_text(json.dumps(mapping, indent=2))
        (output / "ppo_config.json").write_text(
            json.dumps(runner_config(settings["ppo"]), indent=2)
        )
        # Construction verifies the actual installed PPO/observation interface even in smoke mode.
        runner = OnPolicyRunner(
            wrapped,
            runner_config(settings["ppo"]),
            log_dir=str(output / "ppo") if args.mode == "train" else None,
            device=env.device,
        )
        wrapped.initial_normalizers = (runner.obs_normalizer, runner.privileged_obs_normalizer)
        if args.checkpoint is not None:
            ckpt = torch.load(args.checkpoint, weights_only=True, map_location=env.device)
            weights = ckpt["model_state_dict"]
            if weights["actor.0.weight"].shape != runner.alg.policy.actor[0].weight.shape:
                raise ValueError(
                    "Checkpoint observation schema mismatch; implicit padding is forbidden"
                )
            runner.load(str(args.checkpoint), load_optimizer=False)
            print(
                "CHECKPOINT_NOTE: current gripper mass is included; older checkpoints may have been trained without it. Evaluate under the current model.",
                flush=True,
            )
        elif args.mode == "train" and not args.no_pose_init:
            # A fixed final-pose action bias helps the backward curriculum start.
            # No per-frame reference is added to policy actions.
            with torch.no_grad():
                runner.alg.policy.log_std[env.wrist_joint] = np.log(0.03)
                runner.alg.policy.actor[-1].weight.zero_()
                runner.alg.policy.actor[-1].bias.copy_(
                    (env.ref["joint_pos"][env.segment_end] - env.nominal[0]) / cfg.action_scale
                )
        if args.mode == "play":
            from .playback import play

            play(runner, wrapped, app, args, output)
        elif args.mode in ("train", "evaluate"):
            from .evaluation import evaluate

            if args.mode == "evaluate":
                result = evaluate(runner, wrapped, 0.0)
                (output / "evaluation.json").write_text(json.dumps(result, indent=2))
                print(
                    "EVALUATION_RESULT",
                    json.dumps({k: v for k, v in result.items() if k != "episodes"}),
                )
            else:
                from dataclasses import asdict

                contract = {
                    "scope": "Reference-guided contact tracker trial, not full ACP or accepted climb",
                    "reference": str(args.reference),
                    "segment": args.segment,
                    "height_source": "ground_truth_boxes",
                    "real_lidar": "NOT CONNECTED",
                    "gripper_dynamics": env.gripper_mass_report,
                    "reward_stage": args.reward_stage,
                    "gripper_rules": asdict(env.rules),
                    "curriculum": env.curriculum.state(),
                    "observation_schema": "elf3_contact_tracker_gripper_gate_v2",
                    "actor_size": wrapped.obs["policy"].shape[1],
                    "critic_size": wrapped.obs["critic"].shape[1],
                    "initialization": "loaded checkpoint"
                    if args.checkpoint
                    else ("random" if args.no_pose_init else "fixed final-pose action bias"),
                    "gripper_requirement_relaxed": False,
                }
                (output / "training_contract.json").write_text(json.dumps(contract, indent=2))
                before = evaluate(runner, wrapped, 0.0)
                before_tail = evaluate(runner, wrapped, 0.9)
                (output / "evaluation_before.json").write_text(
                    json.dumps({"from_start": before, "tail": before_tail}, indent=2)
                )
                env.training_mode = True
                env.use_curriculum = cfg.curriculum_enabled
                env.evaluation_start_phase = None
                with torch.inference_mode():
                    wrapped.obs, _ = env.reset()
                before_weights = [v.detach().clone() for v in runner.alg.policy.actor.parameters()]
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
                runner.save(str(output / "checkpoint.pt"))
                updated = sum(
                    not torch.equal(a, b)
                    for a, b in zip(before_weights, runner.alg.policy.actor.parameters())
                )
                if not updated:
                    raise RuntimeError("PPO did not change actor parameters")
                after = evaluate(runner, wrapped, 0.0)
                after_tail = evaluate(runner, wrapped, 0.9)
                (output / "evaluation_after.json").write_text(
                    json.dumps({"from_start": after, "tail": after_tail}, indent=2)
                )
                summary = {
                    "ppo_iterations": args.iterations,
                    "changed_actor_parameter_tensors": updated,
                    "training_environment_steps": args.iterations
                    * runner.num_steps_per_env
                    * env.num_envs,
                    "curriculum": env.curriculum.state(),
                    "from_start_success_before": before["success_fraction"],
                    "from_start_success_after": after["success_fraction"],
                    "tail_success_before": before_tail["success_fraction"],
                    "tail_success_after": after_tail["success_fraction"],
                    "reward_stage": args.reward_stage,
                    "full_climb": "NOT RUN",
                    "qualified_for_deployment": False,
                }
                (output / "training_result.json").write_text(json.dumps(summary, indent=2))
                print("TRAINING_RESULT", json.dumps(summary))
        else:
            from .smoke import run_smoke

            run_smoke(env, wrapped, args, output)
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "mode": args.mode,
                    "scope": "Execution complete; inspect evaluation for task success",
                }
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as exc:
        import traceback

        error = {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}
        (output / "error.json").write_text(json.dumps(error, indent=2))
        print(error["traceback"], flush=True)
        failed = True
    finally:
        if env is not None:
            env.close()
        try:
            app.close()
        except SystemExit:
            pass
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
