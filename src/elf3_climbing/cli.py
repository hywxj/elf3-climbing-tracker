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
    p.add_argument(
        "--segment", choices=["balance", "flip", "stand", "prepare", "approach"], default="flip"
    )
    p.add_argument("--approach-end-frame", type=int, choices=[73, 128, 173], default=128)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--failure-warmup-iterations", type=int, default=200)
    p.add_argument("--wrist-noise-std", type=float, default=0.005)
    p.add_argument("--balance-noise-std", type=float, default=0.08)
    p.add_argument("--reward-stage", choices=["prior", "refine"], default="prior")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--play-episodes", type=int, default=0, help="play: 0 loops until window closes")
    p.add_argument("--play-speed", type=float, default=1.0, help="play: 0.5 is half speed")
    p.add_argument("--view", choices=["rear-left", "front", "left", "right"], default="rear-left")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--flat-ground", action="store_true")
    p.add_argument(
        "--wrist-only",
        action="store_true",
        help="Preserve standing actor feedback; train wrist output only",
    )
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
    if args.wrist_only and (
        args.segment != "flip" or not args.flat_ground or args.checkpoint is None
    ):
        p.error("--wrist-only requires flat-ground flip and a checkpoint")
    args.balance_noise_explicit = any(
        a == "--balance-noise-std" or a.startswith("--balance-noise-std=") for a in sys.argv[1:]
    )
    if args.output is None:
        p.error("--output is required")
    if args.segment == "approach" and (
        not args.checkpoint or not args.flat_ground or not args.no_curriculum
    ):
        p.error("Approach continuation requires checkpoint, flat ground and no backward curriculum")
    if (
        min(
            args.num_envs,
            args.steps,
            args.iterations,
            args.eval_interval,
            args.early_stop_patience,
            args.failure_warmup_iterations,
        )
        < 1
    ):
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
    if not math.isfinite(args.wrist_noise_std) or not 0 < args.wrist_noise_std <= 0.02:
        p.error("--wrist-noise-std must be in (0, 0.02]")
    if not math.isfinite(args.balance_noise_std) or not 0 < args.balance_noise_std <= 0.1:
        p.error("--balance-noise-std must be in (0, 0.1]")
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error("Choose a new output directory")
    # Logs/checkpoints are kept in the data area, independently of code and Hiking.
    if not output.is_relative_to(OUTPUT.resolve()):
        p.error(f"Output must be inside {OUTPUT}")
    output.mkdir(parents=True)
    if args.config is not None:
        (output / "input_config.toml").write_bytes(args.config.read_bytes())
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
        from .environment import TrackerCfg, TrackerEnv
        from rsl_rl.runners import OnPolicyRunner

        cfg = TrackerCfg()
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.reference_path = str(args.reference)
        cfg.segment = args.segment
        cfg.flat_ground = args.flat_ground
        cfg.approach_end_frame = args.approach_end_frame
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
            "task": env.task_report,
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
        ppo_cfg = runner_config(settings["ppo"])
        if args.segment == "approach":
            import rsl_rl.runners.on_policy_runner as runner_module
            from .continuation import StandingContinuationActorCritic

            runner_module.StandingContinuationActorCritic = StandingContinuationActorCritic
            ppo_cfg["policy"]["class_name"] = "StandingContinuationActorCritic"
            ppo_cfg["policy"]["leg_indices"] = [
                j
                for j, n in enumerate(env.reference_np["joint_names"].tolist())
                if any(part in n for part in ("waist", "hip", "knee", "ankle"))
            ]
        (output / "ppo_config.json").write_text(json.dumps(ppo_cfg, indent=2))
        runner = OnPolicyRunner(
            wrapped,
            ppo_cfg,
            log_dir=str(output / "ppo") if args.mode == "train" else None,
            device=env.device,
        )
        wrapped.initial_normalizers = (runner.obs_normalizer, runner.privileged_obs_normalizer)
        if args.checkpoint is not None:
            ckpt = torch.load(args.checkpoint, weights_only=True, map_location=env.device)
            weights = ckpt["model_state_dict"]
            previous_segment = (ckpt.get("infos") or {}).get("contract", {}).get("segment")
            if args.segment == "balance" and previous_segment != "balance":
                raise ValueError(
                    "Start balance from scratch or use a balance checkpoint; flip weights are not a standing prior"
                )
            if args.segment == "approach":
                from .continuation import inherit_standing_checkpoint, freeze_prior_statistics

                if previous_segment not in ("flip", "approach"):
                    raise ValueError(
                        "Approach requires the accepted flip prior or its continuation"
                    )
                mapped, inherited = inherit_standing_checkpoint(ckpt)
                missing, unexpected = torch.nn.Module.load_state_dict(
                    runner.alg.policy, mapped, strict=False
                )
                if unexpected or any(
                    not k.startswith("actor.correction.") and k != "actor.mask" for k in missing
                ):
                    raise ValueError(f"Continuation checkpoint mismatch: {missing}, {unexpected}")
                runner.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
                runner.privileged_obs_normalizer.load_state_dict(
                    ckpt["privileged_obs_norm_state_dict"]
                )
                freeze_prior_statistics(runner)
                env.transfer_report = {
                    "method": "frozen_standing_flip_plus_zero_initialized_leg_residual",
                    "source": str(args.checkpoint),
                    "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "inherited_original_actor": inherited,
                    "actor_observations": 632,
                    "critic_observations": 697,
                    "base_actor_frozen": True,
                    "actor_normalizer_frozen": True,
                    "residual_bound_rad": 0.15,
                    "residual_enabled_after_reference_frame": env.flip_end_frame,
                }
                (output / "standing_transfer.json").write_text(
                    json.dumps(env.transfer_report, indent=2)
                )
            elif weights["actor.0.weight"].shape != runner.alg.policy.actor[0].weight.shape:
                raise ValueError(
                    "Checkpoint observation schema mismatch; implicit padding is forbidden"
                )
            if args.segment != "approach":
                runner.load(str(args.checkpoint), load_optimizer=False)
            if previous_segment == "balance" and args.segment == "flip":
                if args.mode != "train" or not args.flat_ground or not args.no_curriculum:
                    raise ValueError(
                        "Standing-to-flip transfer requires train, --flat-ground and --no-curriculum"
                    )
                from .transfer import standing_to_flip

                transfer = standing_to_flip(
                    runner.alg.policy,
                    runner.obs_normalizer,
                    runner.privileged_obs_normalizer,
                    env.wrist_joint,
                    (
                        env.ref["joint_pos"][env.segment_end, env.wrist_joint]
                        - env.nominal[0, env.wrist_joint]
                    )
                    / cfg.action_scale,
                )
                transfer.update(
                    reference_frame=env.segment_end,
                    reference_wrist_angle_rad=float(
                        env.ref["joint_pos"][env.segment_end, env.wrist_joint]
                    ),
                    reset_wrist_angle_rad=float(env.ref["joint_pos"][0, env.wrist_joint]),
                    target_rate_limit_rad_s=float(env.target_rates[env.wrist_joint]),
                )
                env.transfer_report = transfer
                (output / "standing_transfer.json").write_text(json.dumps(transfer, indent=2))
            print(
                f"CHECKPOINT_NOTE: configured gripper added mass is {env.gripper_mass_report['added_mass_kg']} kg; evaluate under this current model.",
                flush=True,
            )
        elif args.mode == "train" and not args.no_pose_init:
            # Initialize a constant action bias; for flip, include static gravity load.
            # No per-frame reference is added to policy actions.
            with torch.no_grad():
                runner.alg.policy.actor[-1].weight.zero_()
                if args.segment == "flip":
                    from .initialization import static_action_bias

                    bias, seed_report = static_action_bias(env)
                    (output / "initialization.json").write_text(json.dumps(seed_report, indent=2))
                    runner.alg.policy.actor[-1].bias.copy_(
                        torch.as_tensor(bias, device=env.device, dtype=torch.float32)
                    )
                else:
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
                from .training import train

                train(runner, wrapped, args, output)
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
