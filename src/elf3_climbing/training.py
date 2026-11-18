"""Bounded PPO blocks with frozen precision exploration and explicit evaluations."""

from dataclasses import asdict
import json
import math
import torch

from .evaluation import evaluate
from .training_control import EvaluationGuard, finalize_checkpoint, save_checkpoint_atomically


def suite(runner, wrapped, phase):
    start = evaluate(runner, wrapped, 0.0)
    tail = evaluate(runner, wrapped, 0.9)
    current = start if phase == 0 else tail if phase == 0.9 else evaluate(runner, wrapped, phase)
    return {"from_start": start, "tail": tail, "current": current}


def score(results):
    return (
        results["from_start"]["success_fraction"],
        results["tail"]["success_fraction"],
        -sum(
            x["terminal_root_velocity_error_m_s"] / 0.25
            + x["terminal_root_angular_velocity_error_rad_s"] / 0.5
            for x in results["from_start"]["episodes"]
        )
        / max(1, len(results["from_start"]["episodes"])),
    )


def train(runner, wrapped, args, output):
    env = wrapped.env
    # Fix the small short-segment exploration distribution. Entropy cannot grow it.
    with torch.no_grad():
        runner.alg.policy.log_std.fill_(math.log(runner.policy_cfg["init_noise_std"]))
        for j, name in enumerate(env.reference_np["joint_names"].tolist()):
            if any(part in name for part in ("waist", "hip", "knee", "ankle")):
                runner.alg.policy.log_std[j] = math.log(args.balance_noise_std)
            elif "wrist" in name:
                runner.alg.policy.log_std[j] = math.log(args.wrist_noise_std)
    runner.alg.policy.log_std.requires_grad_(False)
    runner.current_learning_iteration = 0  # --checkpoint is an explicit warm start, not resume
    contract = {
        "protocol": "precision_tracking_v8",
        "reward_version": "reference_velocity_with_terminal_hold_v8",
        "reference": str(args.reference),
        "segment": args.segment,
        "height_source": "ground_truth_boxes",
        "real_lidar": "NOT CONNECTED",
        "gripper_dynamics": env.gripper_mass_report,
        "gripper_rules": asdict(env.rules),
        "gripper_requirement_relaxed": False,
        "additional_terminal_stability": {
            "scope": "Completion only; current segment ends in a held pose. No absolute-speed episode termination.",
            "linear_velocity_error_m_s": 0.25,
            "angular_velocity_error_rad_s": 0.5,
            "continuous_hold_steps": env.cfg.end_hold_steps,
            "continuous_hold_s": env.cfg.end_hold_steps * env.step_dt,
        },
        "reward_stage": args.reward_stage,
        "exploration": {
            "fixed": True,
            "joint_std": runner.policy_cfg["init_noise_std"],
            "wrist_std": args.wrist_noise_std,
            "balance_std": args.balance_noise_std,
            "joint_std_by_name": dict(
                zip(
                    env.reference_np["joint_names"].tolist(),
                    runner.alg.policy.log_std.detach().exp().cpu().tolist(),
                )
            ),
        },
        "actor_size": wrapped.obs["policy"].shape[1],
        "critic_size": wrapped.obs["critic"].shape[1],
        "initialization": "loaded checkpoint"
        if args.checkpoint
        else "random"
        if args.no_pose_init
        else "static gravity-aware action bias"
        if args.segment == "flip"
        else "fixed final-pose action bias",
        "eval_interval": args.eval_interval,
        "early_stop_patience": args.early_stop_patience,
        "scope": "Short-segment trial, not full ACP or accepted climb",
    }
    (output / "training_contract.json").write_text(json.dumps(contract, indent=2))
    before = suite(runner, wrapped, env.curriculum.phase)
    (output / "evaluation_before.json").write_text(json.dumps(before, indent=2))
    best_score = score(before)
    best_iteration = 0
    runner.logger_type = "tensorboard"
    save_checkpoint_atomically(
        runner, output / "best.pt", infos={"completed_iterations": 0, "contract": contract}
    )
    guard = EvaluationGuard(
        patience=args.early_stop_patience,
        best_start=before["from_start"]["success_fraction"],
        best_tail=before["tail"]["success_fraction"],
    )
    before_weights = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
    completed = 0
    stop_reason = "iteration_budget"
    results = before
    evaluated_results = before
    evaluated_iterations = 0
    try:
        while completed < args.iterations:
            phase = env.curriculum.phase if env.cfg.curriculum_enabled else 0.0
            env.training_mode = True
            env.use_curriculum = env.cfg.curriculum_enabled
            env.evaluation_start_phase = None
            env.outcome_counts.zero_()
            with torch.inference_mode():
                wrapped.obs, _ = env.reset()
            block = min(args.eval_interval, args.iterations - completed)
            runner.current_learning_iteration = completed
            runner.learn(num_learning_iterations=block, init_at_random_ep_len=False)
            completed += block
            runner.current_learning_iteration = completed
            counts = dict(zip(env.outcome_names, env.outcome_counts.cpu().tolist()))
            training_rate = counts["successes"] / max(1, counts["episodes"])
            # Evaluate at PPO boundaries; never insert resets into a collected rollout.
            results = suite(runner, wrapped, phase)
            start_rate = results["from_start"]["success_fraction"]
            tail_rate = results["tail"]["success_fraction"]
            phase_rate = results["current"]["success_fraction"]
            reason = guard.inspect(start_rate, tail_rate, phase, phase_rate, training_rate)
            improved = score(results) > best_score
            info = {
                "completed_iterations": completed,
                "curriculum": env.curriculum.state(),
                "contract": contract,
            }
            if improved:
                best_score = score(results)
                best_iteration = completed
                save_checkpoint_atomically(runner, output / "best.pt", infos=info)
            save_checkpoint_atomically(runner, output / "latest.pt", infos=info)
            evaluated_results = results
            evaluated_iterations = completed
            promoted = (
                not reason
                and env.cfg.curriculum_enabled
                and env.curriculum.promote(phase, phase_rate)
            )
            record = {
                "completed_iterations": completed,
                "phase": phase,
                "training_success_fraction": training_rate,
                "outcomes": counts,
                "from_start_success_fraction": start_rate,
                "tail_success_fraction": tail_rate,
                "current_phase_success_fraction": phase_rate,
                "promoted": bool(promoted),
                "next_phase": env.curriculum.phase,
                "stop_reason": reason,
                "best_iteration": best_iteration,
            }
            with (output / "evaluation_history.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            (output / f"evaluation_{completed:05d}.json").write_text(json.dumps(results, indent=2))
            if runner.writer is not None:
                for key, value in [
                    ("from_start_success", start_rate),
                    ("tail_success", tail_rate),
                    ("current_phase_success", phase_rate),
                    ("training_success", training_rate),
                ]:
                    runner.writer.add_scalar("Evaluation/" + key, value, completed)
                for key, value in counts.items():
                    runner.writer.add_scalar("Outcomes/" + key, value, completed)
                runner.writer.flush()
            print("TRACKER_CHECK", json.dumps(record), flush=True)
            if reason:
                stop_reason = reason
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("TRAINING_INTERRUPTED: completed-block latest.pt/best.pt retained", flush=True)
    updated = sum(
        not torch.equal(a, b) for a, b in zip(before_weights, runner.alg.policy.actor.parameters())
    )
    assessed = finalize_checkpoint(
        runner, output, evaluated_iterations, contract, interrupted=stop_reason == "interrupted"
    )
    results = evaluated_results
    (output / "evaluation_after.json").write_text(json.dumps(results, indent=2))
    summary = {
        "requested_iterations": args.iterations,
        "ppo_iterations": completed,
        "evaluated_iterations": evaluated_iterations,
        "checkpoint_source": assessed.name,
        "partial_block_updates": "unknown, excluded" if stop_reason == "interrupted" else "none",
        "stop_reason": stop_reason,
        "changed_actor_parameter_tensors": updated,
        "best_iteration": best_iteration,
        "training_environment_steps": completed * runner.num_steps_per_env * env.num_envs,
        "curriculum": env.curriculum.state(),
        "from_start_success_before": before["from_start"]["success_fraction"],
        "from_start_success_after": results["from_start"]["success_fraction"],
        "tail_success_before": before["tail"]["success_fraction"],
        "tail_success_after": results["tail"]["success_fraction"],
        "reward_stage": args.reward_stage,
        "full_climb": "NOT RUN",
        "qualified_for_deployment": False,
    }
    (output / "training_result.json").write_text(json.dumps(summary, indent=2))
    print("TRAINING_RESULT", json.dumps(summary), flush=True)
