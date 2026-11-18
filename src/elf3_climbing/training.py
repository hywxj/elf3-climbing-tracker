"""Bounded PPO blocks with frozen precision exploration and explicit evaluations."""

from dataclasses import asdict
import json
import math
import torch

from .evaluation import evaluate
from .training_control import (
    EvaluationGuard,
    BalanceExploration,
    initial_balance_std,
    finalize_checkpoint,
    save_checkpoint_atomically,
)


def suite(runner, wrapped, phase):
    start = evaluate(runner, wrapped, 0.0)
    tail = start if wrapped.env.balance_only else evaluate(runner, wrapped, 0.9)
    current = start if phase == 0 else tail if phase == 0.9 else evaluate(runner, wrapped, phase)
    probe = (
        evaluate(runner, wrapped, phase, hold_steps=50)
        if wrapped.env.cfg.end_hold_steps > 50
        else current
    )
    return {"from_start": start, "tail": tail, "current": current, "short_probe": probe}


def score(results, track_flip=False):
    quality = (
        results["from_start"]["success_fraction"],
        results["tail"]["success_fraction"],
        sum(x["max_stable_hold_s"] for x in results["from_start"]["episodes"])
        / max(1, len(results["from_start"]["episodes"])),
        sum(x["max_stable_hold_s"] for x in results["tail"]["episodes"])
        / max(1, len(results["tail"]["episodes"])),
    )
    episodes = results["from_start"]["episodes"]
    if track_flip:
        # Failed motion never outranks equally unsuccessful but surviving motion
        # solely by reaching a wrist angle during a fall.
        quality += (
            sum(not x["terminated"] for x in episodes) / max(1, len(episodes)),
            -sum(x["terminal_gripper_error_rad"] for x in episodes) / max(1, len(episodes)),
        )
    return quality + (
        -sum(
            x["terminal_root_velocity_error_m_s"] / 0.25
            + x["terminal_root_angular_velocity_error_rad_s"] / 0.5
            for x in results["from_start"]["episodes"]
        )
        / max(1, len(results["from_start"]["episodes"])),
    )


def train(runner, wrapped, args, output):
    env = wrapped.env
    wrist_only = args.wrist_only
    if wrist_only:
        from .transfer import preserve_standing_feedback

        runner.standing_preservation_hooks = preserve_standing_feedback(
            runner.alg.policy, runner.obs_normalizer, env.wrist_joint
        )
    balance_indices = [
        j
        for j, name in enumerate(env.reference_np["joint_names"].tolist())
        if any(part in name for part in ("waist", "hip", "knee", "ankle"))
    ]
    restored_balance_std = (
        float(runner.alg.policy.log_std[balance_indices].detach().exp().median())
        if args.checkpoint
        else None
    )
    exploration = BalanceExploration(
        std=args.balance_noise_std, minimum=min(0.01, args.balance_noise_std)
    )
    # Wrist precision stays fixed; balance exploration is annealed only after evaluations.
    with torch.no_grad():
        runner.alg.policy.log_std.fill_(math.log(runner.policy_cfg["init_noise_std"]))
        for j, name in enumerate(env.reference_np["joint_names"].tolist()):
            if any(part in name for part in ("waist", "hip", "knee", "ankle")):
                runner.alg.policy.log_std[j] = math.log(args.balance_noise_std)
            elif "wrist" in name:
                runner.alg.policy.log_std[j] = math.log(args.wrist_noise_std)
        if wrist_only:
            runner.alg.policy.log_std.fill_(math.log(0.001))
            runner.alg.policy.log_std[env.wrist_joint] = math.log(args.wrist_noise_std)
    runner.alg.policy.log_std.requires_grad_(False)
    runner.current_learning_iteration = 0  # --checkpoint is an explicit warm start, not resume
    contract = {
        "protocol": "ground_balance_v1" if env.balance_only else "precision_tracking_v10",
        "task": env.task_report,
        "transfer": getattr(env, "transfer_report", None),
        "wrist_only": wrist_only,
        "standing_feedback": "actor hidden layers, other output rows and actor normalizer frozen"
        if wrist_only
        else "joint optimization",
        "checkpoint_selection": "full success, continuous hold, then survival and wrist error for flip, then velocity error",
        "reward_version": "tolerant_reference_velocity_v9",
        "reference": str(args.reference),
        "segment": args.segment,
        "height_source": "ground_truth_boxes",
        "real_lidar": "NOT CONNECTED",
        "gripper_dynamics": env.gripper_mass_report,
        "gripper_rules": asdict(env.rules),
        "gripper_requirement_relaxed": env.balance_only,
        "additional_terminal_stability": {
            "scope": "Completion only; current segment ends in a held pose. No absolute-speed episode termination.",
            "linear_velocity_error_m_s": 0.25,
            "angular_velocity_error_rad_s": 0.5,
            "continuous_hold_steps": env.cfg.end_hold_steps,
            "continuous_hold_s": env.cfg.end_hold_steps * env.step_dt,
        },
        "reward_stage": args.reward_stage,
        "velocity_reward_full_credit_band": {
            "linear_error_m_s": env.cfg.linear_velocity_reward_tolerance,
            "angular_error_rad_s": env.cfg.angular_velocity_reward_tolerance,
            "outside_band": "exp(-max(error-tolerance,0)^2/0.25)",
        },
        "exploration": {
            "learned_variance": False,
            "wrist_fixed": True,
            "balance_schedule": (
                "fixed 0.001 rad outside left wrist; no balance annealing in wrist-only stage"
                if wrist_only
                else "halve after two full 5 s evaluations >=90% if sampled success <80%"
                if env.balance_only
                else "halve after two matching-phase 1 s probes >=90% if sampled full-hold success <80%; promotion still requires full hold"
            ),
            "balance_minimum_std": exploration.minimum,
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
        "failure_warmup_iterations": args.failure_warmup_iterations,
        "stall_progress_margin_s": 0.1,
        "scope": "Short-segment trial, not full ACP or accepted climb",
    }
    before = suite(runner, wrapped, env.curriculum.phase)
    exploration.std = (
        0.001
        if wrist_only
        else initial_balance_std(
            args.balance_noise_std,
            restored_balance_std,
            before["current" if env.balance_only else "short_probe"]["success_fraction"],
            args.balance_noise_explicit,
        )
    )
    exploration.minimum = min(exploration.minimum, exploration.std)
    with torch.no_grad():
        runner.alg.policy.log_std[balance_indices] = math.log(exploration.std)
    contract["exploration"]["balance_minimum_std"] = exploration.minimum
    contract["exploration"]["requested_balance_std"] = args.balance_noise_std
    contract["exploration"]["effective_initial_balance_std"] = exploration.std
    contract["exploration"]["warm_start_precision_cap_applied"] = (
        not wrist_only and exploration.std < args.balance_noise_std
    )
    contract["exploration"]["joint_std_by_name"] = dict(
        zip(
            env.reference_np["joint_names"].tolist(),
            runner.alg.policy.log_std.detach().exp().cpu().tolist(),
        )
    )
    (output / "training_contract.json").write_text(json.dumps(contract, indent=2))
    (output / "evaluation_before.json").write_text(json.dumps(before, indent=2))
    track_flip = args.segment == "flip"
    best_score = score(before, track_flip)
    best_iteration = 0
    runner.logger_type = "tensorboard"
    save_checkpoint_atomically(
        runner, output / "best.pt", infos={"completed_iterations": 0, "contract": contract}
    )
    guard = EvaluationGuard(
        patience=args.early_stop_patience,
        min_iterations_before_stall=args.failure_warmup_iterations,
        best_start=before["from_start"]["success_fraction"],
        best_tail=before["tail"]["success_fraction"],
    )
    before_weights = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
    before_actor_norm = {k: v.clone() for k, v in runner.obs_normalizer.state_dict().items()}
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
            balance_std_used = exploration.std
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
            hold_progress = sum(
                e["max_stable_hold_s"] for e in results["current"]["episodes"]
            ) / max(1, len(results["current"]["episodes"]))
            reason = guard.inspect(
                start_rate,
                tail_rate,
                phase,
                phase_rate,
                training_rate,
                completed_iterations=completed,
                progress=hold_progress,
            )
            annealed = (
                not wrist_only
                and not reason
                and exploration.observe(
                    phase,
                    phase_rate if env.balance_only else results["short_probe"]["success_fraction"],
                    training_rate,
                )
            )
            if annealed:
                with torch.no_grad():
                    runner.alg.policy.log_std[balance_indices] = math.log(exploration.std)
                env.curriculum.samples.clear()  # New sampling distribution needs a fresh window.
            improved = score(results, track_flip) > best_score
            info = {
                "completed_iterations": completed,
                "curriculum": env.curriculum.state(),
                "balance_noise_std": exploration.std,
                "contract": contract,
            }
            if improved:
                best_score = score(results, track_flip)
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
                "mean_max_stable_hold_s": hold_progress,
                "mean_terminal_gripper_error_rad": sum(
                    e["terminal_gripper_error_rad"] for e in results["from_start"]["episodes"]
                )
                / max(1, len(results["from_start"]["episodes"])),
                "balance_noise_std_used": balance_std_used,
                "next_balance_noise_std": exploration.std,
                "exploration_annealed": bool(annealed),
                "outcomes": counts,
                "from_start_success_fraction": start_rate,
                "tail_success_fraction": tail_rate,
                "current_phase_success_fraction": phase_rate,
                "short_probe_success_fraction": results["short_probe"]["success_fraction"],
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
                runner.writer.add_scalar("Exploration/balance_std", exploration.std, completed)
                runner.writer.add_scalar(
                    "Evaluation/current_mean_max_stable_hold_s", hold_progress, completed
                )
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
    standing_preserved = None
    if wrist_only:
        after_weights = list(runner.alg.policy.actor.parameters())
        body_rows = [j for j in range(29) if j != env.wrist_joint]
        standing_preserved = (
            all(torch.equal(a, b) for a, b in zip(before_weights[:-2], after_weights[:-2]))
            and torch.equal(before_weights[-2][body_rows], after_weights[-2][body_rows])
            and torch.equal(before_weights[-1][body_rows], after_weights[-1][body_rows])
            and all(
                torch.equal(v, runner.obs_normalizer.state_dict()[k])
                for k, v in before_actor_norm.items()
            )
        )
        if not standing_preserved:
            raise RuntimeError("Standing feedback changed during wrist-only training")
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
        "standing_feedback_preserved": standing_preserved,
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
