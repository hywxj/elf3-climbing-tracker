"""Deterministic first-episode evaluation; never advances training curriculum."""

import torch


def evaluate(runner, wrapped, start_phase=0.0):
    env = wrapped.env
    old = (env.training_mode, env.use_curriculum, env.evaluation_start_phase, env.cfg.random_start)
    env.training_mode = False
    env.use_curriculum = False
    env.evaluation_start_phase = start_phase
    env.cfg.random_start = False
    runner.eval_mode()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    episodes = []
    peak_root = torch.zeros(env.num_envs, device=env.device)
    peak_grip = torch.zeros_like(peak_root)
    gate_ever = torch.zeros_like(active)
    try:
        with torch.inference_mode():
            wrapped.obs, _ = env.reset()
            for step in range(env.max_episode_length + 2):
                obs = runner.obs_normalizer(wrapped.obs["policy"])
                actions = runner.alg.policy.act_inference(obs)
                _, _, done, _ = wrapped.step(actions)
                m = env.last_metrics
                peak_root[active] = torch.maximum(peak_root[active], m["root_error_m"][active])
                peak_grip[active] = torch.maximum(peak_grip[active], m["gripper_error_rad"][active])
                gate_ever[active] |= m["gate_allowed"][active]
                finished = active & done.bool()
                for k in finished.nonzero().flatten().tolist():
                    episodes.append(
                        {
                            "environment": k,
                            "steps": step + 1,
                            "success": bool(m["success"][k]),
                            "reset_frame": int(m["reset_frame"][k]),
                            "terminal_reference_frame": int(m["reference_frame"][k]),
                            "assumed_prior_flip": bool(m["assumed_prerequisite"][k]),
                            "gate_ever_ready": bool(gate_ever[k]),
                            "root_error_max_m": float(peak_root[k]),
                            "terminal_orientation_error_rad": float(m["orientation_error_rad"][k]),
                            "terminal_joint_rmse_rad": float(m["joint_rmse_rad"][k]),
                            "terminal_foot_forces_N": m["foot_force_N"][k].tolist(),
                            "terminal_foot_advance_m": float(m["foot_advance_m"][k]),
                            "premature_arm_contact": bool(m["premature_arm_contact"][k]),
                            "terminal_gripper_error_rad": float(m["gripper_error_rad"][k]),
                            "gripper_error_max_rad": float(peak_grip[k]),
                            "terminal_gripper_clearance_m": float(m["gripper_clearance_m"][k]),
                            "premature_advance": bool(m["premature_advance"][k]),
                            "gripper_collision": bool(m["gripper_collision"][k]),
                            "lost_flip": bool(m["lost_flip"][k]),
                            "unexpected_contact": bool(m["failure_forbidden_contact"][k]),
                        }
                    )
                active[finished] = False
                if not active.any():
                    break
        count = sum(e["success"] for e in episodes)
        return {
            "start_phase": start_phase,
            "completed_episodes": len(episodes),
            "successes": count,
            "success_fraction": count / max(1, env.num_envs),
            "unfinished": int(active.sum()),
            "scope": "Fixed scene, deterministic first episode per clone; not randomized robustness or full climb",
            "episodes": episodes,
        }
    finally:
        env.training_mode, env.use_curriculum, env.evaluation_start_phase, env.cfg.random_start = (
            old
        )
