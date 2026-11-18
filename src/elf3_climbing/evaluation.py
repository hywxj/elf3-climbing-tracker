"""Deterministic first-episode evaluation; never advances training curriculum."""

import torch


def evaluate(runner, wrapped, start_phase=0.0, hold_steps=None):
    env = wrapped.env
    old = (env.training_mode, env.use_curriculum, env.evaluation_start_phase, env.cfg.random_start)
    original_hold_steps = env.cfg.end_hold_steps
    if hold_steps is not None:
        env.cfg.end_hold_steps = hold_steps
    env.training_mode = False
    env.use_curriculum = False
    env.evaluation_start_phase = start_phase
    env.cfg.random_start = False
    runner.eval_mode()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    episodes = []
    peak_root = torch.zeros(env.num_envs, device=env.device)
    peak_grip = torch.zeros_like(peak_root)
    peak_hold = torch.zeros_like(peak_root)
    gate_ever = torch.zeros_like(active)
    first_gate_time = torch.full_like(peak_root, float("nan"))
    try:
        with torch.inference_mode():
            wrapped.obs, _ = env.reset()
            initial_wrist = env.robot.data.joint_pos[:, env.joint_ids[env.wrist_joint]].clone()
            for step in range(env.max_episode_length + 2):
                obs = runner.obs_normalizer(wrapped.obs["policy"])
                actions = runner.alg.policy.act_inference(obs)
                _, _, done, _ = wrapped.step(actions)
                m = env.last_metrics
                peak_root[active] = torch.maximum(peak_root[active], m["root_error_m"][active])
                peak_grip[active] = torch.maximum(peak_grip[active], m["gripper_error_rad"][active])
                peak_hold[active] = torch.maximum(
                    peak_hold[active], m["stable_hold_steps"][active].float()
                )
                newly_ready = active & m["gate_allowed"] & ~gate_ever
                first_gate_time[newly_ready] = (step + 1) * env.step_dt
                gate_ever[active] |= m["gate_allowed"][active]
                finished = active & done.bool()
                for k in finished.nonzero().flatten().tolist():
                    episodes.append(
                        {
                            "environment": k,
                            "steps": step + 1,
                            "initial_gripper_angle_rad": float(initial_wrist[k]),
                            "terminal_gripper_angle_rad": float(m["joint_pos"][k, env.wrist_joint]),
                            "first_gate_ready_s": float(first_gate_time[k])
                            if torch.isfinite(first_gate_time[k])
                            else None,
                            "max_stable_hold_s": float(peak_hold[k]) * env.step_dt,
                            "success": bool(m["success"][k]),
                            "terminal_root_velocity_error_m_s": float(
                                m["root_velocity_error_m_s"][k]
                            ),
                            "terminal_root_angular_velocity_error_rad_s": float(
                                m["root_angular_velocity_error_rad_s"][k]
                            ),
                            "terminal_root_speed_m_s": float(m["root_speed_m_s"][k]),
                            "terminal_root_angular_speed_rad_s": float(
                                m["root_angular_speed_rad_s"][k]
                            ),
                            "terminated": bool(m["terminated"][k]),
                            "timeout": bool(m["expired"][k] and not m["success"][k]),
                            "failure_root": bool(m["failure_root"][k]),
                            "failure_orientation": bool(m["failure_orientation"][k]),
                            "failure_joint": bool(m["failure_joint"][k]),
                            "terminal_gripper_speed_rad_s": float(m["gripper_speed_rad_s"][k]),
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
                            "terminal_forbidden_body_forces_N": {
                                env.contacts.body_names[body]: float(force)
                                for body, force in zip(
                                    env.forbidden_sensor_ids, m["forbidden_body_force_N"][k]
                                )
                            },
                        }
                    )
                active[finished] = False
                if not active.any():
                    break
        count = sum(e["success"] for e in episodes)
        return {
            "start_phase": start_phase,
            "required_hold_steps": env.cfg.end_hold_steps,
            "required_hold_s": env.cfg.end_hold_steps * env.step_dt,
            "completed_episodes": len(episodes),
            "successes": count,
            "success_fraction": count / max(1, env.num_envs),
            "unfinished": int(active.sum()),
            "scope": "Fixed scene, deterministic first episode per clone; not randomized robustness or full climb",
            "episodes": episodes,
        }
    finally:
        env.cfg.end_hold_steps = original_hold_steps
        env.training_mode, env.use_curriculum, env.evaluation_start_phase, env.cfg.random_start = (
            old
        )
