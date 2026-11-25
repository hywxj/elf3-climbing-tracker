"""Deterministic first-episode evaluation; never advances training curriculum."""

import torch


def evaluate(runner, wrapped, start_phase=0.0, hold_steps=None, *, sampled=False):
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
    episode_return = torch.zeros_like(peak_root)
    peak_grip = torch.zeros_like(peak_root)
    peak_hold = torch.zeros_like(peak_root)
    knee_hold = torch.zeros_like(peak_root)
    knee_hold_peak = torch.zeros_like(peak_root)
    support_streak_peak = torch.zeros_like(peak_root)
    measured_push = torch.zeros_like(active)
    knee_force_peak = torch.zeros_like(peak_root)
    foot_load_samples = torch.zeros((env.num_envs, 2), device=env.device)
    slip_sum = torch.zeros_like(foot_load_samples)
    outward_peak = torch.zeros_like(foot_load_samples)
    wrist_contact_steps = torch.zeros_like(peak_root)
    wrist_unloaded = torch.zeros_like(active)
    first_wrist_load = torch.zeros((env.num_envs, 2), device=env.device)
    gate_ever = torch.zeros_like(active)
    first_gate_time = torch.full_like(peak_root, float("nan"))
    walk_count = torch.zeros_like(peak_root)
    walk_wrist_peak = torch.zeros_like(peak_root)
    walk_velocity_sum = torch.zeros_like(peak_root)
    walk_tilt_peak = torch.zeros_like(peak_root)
    walk_foot_peak = torch.zeros_like(peak_root)
    try:
        with torch.inference_mode():
            wrapped.obs, _ = env.reset()
            initial_wrist = env.robot.data.joint_pos[:, env.joint_ids[env.wrist_joint]].clone()
            for step in range(env.max_episode_length + 2):
                obs = runner.obs_normalizer(wrapped.obs["policy"])
                actions = runner.alg.policy.act(obs) if sampled else runner.alg.policy.act_inference(obs)
                _, reward, done, _ = wrapped.step(actions)
                episode_return += reward.reshape(-1) * active
                m = env.last_metrics
                walking = active & (m["reference_frame"] > env.flip_end_frame)
                walk_count[walking] += 1
                walk_wrist_peak[walking] = torch.maximum(
                    walk_wrist_peak[walking], m["gripper_error_rad"][walking]
                )
                walk_velocity_sum[walking] += m["root_velocity_error_m_s"][walking]
                walk_tilt_peak[walking] = torch.maximum(
                    walk_tilt_peak[walking], m["trunk_tilt_rad"][walking]
                )
                walk_foot_peak[walking] = torch.maximum(
                    walk_foot_peak[walking], m["foot_reference_error_m"][walking].amax(-1)
                )
                peak_root[active] = torch.maximum(peak_root[active], m["root_error_m"][active])
                peak_grip[active] = torch.maximum(peak_grip[active], m["gripper_error_rad"][active])
                peak_hold[active] = torch.maximum(
                    peak_hold[active], m["stable_hold_steps"][active].float()
                )
                if env.knee_entry:
                    if step == 0:
                        first_wrist_load = m["wrist_top_load_N"]
                    wrist_unloaded |= active & (wrist_contact_steps > 0) & ~m["wrist_loaded"]
                    wrist_contact_steps += active & m["wrist_loaded"]
                    if "support_streak" in m:
                        support_streak_peak = torch.maximum(support_streak_peak, m["support_streak"] * active)
                        measured_push |= active & m["bilateral_push"]
                    loaded = active[:, None] & (m["foot_ground_patch_N"] > 8.0)
                    foot_load_samples += loaded
                    slip_sum += m["foot_ground_slip_m_s"] * loaded
                    outward_peak = torch.maximum(outward_peak, m["foot_ground_outward_m_s"] * loaded)
                    knee_hold = torch.where(active & m["right_knee_supported"], knee_hold + 1, 0)
                    knee_hold_peak = torch.maximum(knee_hold_peak, knee_hold)
                    knee_force_peak = torch.maximum(knee_force_peak, m["right_knee_top_load_N"] * active)
                newly_ready = active & m["gate_allowed"] & ~gate_ever
                first_gate_time[newly_ready] = (step + 1) * env.step_dt
                gate_ever[active] |= m["gate_allowed"][active]
                finished = active & done.bool()
                for k in finished.nonzero().flatten().tolist():
                    episodes.append(
                        {
                            "environment": k,
                            "steps": step + 1,
                            "left_swing_seen": bool(m["left_swing_seen"][k]),
                            "walking_samples": int(walk_count[k]),
                            "walking_wrist_error_max_rad": float(walk_wrist_peak[k])
                            if walk_count[k]
                            else None,
                            "walking_velocity_error_mean_m_s": float(
                                walk_velocity_sum[k] / walk_count[k]
                            )
                            if walk_count[k]
                            else None,
                            "walking_trunk_tilt_max_rad": float(walk_tilt_peak[k])
                            if walk_count[k]
                            else None,
                            "walking_foot_reference_error_max_m": float(walk_foot_peak[k])
                            if walk_count[k]
                            else None,
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
                    if env.knee_entry:
                        episodes[-1].update(
                            episode_return=float(episode_return[k]),
                            right_knee_supported_max_s=float(knee_hold_peak[k]) * env.step_dt,
                            right_knee_peak_top_load_N=float(knee_force_peak[k]),
                            first_step_wrist_top_load_N=first_wrist_load[k].tolist(),
                            sustained_wrist_support_failure=bool(m["wrist_support_failure"][k]),
                            early_left_knee=bool(m["left_knee_early"][k]),
                            early_right_knee=bool(m["right_knee_early"][k]),
                            wrist_contact_steps=int(wrist_contact_steps[k]),
                            wrist_unloaded_after_contact=bool(wrist_unloaded[k]),
                            loaded_foot_samples_lr=foot_load_samples[k].int().tolist(),
                            mean_loaded_foot_slip_lr_m_s=(slip_sum[k] / foot_load_samples[k].clamp_min(1)).tolist(),
                            peak_loaded_foot_outward_lr_m_s=outward_peak[k].tolist(),
                            takeoff_observed=bool(m["takeoff_observed"][k]),
                            takeoff_assumed_at_reset=bool(m["takeoff_assumed"][k]),
                        )
                        if "support_streak" in m:
                            episodes[-1].update(
                                simultaneous_support_max_s=float(support_streak_peak[k]) * env.step_dt,
                                assisted_push_observed=bool(measured_push[k]),
                            )
                active[finished] = False
                if not active.any():
                    break
        count = sum(e["success"] for e in episodes)
        return {
            "start_phase": start_phase,
            "sampled_actions": sampled,
            "required_hold_steps": env.required_hold_steps,
            "required_hold_s": env.required_hold_steps * env.step_dt,
            "completed_episodes": len(episodes),
            "successes": count,
            "success_fraction": count / max(1, env.num_envs),
            "unfinished": int(active.sum()),
            "scope": "Fixed scene, first episode per clone",
            "episodes": episodes,
        }
    finally:
        env.cfg.end_hold_steps = original_hold_steps
        env.training_mode, env.use_curriculum, env.evaluation_start_phase, env.cfg.random_start = (
            old
        )
