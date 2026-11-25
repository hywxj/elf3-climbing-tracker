"""Repeated learned-policy rollouts with wall-clock pacing; no optimizer updates."""

import csv
from contextlib import ExitStack
import json
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def knee_frame(metrics, env, episode, step):
    dt = env.step_dt
    m = metrics
    faults = {
        "sustained_wrist_support": "wrist_support_failure",
        "gripper_collision": "gripper_collision",
        "lost_flip": "lost_flip",
        "early_left_knee": "left_knee_early",
        "early_right_knee": "right_knee_early",
        "unexpected_contact": "failure_forbidden_contact",
        "root_position": "failure_root",
        "root_orientation": "failure_orientation",
        "joint_tracking": "failure_joint",
        "premature_advance": "premature_advance",
        "premature_arm_contact": "premature_arm_contact",
    }
    reasons = [label for label, key in faults.items() if bool(m[key][0])]
    if bool(m["success"][0]):
        reasons.append("success")
    elif bool(m["expired"][0]) and not reasons:
        reasons.append("timeout")
    peak = int(m["forbidden_body_force_N"][0].argmax())
    body = env.contacts.body_names[env.forbidden_sensor_ids[peak]]
    row = {
        "episode": episode,
        "time_s": round(step * dt, 4),
        "reference_frame": int(m["reference_frame"][0]),
        "left_forearm_N": float(m["forearm_top_load_N"][0, 0]),
        "right_forearm_N": float(m["forearm_top_load_N"][0, 1]),
        "left_foot_ground_N": float(m["foot_ground_load_N"][0, 0]),
        "right_foot_ground_N": float(m["foot_ground_load_N"][0, 1]),
        "left_foot_patch_N": float(m["foot_ground_patch_N"][0, 0]),
        "right_foot_patch_N": float(m["foot_ground_patch_N"][0, 1]),
        "left_foot_slip_m_s": float(m["foot_ground_slip_m_s"][0, 0]),
        "right_foot_slip_m_s": float(m["foot_ground_slip_m_s"][0, 1]),
        "left_foot_outward_m_s": float(m["foot_ground_outward_m_s"][0, 0]),
        "right_foot_outward_m_s": float(m["foot_ground_outward_m_s"][0, 1]),
        "com_vz_m_s": float(m["com_vz_m_s"][0]),
        "bilateral_push": bool(m["bilateral_push"][0]),
        "support_rule": env.cfg.knee_support_rule,
        "min_foot_clearance_m": float(m["min_foot_clearance_m"][0]),
        "both_feet_airborne": bool((m["max_foot_load_N"][0] < 3.0) & (m["min_foot_clearance_m"][0] > 0.01)),
        "takeoff_observed": bool(m["takeoff_observed"][0]),
        "takeoff_assumed": bool(m["takeoff_assumed"][0]),
        "right_knee_top_N": float(m["right_knee_top_load_N"][0]),
        "right_knee_supported": bool(m["right_knee_supported"][0]),
        "wrist_loaded": bool(m["wrist_loaded"][0]),
        "left_wrist_platform_N": float(m["wrist_platform_force_N"][0, 0]),
        "right_wrist_platform_N": float(m["wrist_platform_force_N"][0, 1]),
        "left_wrist_y_platform_N": float(m["wrist_y_platform_force_N"][0, 0]),
        "right_wrist_y_platform_N": float(m["wrist_y_platform_force_N"][0, 1]),
        "wrist_contact_rule": env.cfg.knee_wrist_contact,
        "peak_monitored_body": body,
        "peak_monitored_body_N": float(m["forbidden_body_force_N"][0, peak]),
        "gripper_clearance_m": float(m["gripper_clearance_m"][0]),
        "stable_hold_s": round(float(m["stable_hold_steps"][0]) * dt, 4),
        "reason": "|".join(reasons),
    }
    for name in ("hip_height_above_platform_m", "right_knee_inside_edge_m"):
        if name in m:
            row[name] = float(m[name][0])
    wrist_tabletop = m.get("wrist_tabletop_load_N")
    if wrist_tabletop is not None:
        row["left_wrist_tabletop_N"] = float(wrist_tabletop[0, 0])
        row["right_wrist_tabletop_N"] = float(wrist_tabletop[0, 1])
    if "support_streak" in m:
        row.update(left_arm_support_N=float(m["arm_support_top_N"][0, 0]),
                   right_arm_support_N=float(m["arm_support_top_N"][0, 1]),
                   transfer_lost=bool(m["transfer_lost"][0]),
                   knee_bearing=bool(m["knee_bearing"][0]),
                   wrist_forbidden=bool(m["wrist_forbidden"][0]),
                   simultaneous_support=bool(m["simultaneous_support"][0]),
                   hip_forward_m_s=float(m["hip_forward_m_s"][0]), hip_vz_m_s=float(m["hip_vz_m_s"][0]))
    return row


def knee_load_summary(rows, push_start, step_dt, body_weight):
    """Summarize measured loads on all samples of each phase, including loss of support."""
    phases = {"preload": [], "push": [], "transfer": [], "knee": []}
    launched = False
    knee_reached = False
    for row in rows:
        # A failed push can still lift both feet. Keep its unsupported flight in the report.
        launched |= row["both_feet_airborne"] or row["takeoff_assumed"]
        knee_reached |= launched and row["right_knee_top_N"] > 8.0
        if knee_reached:
            phase = "knee"
        elif launched:
            phase = "transfer"
        elif row["reference_frame"] >= push_start:
            phase = "push"
        else:
            phase = "preload"
        phases[phase].append(row)

    lowest_hip = rows[0]["hip_height_above_platform_m"]
    max_rise = 0.0
    for row in rows:
        height = row["hip_height_above_platform_m"]
        lowest_hip = min(lowest_hip, height)
        max_rise = max(max_rise, height - lowest_hip)
    summary = {
        "body_weight_N": body_weight, "phases": {},
        "max_hip_rise_from_prior_low_m": max_rise,
        "peak_hip_height_above_platform_m": max(row["hip_height_above_platform_m"] for row in rows),
        "peak_right_knee_inside_edge_m": max(row["right_knee_inside_edge_m"] for row in rows),
    }
    for name, samples in phases.items():
        stage = {"samples": len(samples), "duration_s": len(samples) * step_dt}
        summary["phases"][name] = stage
        if not samples:
            continue
        loads = ["left_forearm_N", "right_forearm_N", "left_foot_patch_N",
                 "right_foot_patch_N", "right_knee_top_N"]
        if "left_wrist_tabletop_N" in samples[0]:
            loads.extend(["left_wrist_tabletop_N", "right_wrist_tabletop_N"])
        stage["mean_load_N"] = {
            key: sum(row[key] for row in samples) / len(samples) for key in loads
        }
        stage["mean_right_knee_body_weight_fraction"] = stage["mean_load_N"]["right_knee_top_N"] / body_weight
        stage["both_forearms_above_3N_fraction"] = sum(
            min(row["left_forearm_N"], row["right_forearm_N"]) > 3.0 for row in samples
        ) / len(samples)
        stage["both_feet_airborne_fraction"] = sum(row["both_feet_airborne"] for row in samples) / len(samples)
    return summary


def play(runner, wrapped, app, args, output):
    env = wrapped.env
    if env.num_envs != 1:
        raise ValueError("Playback uses one environment for an unambiguous camera and episode log")
    env.training_mode = False
    env.use_curriculum = False
    env.evaluation_start_phase = args.play_start_phase
    env.cfg.random_start = False
    if env.knee_entry:
        env.knee_contacts.enable_pose_trace()
    runner.eval_mode()
    offsets = {
        "rear-left": [-2.5, 2.0, 1.2],
        "front": [2.8, 0.8, 1.4],
        "left": [0.0, 3.0, 0.9],
        "right": [0.0, -3.0, 0.9],
    }
    episodes = 0
    successes = 0
    steps = 0
    peak_root = 0.0
    interrupted = False
    knee_rows = []
    last_load_summary = None
    with torch.inference_mode():
        wrapped.obs, _ = env.reset()
        origin = env.robot.data.root_link_pos_w[0].cpu().numpy()
        quat = env.robot.data.root_link_quat_w[0].cpu().numpy()
        yaw = Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")[2]
        eye = origin + Rotation.from_euler("z", yaw).apply(offsets[args.view])
        target = origin + np.array([0.0, 0.0, -0.2])
        env.sim.set_camera_view(eye=eye, target=target)
        try:
            with ExitStack() as files:
                log = files.enter_context(
                    (output / "playback_episodes.jsonl").open("w", buffering=1)
                )
                trace = None
                if env.knee_entry:
                    trace = files.enter_context(
                        (output / "knee_contacts.csv").open("w", newline="", buffering=1)
                    )
                writer = None
                while app.is_running() and (
                    args.play_episodes == 0 or episodes < args.play_episodes
                ):
                    start = time.monotonic()
                    if not env.sim.is_playing():
                        env.sim.render()
                        time.sleep(0.02)
                        continue
                    obs = runner.obs_normalizer(wrapped.obs["policy"])
                    action = runner.alg.policy.act_inference(obs)
                    _, _, done, _ = wrapped.step(action)
                    steps += 1
                    m = env.last_metrics
                    peak_root = max(peak_root, float(m["root_error_m"][0]))
                    if trace is not None:
                        row = knee_frame(m, env, episodes + 1, steps)
                        if writer is None:
                            writer = csv.DictWriter(trace, fieldnames=row)
                            writer.writeheader()
                        writer.writerow(row)
                        knee_rows.append(row)
                    if bool(done[0]):
                        episodes += 1
                        success = bool(m["success"][0])
                        successes += int(success)
                        record = {
                            "episode": episodes,
                            "success": success,
                            "steps": steps,
                            "duration_s": steps * env.step_dt,
                            "root_error_max_m": peak_root,
                            "terminal_gripper_error_rad": float(m["gripper_error_rad"][0]),
                            "gripper_collision": bool(m["gripper_collision"][0]),
                            "premature_advance": bool(m["premature_advance"][0]),
                            "lost_flip": bool(m["lost_flip"][0]),
                        }
                        if trace is not None:
                            last_load_summary = knee_load_summary(
                                knee_rows, env.knee_contacts.push_start, env.step_dt,
                                float(env.knee_contacts.total_mass[0]) * abs(env.cfg.sim.gravity[2]),
                            )
                            knee_rows.clear()
                            record.update(
                                contact_loads=last_load_summary,
                                start_phase=args.play_start_phase,
                                reset_frame=int(m["reset_frame"][0]),
                                final_reference_frame=row["reference_frame"],
                                reason=row["reason"],
                                peak_monitored_body=row["peak_monitored_body"],
                                peak_monitored_body_N=row["peak_monitored_body_N"],
                            )
                        log.write(json.dumps(record) + "\n")
                        print("PLAY_EPISODE", json.dumps(record), flush=True)
                        steps, peak_root = 0, 0.0
                    # Slow playback changes wall-clock pacing, never physics dt or policy rate.
                    remaining = env.step_dt / args.play_speed - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
    result = {
        "episodes": episodes,
        "successes": successes,
        "optimizer_updates": 0,
        "checkpoint": str(args.checkpoint),
        "play_speed": args.play_speed,
        "interrupted": interrupted,
        "start_phase": args.play_start_phase,
        "scope": "Deterministic policy playback starting at the requested reference phase",
    }
    if last_load_summary is not None:
        result["last_completed_episode_contact_loads"] = last_load_summary
    (output / "playback.json").write_text(json.dumps(result, indent=2))
    return result
