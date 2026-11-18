"""Optional simulation interface diagnostic; never a training prerequisite."""

import json
import numpy as np
import torch


def run_smoke(env, wrapped, args, output):
    rows = []
    trace = {
        k: [] for k in ["reference_frame", "root_pose", "joint_pos", "joint_vel", "foot_force_N"]
    }
    done_trace = []
    for step in range(args.steps):
        actions = (
            env.reference_pd_action()
            if args.smoke_controller == "reference-pd"
            else torch.zeros((env.num_envs, 29), device=env.device)
        )
        with torch.no_grad():
            obs, reward, done, extras = wrapped.step(actions)
        if not (
            torch.isfinite(obs).all()
            and torch.isfinite(reward).all()
            and torch.isfinite(extras["observations"]["critic"]).all()
        ):
            raise RuntimeError("Nonfinite observation or reward")
        if step < 10:
            net = torch.linalg.vector_norm(
                env.contacts.data.net_forces_w[:, env.sensor_ids[:2]], dim=-1
            )
            filtered = env.last_metrics["foot_force_N"]
            if ((net > 20) & (filtered < 1)).any():
                raise RuntimeError(
                    "Terrain contact filter reports zero despite loaded feet; do not use these force labels"
                )
        for k in trace:
            trace[k].append(env.last_metrics[k].cpu().numpy())
        done_trace.append(done.cpu().numpy())
        rows.append(
            {
                "step": step,
                "mean_reward": float(reward.mean()),
                "done_count": int(done.sum()),
                "root_error_max_m": float(env.last_metrics["root_error_m"].max()),
                "joint_rmse_max_rad": float(env.last_metrics["joint_rmse_rad"].max()),
                "torque_limit_ratio_max": float(env.last_metrics["torque_ratio"].max()),
                "orientation_error_max_rad": float(env.last_metrics["orientation_error_rad"].max()),
                "failure_orientation_count": int(env.last_metrics["failure_orientation"].sum()),
                "failure_root_count": int(env.last_metrics["failure_root"].sum()),
                "failure_forbidden_contact_count": int(
                    env.last_metrics["failure_forbidden_contact"].sum()
                ),
                "foot_force_min_N": float(env.last_metrics["foot_force_N"].min()),
            }
        )
    summary = {
        "interface_smoke": "PASS",
        "terrain_force_filter": "PASS_NONZERO_LOADED_FEET",
        "controller": args.smoke_controller,
        "steps": args.steps,
        "num_envs": env.num_envs,
        "physical_episodes": env.completed,
        "failed_episodes": env.failed,
        "successful_short_episodes": env.successful,
        "optimizer_updates": 0,
        "short_segment_tracking": "PASS"
        if env.completed > 0 and env.successful == env.completed
        else "FAIL",
        "full_climb": "NOT RUN",
        "real_lidar": "NOT CONNECTED",
        "scope": "Physics advanced with PD joint targets; root/joints written only at reset. This is not a learned policy.",
    }
    np.savez_compressed(
        output / "physics_rollout.npz",
        **{k: np.stack(v) for k, v in trace.items()},
        done=np.stack(done_trace),
        fps=np.array(50.0),
        joint_names=env.reference_np["joint_names"],
    )
    (output / "smoke.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2))
    print("CLIMBING_SMOKE_RESULT", json.dumps(summary))
