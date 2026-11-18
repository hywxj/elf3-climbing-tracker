"""Read-only checkpoint/physics diagnosis; no optimization or hardware access."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher
from elf3_climbing.paths import OUTPUT

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--phase", type=float, default=0.9)
p.add_argument("--hold-seconds", type=float, default=1.0)
p.add_argument("--learned-only", action="store_true")
p.add_argument("--steps", type=int, default=350)
p.add_argument("--model-audit", action="store_true")
p.add_argument(
    "--feedback-check",
    action="store_true",
    help="Diagnostic root pitch feedback; not a learned policy",
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.output = args.output.expanduser().resolve()
if args.output.exists() or not args.output.is_relative_to(OUTPUT.resolve()):
    p.error("Choose a new directory under runtime/runs")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import torch
from elf3_climbing.environment import TrackerCfg, TrackerEnv
from elf3_climbing.runner import LegacyRslAdapter, runner_config
from elf3_climbing.initialization import static_action_bias
from rsl_rl.runners import OnPolicyRunner

cfg = TrackerCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
cfg.segment = "flip"
cfg.end_hold_steps = round(args.hold_seconds / 0.02)
cfg.flip_hold_steps = cfg.end_hold_steps
cfg.extra_budget_steps = max(100, cfg.end_hold_steps + 100)
if args.feedback_check:
    cfg.end_hold_steps = 250
    cfg.flip_hold_steps = 250
    cfg.extra_budget_steps = 350
env = TrackerEnv(cfg)
wrapped = LegacyRslAdapter(env)
runner = OnPolicyRunner(wrapped, runner_config(), log_dir=None, device=env.device)
runner.load(str(args.checkpoint), load_optimizer=False)
runner.eval_mode()
env.evaluation_start_phase = args.phase
names = env.reference_np["joint_names"].tolist()
bodies = env.robot.body_names
sensor_names = env.contacts.body_names
snapshot = {}
original_dones = env._get_dones


def traced_dones():
    result = original_dones()
    d = env.robot.data
    snapshot.clear()
    snapshot.update(
        {
            "root_pos": (d.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().tolist(),
            "root_quat": d.root_link_quat_w[0].cpu().tolist(),
            "root_lin_vel": d.root_link_lin_vel_w[0].cpu().tolist(),
            "root_ang_vel": d.root_ang_vel_w[0].cpu().tolist(),
            "q": d.joint_pos[0, env.joint_ids].cpu().tolist(),
            "qd": d.joint_vel[0, env.joint_ids].cpu().tolist(),
            "targets": env.targets[0].cpu().tolist(),
            "torque": d.applied_torque[0, env.joint_ids].cpu().tolist(),
            "body_contact_force": env.contacts.data.net_forces_w[0].cpu().tolist(),
            "body_pos": (d.body_link_pos_w[0] - env.scene.env_origins[0]).cpu().tolist(),
            "body_com_pos": (d.body_com_pos_w[0] - env.scene.env_origins[0]).cpu().tolist(),
            "support_force": env.support_forces()[0].cpu().tolist(),
            "done": bool(result[0][0] or result[1][0]),
            "success": bool(env.last_metrics["success"][0]),
            "forbidden": bool(env.last_metrics["failure_forbidden_contact"][0]),
            "gripper_collision": bool(env.last_metrics["gripper_collision"][0]),
        }
    )
    return result


env._get_dones = traced_dones
try:
    with torch.inference_mode():
        if args.model_audit:
            from elf3_climbing.calibration import audit_balance_model

            audit = audit_balance_model(env)
            (args.output / "model_audit.json").write_text(json.dumps(audit, indent=2))
            print("MODEL_AUDIT", audit["status"], audit["frames"], flush=True)
        static_bias, static_report = static_action_bias(env)
        seed = torch.as_tensor(static_bias, device=env.device, dtype=torch.float32)[None]
        fixed = (env.ref["joint_pos"][env.segment_end] - env.nominal) / cfg.action_scale
        all_cases = {}
        cases = (
            ("learned",)
            if args.learned_only
            else ("fixed_reference_bias", "gravity_bias", "learned")
        )
        if args.feedback_check:
            cases += (
                "pitch_feedback_3_1",
                "pitch_feedback_5_1",
                "com_feedback_4_08",
                "com_feedback_6_15",
            )
        for case in cases:
            wrapped.obs, _ = env.reset()
            rows = []
            masses = env.robot.root_physx_view.get_masses().to(env.device)
            weights = masses / masses.sum(dim=1, keepdim=True)
            initial_com = (env.robot.data.body_com_pos_w * weights[..., None]).sum(dim=1)
            initial_gravity = env.robot.data.projected_gravity_b.clone()
            for step in range(args.steps):
                action = (
                    runner.alg.policy.act_inference(runner.obs_normalizer(wrapped.obs["policy"]))
                    if case == "learned"
                    else seed
                    if case == "gravity_bias"
                    else fixed
                )
                if case.startswith("pitch_feedback"):
                    kp = 3.0 if case == "pitch_feedback_3_1" else 5.0
                    action = seed.clone()
                    correction = (
                        kp * (env.robot.data.projected_gravity_b[:, 0] - initial_gravity[:, 0])
                        + env.robot.data.root_ang_vel_b[:, 1]
                    )
                    for joint in ("l_ankle_y_joint", "r_ankle_y_joint"):
                        action[:, names.index(joint)] += correction
                if case.startswith("com_feedback"):
                    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

                    com = (env.robot.data.body_com_pos_w * weights[..., None]).sum(dim=1)
                    velocity = (env.robot.data.body_com_lin_vel_w * weights[..., None]).sum(dim=1)
                    rotation = yaw_quat(env.robot.data.root_link_quat_w)
                    error = quat_apply_inverse(rotation, com - initial_com)
                    velocity = quat_apply_inverse(rotation, velocity)
                    kp, kd = (4.0, 0.8) if case == "com_feedback_4_08" else (6.0, 1.5)
                    action = seed.clone()
                    for side in ("l", "r"):
                        action[:, names.index(side + "_ankle_y_joint")] += (
                            kp * error[:, 0] + kd * velocity[:, 0]
                        )
                        action[:, names.index(side + "_ankle_x_joint")] -= 1.8 * (
                            kp * error[:, 1] + kd * velocity[:, 1]
                        )
                wrapped.step(action)
                rows.append({"step": step + 1, "time_s": (step + 1) * env.step_dt, **snapshot})
                if snapshot["done"]:
                    break
            all_cases[case] = rows
            end = rows[-1]
            print(
                case,
                "steps",
                len(rows),
                "root_pos",
                end["root_pos"],
                "root_vel",
                end["root_lin_vel"],
                "forbidden",
                end["forbidden"],
                flush=True,
            )
        report = {
            "checkpoint": str(args.checkpoint),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "phase": args.phase,
            "feedback_check": args.feedback_check,
            "end_hold_steps": cfg.end_hold_steps,
            "joint_names": names,
            "body_names": bodies,
            "contact_sensor_body_names": sensor_names,
            "joint_kp": env.robot.data.joint_stiffness[0, env.joint_ids].cpu().tolist(),
            "joint_kd": env.robot.data.joint_damping[0, env.joint_ids].cpu().tolist(),
            "body_masses": env.robot.root_physx_view.get_masses()[0].cpu().tolist(),
            "reference_root": env.ref["root_pos"][env.segment_end].cpu().tolist(),
            "reference_joint": env.ref["joint_pos"][env.segment_end].cpu().tolist(),
            "static_seed": static_report,
            "cases": all_cases,
            "scope": "First physical episode, fixed scene; diagnosis only, no PPO updates",
        }
        (args.output / "balance_trace.json").write_text(json.dumps(report, indent=2))
finally:
    env.close()
    app.close()
