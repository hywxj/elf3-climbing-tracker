"""Measure the fixed-pose policy initialization with and without action exploration."""

import argparse
import json
import hashlib
from elf3_climbing.paths import OUTPUT
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=128)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.output = args.output.expanduser().resolve()
if not args.output.is_relative_to(OUTPUT.resolve()):
    p.error(f"Output must be inside {OUTPUT}")
if args.output.exists():
    p.error("Choose a new output directory")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import torch
from elf3_climbing.environment import TrackerCfg, TrackerEnv, __file__ as environment_file

cfg = TrackerCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.segment = "flip"
env = TrackerEnv(cfg)
env.evaluation_start_phase = 0.9
cases = []
try:
    with torch.inference_mode():
        for sigma, wrist_sigma in [(0.0, 0.0), (0.08, 0.03), (0.03, 0.01), (0.01, 0.005)]:
            torch.manual_seed(42)
            env.reset()
            mean = (env.ref["joint_pos"][env.segment_end] - env.nominal) / cfg.action_scale
            std = torch.full_like(mean, sigma)
            std[:, env.wrist_joint] = wrist_sigma
            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            records = []
            for step in range(env.max_episode_length + 2):
                _, _, terminated, truncated, _ = env.step(mean + torch.randn_like(mean) * std)
                finished = active & (terminated | truncated)
                m = env.last_metrics
                for k in finished.nonzero().flatten().tolist():
                    records.append(
                        {
                            "success": bool(m["success"][k]),
                            "steps": step + 1,
                            "wrist_error_rad": float(m["gripper_error_rad"][k]),
                            "wrist_speed_rad_s": float(m["gripper_speed_rad_s"][k]),
                            "root_error_m": float(m["root_error_m"][k]),
                            "foot_forces_N": m["foot_force_N"][k].tolist(),
                            "premature_advance": bool(m["premature_advance"][k]),
                            "gripper_collision": bool(m["gripper_collision"][k]),
                            "lost_flip": bool(m["lost_flip"][k]),
                            "failure_root": bool(m["failure_root"][k]),
                            "failure_orientation": bool(m["failure_orientation"][k]),
                        }
                    )
                active[finished] = False
                if not active.any():
                    break
            summary = {
                "action_std": sigma,
                "wrist_std": wrist_sigma,
                "episodes": len(records),
                "successes": sum(x["success"] for x in records),
                "mean_terminal_wrist_error_rad": sum(x["wrist_error_rad"] for x in records)
                / len(records),
                "mean_terminal_wrist_speed_rad_s": sum(x["wrist_speed_rad_s"] for x in records)
                / len(records),
                "mean_episode_steps": sum(x["steps"] for x in records) / len(records),
            }
            print(json.dumps(summary), flush=True)
            cases.append({**summary, "records": records})
    (args.output / "exploration.json").write_text(
        json.dumps(
            {
                "mass_kg": 0.5,
                "phase": 0.9,
                "optimizer_updates": 0,
                "cases": cases,
                "end_hold_steps": cfg.end_hold_steps,
                "environment_sha256": hashlib.sha256(
                    Path(environment_file).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
finally:
    env.close()
    app.close()
