"""Repeated learned-policy rollouts with wall-clock pacing; no optimizer updates."""

import json
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def play(runner, wrapped, app, args, output):
    env = wrapped.env
    if env.num_envs != 1:
        raise ValueError("Playback uses one environment for an unambiguous camera and episode log")
    env.training_mode = False
    env.use_curriculum = False
    env.evaluation_start_phase = 0.0
    env.cfg.random_start = False
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
    with torch.inference_mode():
        wrapped.obs, _ = env.reset()
        origin = env.robot.data.root_link_pos_w[0].cpu().numpy()
        quat = env.robot.data.root_link_quat_w[0].cpu().numpy()
        yaw = Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")[2]
        eye = origin + Rotation.from_euler("z", yaw).apply(offsets[args.view])
        target = origin + np.array([0.0, 0.0, -0.2])
        env.sim.set_camera_view(eye=eye, target=target)
        try:
            with (output / "playback_episodes.jsonl").open("w", buffering=1) as log:
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
        "scope": "Repeated deterministic policy execution from segment start; not robustness validation",
    }
    (output / "playback.json").write_text(json.dumps(result, indent=2))
    return result
