"""First-episode walking evaluation and block-wise checkpoint selection."""

import json

import torch

from .evaluation import evaluate
from .training_control import save_checkpoint_atomically


@torch.inference_mode()
def evaluate_walk(runner, wrapped):
    env = wrapped.env
    old_fixed = env.fixed_commands
    env.fixed_commands = True
    runner.eval_mode()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    records = []
    try:
        # Stand/flip regression uses the existing five-second criterion and the same policy.
        env.prefix_test = True
        prefix = evaluate(runner, wrapped, 0.0)["success_fraction"]
        env.prefix_test = False
        wrapped.obs, _ = env.reset()
        for _ in range(env.max_episode_length + 2):
            _, _, done, _ = wrapped.step(runner.alg.policy.act_inference(wrapped.obs["policy"]))
            finished = active & done.bool()
            if finished.any():
                ids = finished.nonzero().flatten()
                values = {k: v[ids].tolist() for k, v in env.last_metrics.items()}
                records.extend(dict(zip(values, row)) for row in zip(*values.values()))
                active &= ~finished
            if not active.any():
                break
        if len(records) != env.num_envs:
            raise RuntimeError("Walking evaluation ended before every first episode completed")
    finally:
        env.prefix_test = False
        env.fixed_commands = old_fixed

    def mean(key):
        return sum(r[key] for r in records) / len(records)

    summary = {
        "episodes": len(records),
        "flip_hold_5s_fraction": prefix,
        "walk_success_fraction": mean("success"),
        "flip_released_fraction": mean("walking_started"),
        "survived_fraction": 1 - mean("terminated"),
        "alternating_steps": mean("left_steps") + mean("right_steps"),
        "forward_distance_m": mean("distance_m"),
        "halt_hold_s": mean("halt_hold_s"),
        "lost_flip_fraction": mean("lost_flip"),
    }
    for part in ("first", "second"):
        n = sum(r[part + "_samples"] for r in records)
        summary[part + "_speed_mae_m_s"] = (
            sum(r[part + "_speed_mae"] * r[part + "_samples"] for r in records) / n if n else None
        )
    return {"summary": summary, "episodes": records}


def train_walk(runner, wrapped, args, output):
    contract = {
        "segment": "walk",
        "protocol": "flat_velocity_walk_v1",
        "height_source": "ground_truth_boxes",
        "gripper_added_mass_kg": wrapped.env.gripper_mass_report["added_mass_kg"],
        "transfer": wrapped.env.transfer_report,
    }
    (output / "training_contract.json").write_text(json.dumps(contract, indent=2))
    runner.logger_type = "tensorboard"
    completed = 0
    best = None
    while True:
        result = evaluate_walk(runner, wrapped)
        (output / f"walk_eval_{completed:05d}.json").write_text(json.dumps(result, indent=2))
        s = result["summary"]
        print("WALK_EVAL", completed, json.dumps(s), flush=True)
        if s["flip_hold_5s_fraction"] < 0.95:
            raise RuntimeError("Standing/flip regression below 95%; see walking evaluation")
        quality = (
            s["walk_success_fraction"],
            s["alternating_steps"],
            s["survived_fraction"],
            s["halt_hold_s"],
            s["forward_distance_m"],
        )
        infos = {"completed_iterations": completed, "contract": contract}
        save_checkpoint_atomically(runner, output / "latest.pt", infos)
        if best is None or quality > best:
            best = quality
            save_checkpoint_atomically(runner, output / "best.pt", infos)
        if completed == args.iterations:
            return
        with torch.inference_mode():
            wrapped.obs, _ = wrapped.env.reset()
        block = min(args.eval_interval, args.iterations - completed)
        runner.current_learning_iteration = completed
        runner.learn(num_learning_iterations=block, init_at_random_ep_len=False)
        completed += block
        runner.current_learning_iteration = completed
