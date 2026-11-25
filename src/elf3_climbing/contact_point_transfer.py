"""Transfer an existing stable walking actor into the terrain-point actor's shared motor layers."""

import argparse
import json
from pathlib import Path

import torch

from .contact_anchor import anchor_observation, far_ground_weight
from .contact_edge import extend_weights
from .contact_gait_actor import GaitContactActor
from .contact_learning import fit_actor, load_examples, evaluate_actor
from .contact_point_actor import PointContactActor, evaluate_points


def contexts(actor, source, obs):
    original = anchor_observation(actor, source, obs)
    hidden = source.encoder(((original - source.obs_mean) / source.obs_scale).clamp(-8, 8))
    soft_plan = source.planner(hidden).reshape(-1, 8, 4)
    probability = soft_plan[:, :, 0].sigmoid()
    old = torch.cat(
        (hidden, probability, (soft_plan[:, :, 1:] * probability[:, :, None]).flatten(1)), 1
    )
    _, logits, xyz = actor.action_and_plan(obs)
    new = torch.cat((actor.point_state(obs), (logits >= 0).float(), xyz.flatten(1)), 1)
    return old, new


def reset_ground(obs, far):
    return (obs[:, 262:646].abs().amax(1) == 0) & (far > 0.7)


@torch.no_grad()
def align_motor_context(actor, source, train, keep_near=False):
    """Refit only the two linear motor inputs after replacing soft coordinates with visible cells."""
    actor.eval()
    gram = torch.eye(161, dtype=torch.float64) * 0.03
    cross = torch.zeros(161, 131, dtype=torch.float64)
    for obs in train[0].split(256):
        old, new = contexts(actor, source, obs)
        old_projection = torch.cat((source.command(old), source.context(old)), 1)
        projection = torch.cat((actor.command(new), actor.context(new)), 1)
        far = far_ground_weight(obs)
        reset = reset_ground(obs, far)
        # During final calibration the trained predictions near the ledge remain the target.
        change = old_projection - projection
        if keep_near:
            change *= far[:, None]
        importance = (1 + 16 * far + 500 * reset).double()
        x = torch.cat((new, torch.ones(len(obs), 1)), 1).double()
        weighted = x.T * importance
        gram += weighted @ x
        cross += weighted @ change.double()
    difference = torch.linalg.solve(gram, cross).float().T
    for layer, correction in zip((actor.command, actor.context), difference.split((3, 128))):
        layer.weight.add_(correction[:, :-1])
        layer.bias.add_(correction[:, -1])


@torch.no_grad()
def action_agreement(actor, source, data):
    actor.eval()
    values = {"first_ground": [], "far_ground": []}
    for obs in data[0].split(256):
        far = far_ground_weight(obs)
        error = (actor(obs) - source(anchor_observation(actor, source, obs))).abs()
        for group, mask in (("first_ground", reset_ground(obs, far)), ("far_ground", far >= 0.8)):
            values[group].append(error[mask])
    report = {}
    for group, arrays in values.items():
        err = torch.cat(arrays)
        if len(err):
            report[group] = {
                "frames": len(err),
                "mean_absolute_rad": err.mean().item(),
                "p95_absolute_rad": err.flatten().quantile(0.95).item(),
                "max_absolute_rad": err.max().item(),
            }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-report", type=Path, required=True)
    p.add_argument("--gait", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    paths = json.loads(args.previous_report.read_text())["samples"]
    data = load_examples(paths, edge=2, point=True)
    source = GaitContactActor()
    weights = torch.load(args.gait, map_location="cpu", weights_only=True)
    source.load_state_dict(weights)
    source.requires_grad_(False).eval()
    actor = PointContactActor(body_frame_input=True)
    extend_weights(actor, weights)
    history = data["train"][0][:, 262:646]
    actor.obs_mean[262:646] = history.mean(0)
    actor.obs_scale[262:646] = history.std(0).clamp_min(0.08)
    result = {"gait": str(args.gait), "samples": paths}
    result["raw_copy"] = action_agreement(actor, source, data["validation"])
    align_motor_context(actor, source, data["train"])
    result["aligned_copy"] = action_agreement(actor, source, data["validation"])
    # Continue learning in the same single motor network. Keep the transferred servo weights
    # at their source values for this short supervised pass, then make them trainable for PPO.
    actor.backbone.requires_grad_(False)
    result["training"] = fit_actor(
        actor,
        data,
        epochs=args.epochs,
        lr=0.00018,
        batch_size=128,
        fit_normalization=False,
        gait_anchor=source,
    )
    actor.backbone.requires_grad_(True)
    result["after_fit"] = action_agreement(actor, source, data["validation"])
    align_motor_context(actor, source, data["train"], keep_near=True)
    result["after_calibration"] = action_agreement(actor, source, data["validation"])
    result["validation"] = evaluate_actor(actor, data["validation"])
    result["point_validation"] = evaluate_points(actor, data["validation"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".pt.tmp")
    torch.save(actor.state_dict(), tmp)
    tmp.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k in ("raw_copy", "aligned_copy", "after_fit", "after_calibration")
        }
    )


if __name__ == "__main__":
    main()
