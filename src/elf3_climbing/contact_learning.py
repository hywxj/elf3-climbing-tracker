"""Learn joint drives and the next load-bearing contact from measured robot episodes."""

import copy
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from .contact_examples import CONTACT_NAMES

CONTACT_HORIZON_S = 1.5
CONTACT_OBS_DIM = 3 * 29 + 3 * 3 + 126 + 8 * 3 + 8


def _local(q, world):
    """Inverse rotation, quaternion wxyz; accepts one or several vectors per robot."""
    v = q[:, None, 1:]
    w = q[:, None, :1]
    cross = torch.cross(v.expand_as(world), world, dim=-1)
    return world + 2 * (torch.cross(v.expand_as(world), cross, dim=-1) - w * cross)


def encode_contact_observation(raw, nominal):
    """The same current-state encoding for the collector, PPO environment and offline reader."""
    sites = _local(raw["root_quat"], raw["site_pos"] - raw["root_pos"][:, None])
    forces = torch.log1p(raw["contact_force"].clamp_min(0) / 20)
    return torch.cat(
        (
            raw["joint_pos"] - nominal,
            raw["joint_vel"] * 0.05,
            raw["prev_target"] - nominal,
            raw["root_lin_vel_b"],
            0.25 * raw["root_ang_vel_b"],
            raw["projected_gravity_b"],
            raw["terrain"],
            sites.flatten(1),
            forces,
        ),
        dim=1,
    )


class ContactActor(nn.Module):
    """One shared, trainable actor. Contact probabilities gate explicit body-frame positions."""

    def __init__(self):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(CONTACT_OBS_DIM))
        self.register_buffer("obs_scale", torch.ones(CONTACT_OBS_DIM))
        self.encoder = nn.Sequential(
            nn.Linear(CONTACT_OBS_DIM, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU()
        )
        self.planner = nn.Linear(128, 8 * 4)
        self.motor = nn.Sequential(nn.Linear(128 + 8 * 4, 128), nn.ELU(), nn.Linear(128, 29))
        # Identity initialization holds the most recent real command. Its weights are trainable.
        self.drive_memory = nn.Linear(29, 29, bias=False)
        nn.init.eye_(self.drive_memory.weight)
        nn.init.zeros_(self.motor[-1].weight)
        nn.init.zeros_(self.motor[-1].bias)

    def action_and_plan(self, obs):
        state = self.encoder(((obs - self.obs_mean) / self.obs_scale).clamp(-8, 8))
        plan = self.planner(state).reshape(-1, 8, 4)
        logits, positions = plan[:, :, 0], plan[:, :, 1:]
        valid = logits.sigmoid().unsqueeze(-1)
        command = self.motor(
            torch.cat((state, valid.flatten(1), (valid * positions).flatten(1)), 1)
        )
        return self.drive_memory(obs[:, 58:87]) + command, logits, positions

    def forward(self, obs):
        return self.action_and_plan(obs)[0]


def upcoming_contacts(forces, points, step_dt, horizon_s=CONTACT_HORIZON_S, dwell_s=0.06):
    """Label each body’s next new load within 1.5 seconds of each frame. Ignore initial supports.

    Require consecutive load above threshold and re-arm after confirmed unloading. An earlier
    foot landing never hides a later arm or knee contact within the same planning window.
    """
    forces, points = np.asarray(forces), np.asarray(points)
    if (
        step_dt <= 0
        or forces.ndim != 2
        or forces.shape[1] != 8
        or points.shape != (*forces.shape, 3)
        or not np.isfinite(forces).all()
    ):
        raise ValueError("Invalid episode-wide contact forces/positions")
    n = len(forces)
    valid = np.zeros((n, 8), bool)
    locations = np.zeros((n, 8, 3), np.float32)
    dwell = max(1, int(np.ceil(dwell_s / step_dt - 1e-6)))
    horizon = max(1, round(horizon_s / step_dt))
    on = np.array([15, 15, 3, 3, 3, 3, 8, 8])
    off = on * 0.45
    events = []
    for body in range(8):
        loaded = forces[0, body] >= on[body] if n else False
        streak = 0
        for k, force in enumerate(forces[:, body]):
            state_change = force < off[body] if loaded else force >= on[body]
            streak = streak + 1 if state_change else 0
            if streak == dwell:
                if not loaded:
                    first = k - dwell + 1
                    pos = points[first : k + 1, body]
                    usable = np.isfinite(pos).all(1)
                    if usable.any():
                        events.append(
                            (
                                first,
                                body,
                                np.average(
                                    pos[usable], axis=0, weights=forces[first : k + 1, body][usable]
                                ),
                            )
                        )
                loaded = not loaded
                streak = 0
    events.sort(key=lambda e: (e[0], e[1]))
    previous = np.zeros(8, dtype=int)
    for frame, body, point in events:
        start = max(previous[body], frame - horizon)
        valid[start:frame, body] = True
        locations[start:frame, body] = point
        previous[body] = frame
    return valid, locations, [(int(i), CONTACT_NAMES[b]) for i, b, _ in events]


def plan_label_mask(events, step_dt, corrective):
    """Mark unobserved negatives beyond a cut trajectory and all corrective rollout plans unknown."""
    labels = events.float()
    if corrective:
        return torch.full_like(labels, -1)
    horizon_steps = round(CONTACT_HORIZON_S / step_dt)
    tail = labels[-horizon_steps:]
    tail.masked_fill_(tail == 0, -1)
    return labels


def _project_support(sites, centers, half, quat):
    """Intersect loaded landmark XY with a measured horizontal support plane (no force centroid)."""
    points = sites.copy()
    # Select the supporting surface closest to the current landmark Z. A foot can stand
    # under an overhanging corner; prefer the lower support when it is closer in height.
    zerr = np.full(sites.shape[:-1], np.inf)
    for c, h, q in zip(centers, half, quat):
        if abs(q[1]) + abs(q[2]) > 1e-4:
            raise ValueError("Contact labels currently require horizontal terrain boxes")
        yaw = 2 * np.arctan2(q[3], q[0])
        co, si = np.cos(yaw), np.sin(yaw)
        dx, dy = sites[..., 0] - c[0], sites[..., 1] - c[1]
        inside = (np.abs(co * dx + si * dy) <= h[0] + 0.04) & (
            np.abs(-si * dx + co * dy) <= h[1] + 0.04
        )
        top = c[2] + h[2]
        err = np.abs(top - sites[..., 2])
        good = inside & (err < zerr)
        points[..., 2][good] = top
        zerr[good] = err[good]
    points[~np.isfinite(zerr)] = np.nan
    return points


def _episode_batches(path, edge=False, point=False):
    with np.load(path, allow_pickle=False) as pack:
        raw = {
            k: pack[k]
            for k in (
                "joint_pos",
                "joint_vel",
                "prev_target",
                "root_lin_vel_b",
                "root_ang_vel_b",
                "projected_gravity_b",
                "terrain",
            )
        }
        poses = pack["root_pose"]
        raw["root_pos"], raw["root_quat"] = poses[:, :3], poses[:, 3:7]
        raw["site_pos"] = pack["student_site_pos"]
        raw["contact_force"] = pack["student_contact_force"]
        names = tuple(pack["joint_names"].astype(str))
        nominal = np.asarray(pack["nominal"]).reshape(29)
        if (
            "student_site_names" in pack
            and tuple(pack["student_site_names"].astype(str)) != CONTACT_NAMES
        ):
            raise ValueError(f"{path}: inconsistent contact channel names")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        raw_t = {k: tensor(v) for k, v in raw.items()}
        obs = encode_contact_observation(raw_t, tensor(nominal))
        action = tensor(pack["drive_target"] - nominal)
        if (
            obs.shape != (len(poses), CONTACT_OBS_DIM)
            or action.shape != (len(poses), 29)
            or not torch.isfinite(obs).all()
            or not torch.isfinite(action).all()
        ):
            raise ValueError(f"{path}: unexpected or nonfinite observation/target")
        if edge or point:
            from .contact_edge import simulated_edge_features

            geometry = [
                tensor(pack[k])
                for k in (
                    "terrain_box_centers_w",
                    "terrain_box_half_sizes",
                    "terrain_box_quat_wxyz",
                )
            ]
            if point:
                from .contact_point_actor import point_sensor

                parts = [
                    point_sensor(
                        obs[i : i + 192], {k: v[i : i + 192] for k, v in raw_t.items()}, *geometry
                    )
                    for i in range(0, len(obs), 192)
                ]
                obs = torch.cat([core for core, _ in parts])
                scans = torch.cat([scan for _, scan in parts])
            else:
                features = [
                    simulated_edge_features(p, q, *geometry, version=int(edge))
                    for p, q in zip(raw_t["root_pos"].split(192), raw_t["root_quat"].split(192))
                ]
                obs = torch.cat((obs, torch.cat(features)), 1)
        if "student_contact_point" in pack:
            points = pack["student_contact_point"]
            point_method = "measured"
        else:
            points = _project_support(
                raw["site_pos"],
                pack["terrain_box_centers_w"],
                pack["terrain_box_half_sizes"],
                pack["terrain_box_quat_wxyz"],
            )
            point_method = "landmark projection"
        ids = pack["episode_id"]
        steps = pack["step"]
        dt = float(pack["step_dt"])
        source = str(pack["source_kind"].item())
        observed = (
            pack["contact_observed"].astype(bool)
            if "contact_observed" in pack
            else np.ones(len(ids), bool)
        )
        clip_records = (
            json.loads(str(pack["provenance"].item())).get("clips", [])
            if source == "gait_anchor_far"
            else []
        )
        alternate = {int(clip["id"]) for clip in clip_records if clip.get("switches", 0) > 0}
        batches = []
        for eps in np.unique(ids):
            episode_source = (
                "isaac_measured_ground_alternation" if int(eps) in alternate else source
            )
            idx = np.flatnonzero(ids == eps)
            idx = idx[np.argsort(steps[idx])]
            if len(idx) < 2 or not (np.diff(steps[idx]) == 1).all():
                raise ValueError(f"{path}: episode {eps} has repeated or missing steps")
            observed_rows = np.flatnonzero(observed[idx])
            if len(observed_rows) and not (np.diff(observed_rows) == 1).all():
                raise ValueError(f"{path}: interior gap in measured contact samples")
            measured = idx[observed_rows]
            if point:
                from .contact_point_actor import (
                    episode_observation,
                    sustained_contacts,
                    yaw_local,
                    reachable_labels,
                )

                labeler, rotate = sustained_contacts, yaw_local
                episode_obs = episode_observation(obs[idx], scans[idx])
            else:
                labeler, rotate = upcoming_contacts, _local
                episode_obs = obs[idx]
            events_mask, pos, events = labeler(raw["contact_force"][measured], points[measured], dt)
            mask = torch.full((len(idx), 8), -1.0)
            local = torch.zeros((len(idx), 8, 3))
            transformed = rotate(
                raw_t["root_quat"][measured], tensor(pos) - raw_t["root_pos"][measured, None]
            )
            local[observed_rows] = transformed * tensor(events_mask)[:, :, None]
            mask[observed_rows] = plan_label_mask(
                tensor(events_mask),
                dt,
                ("dagger" in source.lower() or episode_source == "gait_anchor_far"),
            )
            if point:
                mask = reachable_labels(scans[idx], local, mask)
            # Group exact or numerically identical replica rollouts on the same side of the split.
            series = np.c_[poses[idx], action[idx].numpy()]
            fingerprint = hashlib.sha256(np.round(series, 3).astype("<f4").tobytes()).hexdigest()
            batches.append(
                (
                    episode_source,
                    fingerprint,
                    (episode_obs, action[idx], mask, local),
                    {
                        "source": episode_source,
                        "episode": int(eps),
                        "frames": len(idx),
                        "dt": dt,
                        "points": point_method,
                        "events": [(int(steps[measured[i]]), name) for i, name in events],
                        "reset_at_start": bool(steps[idx[0]] == 0 and not observed[idx[0]]),
                    },
                )
            )
        return names, nominal, batches


def load_examples(paths, seed=42, edge=False, point=False):
    """Return training/held-out episode batches; singleton-source episodes stay in training."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    grouped = {}
    metadata = []
    contract = None
    for path in paths:
        names, nominal, items = _episode_batches(path, edge=edge, point=point)
        if contract is None:
            contract = (names, nominal)
        if names != contract[0] or not np.allclose(nominal, contract[1], atol=1e-5):
            raise ValueError(f"{path}: motor order or neutral pose differs between sources")
        for source, key, data, meta in items:
            grouped.setdefault(source, {}).setdefault(key, []).append(
                (data, meta["reset_at_start"])
            )
            metadata.append(meta)
    if not grouped:
        raise ValueError("No expert episodes available for cloning")
    train = []
    val = []
    weights = []
    split = {}
    rng = np.random.default_rng(seed)
    for source, groups in grouped.items():
        keys = list(groups)
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * 0.2))) if len(keys) > 1 else 0
        split[source] = {"distinct_episodes": len(keys), "held_out": count}
        source_weights = []
        for j, key in enumerate(keys):
            # Identical clones convey no independent supervision. Reset-state motor targets
            # are measured even before force sensors have produced their first measurement.
            data, explicit_reset = groups[key][0]
            if j < count:
                val.append(data)
                continue
            train.append(data)
            weight = torch.ones(len(data[0]))
            if explicit_reset:
                weight[0] = 16.0
                weight[1:3] = 3.0
            source_weights.append(weight)
        normalization = sum(w.sum() for w in source_weights)
        weights.extend(
            [
                w / normalization * (0.35 if source == "gait_anchor_far" else 1.0)
                for w in source_weights
            ]
        )
    join = lambda b: tuple(torch.cat(cols) for cols in zip(*b)) if b else None
    return {
        "train": join(train),
        "validation": join(val),
        "weights": torch.cat(weights),
        "episodes": metadata,
        "split": split,
        "joint_names": list(contract[0]),
        "nominal": contract[1].tolist(),
    }


def contact_imitation_loss(actor, batch, gait_anchor=None, rng=None, teacher_probability=0.0):
    """Can be reused as a supervised PPO anchor on demonstrations from the *training* split."""
    obs, target, valid, locations = batch
    if hasattr(actor, "imitation_components"):
        prediction, logits, positions, point_loss = actor.imitation_components(
            obs, valid, locations, teacher_probability, rng
        )
    else:
        prediction, logits, positions = actor.action_and_plan(obs)
        point_loss = 0.0
    # Accurately imitate small motor changes as well as large transitions.
    per_joint = F.smooth_l1_loss(prediction, target, beta=0.05, reduction="none")
    if gait_anchor is None:
        drive = per_joint.mean()
    else:
        from .contact_anchor import anchor_observation, far_ground_weight, perturbed_feedback

        far = far_ground_weight(obs)
        drive = (per_joint.mean(1) * (1 - 0.98 * far)).mean()
        perturbed = perturbed_feedback(obs, rng)
        with torch.no_grad():
            desired = gait_anchor(anchor_observation(actor, gait_anchor, perturbed))
        motor_error = (actor(perturbed) - desired).square().mean(1)
        drive += 35 * (motor_error * far).mean()
    absent = (valid == 0).float()
    valid = valid.clamp_min(0)
    prob_loss = F.binary_cross_entropy_with_logits(logits, valid, reduction="none")
    positive = (prob_loss * valid).sum() / valid.sum().clamp_min(1)
    negative = (prob_loss * absent).sum() / absent.sum().clamp_min(1)
    place = (
        F.smooth_l1_loss(positions, locations, beta=0.04, reduction="none") * valid[:, :, None]
    ).sum() / (3 * valid.sum()).clamp_min(1)
    return drive + 0.035 * (positive + negative) + 0.12 * place + point_loss


@torch.no_grad()
def evaluate_actor(actor, batch, device=None):
    if batch is None:
        return None
    device = device or next(actor.parameters()).device
    total = labels = hits = predicted_labels = 0
    motor_error = dist = persistence_error = 0.0
    for i in range(0, len(batch[0]), 512):
        obs, y, mask, pos = (a[i : i + 512].to(device) for a in batch)
        action, logits, xyz = actor.action_and_plan(obs)
        total += y.numel()
        motor_error += (action - y).abs().sum().item()
        predicted_labels += ((logits > 0) & (mask >= 0)).sum().item()
        mask = mask.clamp_min(0)
        count = mask.sum().item()
        labels += count
        persistence_error += (obs[:, 58:87] - y).abs().sum().item()
        hits += ((logits > 0) * mask).sum().item()
        dist += ((xyz - pos).norm(dim=2) * mask).sum().item()
    return {
        "action_mae_rad": motor_error / total,
        "valid_contact_labels": int(labels),
        "hold_previous_target_mae_rad": persistence_error / total,
        "contact_precision": hits / predicted_labels if predicted_labels else None,
        "contact_recall": hits / labels if labels else None,
        "position_mae_m": dist / labels if labels else None,
    }


@torch.no_grad()
def anchor_deviation(actor, anchor, batch):
    from .contact_anchor import anchor_observation, far_ground_weight

    if batch is None or anchor is None:
        return None
    denom = total = 0.0
    device = next(actor.parameters()).device
    for obs in batch[0].split(512):
        obs = obs.to(device)
        far = far_ground_weight(obs)
        total += (
            ((actor(obs) - anchor(anchor_observation(actor, anchor, obs))).abs().mean(1) * far)
            .sum()
            .item()
        )
        denom += far.sum().item()
    return {
        "far_effective_frames": denom,
        "far_drive_difference_rad": total / denom if denom else None,
    }


def fit_actor(
    actor,
    paths,
    *,
    epochs=120,
    batch_size=256,
    lr=0.001,
    seed=42,
    fit_normalization=True,
    gait_anchor=None,
):
    """Fit the provided actor in place. Paths may also be the result of load_examples()."""
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Expected positive training duration, minibatch size and learning rate")
    examples = paths if isinstance(paths, dict) else load_examples(paths, seed)
    train, val = examples["train"], examples["validation"]
    weights = examples["weights"]
    device = next(actor.parameters()).device
    if fit_normalization:
        with torch.no_grad():
            w = weights / weights.sum()
            mean = (train[0] * w[:, None]).sum(0)
            actor.obs_mean.copy_(mean)
            actor.obs_scale.copy_(
                ((train[0] - mean).square() * w[:, None]).sum(0).sqrt().clamp_min(0.08)
            )
    groups = (
        actor.supervised_parameter_groups(lr)
        if hasattr(actor, "supervised_parameter_groups")
        else actor.parameters()
    )
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-5)
    rng = torch.Generator().manual_seed(seed)
    best = float("inf")
    best_state = None
    for epoch in range(epochs):
        teacher_probability = (
            min(1.0, max(0.0, (0.75 - epoch / max(epochs - 1, 1)) / 0.50))
            if hasattr(actor, "point_spec")
            else 0.0
        )
        actor.train()
        indices = torch.multinomial(weights, len(weights), replacement=True, generator=rng)
        for ix in indices.split(batch_size):
            batch = [value[ix].to(device) for value in train]
            obs = batch[0].clone()
            # Motor/sensor noise makes the clone recover small tracking deviations without
            # leaking a clean demonstration command into the previous-target feature.
            obs[:, :29] += torch.randn((len(ix), 29), generator=rng).to(device) * 0.007
            obs[:, 58:87] += torch.randn((len(ix), 29), generator=rng).to(device) * 0.013
            batch[0] = obs
            loss = contact_imitation_loss(actor, batch, gait_anchor, rng, teacher_probability)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
            optimizer.step()
        if val:
            actor.eval()
            m = evaluate_actor(actor, val)
            score = m["action_mae_rad"] + 0.05 * (m["position_mae_m"] or 0)
            anchor_error = anchor_deviation(actor, gait_anchor, val)
            if anchor_error:
                score += 2 * (anchor_error["far_drive_difference_rad"] or 0)
            if score < best and teacher_probability == 0:
                best = score
                best_state = copy.deepcopy(actor.state_dict())
    if best_state is not None:
        actor.load_state_dict(best_state)
    actor.eval()
    return {
        "epochs": epochs,
        "contact_target": "per body next confirmed loading event within 1.5 s; truncated unknown negatives and corrective rollout plans excluded",
        "train": evaluate_actor(actor, train),
        "validation": evaluate_actor(actor, val),
        "anchor_validation": anchor_deviation(actor, gait_anchor, val),
        "split": examples["split"],
        "episodes": examples["episodes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path, nargs="+")
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination for actor.state_dict()"
    )
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--gait-drive-model",
        type=Path,
        help="Initialize the shared motor layers from the bundled gait and the position-drive model XML",
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Fine tune an existing student while preserving its observation scaling",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--gait-anchor",
        type=Path,
        help="Distill stable gait feedback only during offline learning on open ground",
    )
    parser.add_argument(
        "--edge",
        action="store_true",
        help="Append 5 cm simulated height-scan geometry with explicit visibility",
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Learn per-body selection from visible height points with five-step history",
    )
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    data = load_examples(
        args.examples, args.seed, edge=2 if args.edge or args.point else 0, point=args.point
    )
    if args.gait_drive_model and args.initial_weights:
        parser.error("Select one source of initial weights")
    weights = (
        torch.load(args.initial_weights, weights_only=True, map_location="cpu")
        if args.initial_weights
        else None
    )
    if args.point:
        from .contact_point_actor import PointContactActor

        actor = PointContactActor()
    elif args.gait_drive_model or (weights is not None and "backbone.0.weight" in weights):
        from .contact_gait_actor import GaitContactActor

        if args.edge:
            from .contact_edge import EdgeGaitContactActor

            actor = EdgeGaitContactActor()
        else:
            actor = GaitContactActor()
        if args.gait_drive_model:
            actor.initialize_vendor(data["joint_names"], data["nominal"], args.gait_drive_model)
    else:
        if args.edge:
            from .contact_edge import EdgeContactActor

            actor = EdgeContactActor()
        else:
            actor = ContactActor()
    if weights is not None:
        if (args.point and "point_spec" not in weights) or (
            args.edge and "edge_spec" not in weights
        ):
            from .contact_edge import extend_weights

            extend_weights(actor, weights)
        else:
            actor.load_state_dict(weights)
    gait_anchor = None
    if args.gait_anchor:
        from .contact_gait_actor import GaitContactActor

        gait_anchor = GaitContactActor()
        gait_anchor.load_state_dict(
            torch.load(args.gait_anchor, weights_only=True, map_location="cpu")
        )
        gait_anchor.requires_grad_(False).eval()
    if args.edge and weights is not None and "edge_spec" not in weights:
        edge_columns = data["train"][0][:, CONTACT_OBS_DIM:]
        actor.obs_mean[CONTACT_OBS_DIM:] = edge_columns.mean(0)
        actor.obs_scale[CONTACT_OBS_DIM:] = edge_columns.std(0).clamp_min(0.08)
    if args.point and weights is not None and "point_spec" not in weights:
        observed = data["train"][0]
        # Only new temporal channels need calibration when transferring the existing actor.
        c = observed[:, 262:646]
        actor.obs_mean[262:646] = c.mean(0)
        actor.obs_scale[262:646] = c.std(0).clamp_min(0.08)
    report = fit_actor(
        actor,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.learning_rate,
        fit_normalization=weights is None,
        gait_anchor=gait_anchor,
    )
    report["initial_weights"] = str(args.initial_weights) if args.initial_weights else None
    if args.point:
        from .contact_point_actor import evaluate_points

        report["contact_target"] = (
            "per body, 1.5s, ten consecutive loaded 50 Hz steps, filter scan footprint, gravity-aligned base yaw"
        )
        report["teacher_plan_schedule"] = (
            "first quarter measured plan, next half stochastic annealing to zero, final quarter and all evaluation predicted"
        )
        report["point_validation"] = evaluate_points(actor, data["validation"])
    report["gait_anchor"] = str(args.gait_anchor) if args.gait_anchor else None
    report["actor"] = type(actor).__name__
    report["drive_model"] = str(args.gait_drive_model) if args.gait_drive_model else None
    report["samples"] = [str(p.resolve()) for p in args.examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(actor.state_dict(), temporary)
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("episodes", "samples")}, indent=2)
    )


if __name__ == "__main__":
    main()
