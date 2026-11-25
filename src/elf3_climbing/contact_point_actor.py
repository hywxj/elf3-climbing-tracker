"""Terrain-point planner, five-step proprioception and a single whole-body motor policy."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contact_edge import (
    EDGE_OBS_DIM,
    NX,
    NY,
    EdgeGaitContactActor,
    edge_from_scan,
    sample_height_scan,
    scan_axes,
)
from .contact_learning import CONTACT_NAMES


PROPRIO_DIM = 96
HISTORY_DIM = 4 * PROPRIO_DIM
N_POINTS = 20 * 19
POINT_OBS_DIM = EDGE_OBS_DIM + HISTORY_DIM + N_POINTS
MISSING_HEIGHT = -5.0


def yaw_local(quat, relative):
    w, x, y, z = quat.T
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))[:, None]
    dx, dy, dz = relative.unbind(-1)
    return torch.stack((yaw.cos() * dx + yaw.sin() * dy, -yaw.sin() * dx + yaw.cos() * dy, dz), -1)


def point_sensor(base, raw, centers, halves, quats, *, lidar=None):
    """Build the proprioceptive and terrain input. A measured map replaces all terrain channels.

    lidar, when supplied, is a batch of (39x37 heights, 39x37 valid flags, 126 base features)
    assembled from LidarElevation. In that case the simulation geometry can be omitted.
    """
    if lidar is None:
        h, valid = sample_height_scan(raw["root_pos"], raw["root_quat"], centers, halves, quats)
        state = base[:, :222]
    else:
        h, valid, terrain = (torch.as_tensor(a, device=base.device) for a in lidar)
        if (
            h.shape != (len(base), NX, NY)
            or valid.shape != h.shape
            or terrain.shape != (len(base), 126)
        ):
            raise ValueError("Expected one LiDAR height map and 126 base features per robot")
        h = h.to(base.dtype)
        valid = valid.bool() & torch.isfinite(h)
        state = torch.cat((base[:, :PROPRIO_DIM], terrain.to(base.dtype)), 1)
    landmarks = yaw_local(raw["root_quat"], raw["site_pos"] - raw["root_pos"][:, None])
    core = torch.cat(
        (
            state,
            landmarks.flatten(1),
            base[:, 246:254],
            edge_from_scan(h, valid, version=2),
        ),
        1,
    )
    return core, torch.where(valid, h, MISSING_HEIGHT)[:, ::2, ::2].flatten(1)


def episode_observation(core, heights):
    """The previous four measured states, zero-padded at this episode's beginning."""
    n = len(core)
    past = core.new_zeros((n, 4, PROPRIO_DIM))
    for lag in range(1, 5):
        past[lag:, 4 - lag] = core[:-lag, :PROPRIO_DIM]
    return torch.cat((core, past.flatten(1), heights), 1)


class PointHistory:
    def __init__(self, num_envs, device):
        self.states = torch.zeros(num_envs, 5, PROPRIO_DIM, device=device)
        self.tick = None

    def reset(self, ids):
        self.states[ids] = 0

    def encode(self, base_obs, raw, centers, halves, quats, tick, *, lidar=None):
        core, scan = point_sensor(base_obs, raw, centers, halves, quats, lidar=lidar)
        if self.tick != tick:
            self.states = torch.roll(self.states, -1, dims=1)
            self.tick = tick
        self.states[:, -1] = core[:, :PROPRIO_DIM]
        return torch.cat((core, self.states[:, :4].flatten(1), scan), 1)


def sustained_contacts(forces, points, dt):
    """Next onset per body, with ten contiguous measured control steps and a loaded knee."""
    n = len(forces)
    mask = np.zeros((n, 8), bool)
    locations = np.zeros((n, 8, 3), np.float32)
    events = []
    thresholds = (15, 15, 3, 3, 3, 3, 55.73, 55.73)
    for body, on in enumerate(thresholds):
        loaded = forces[:, body] >= on
        changes = np.diff(np.r_[False, loaded, False].astype("i1"))
        previous = 0
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if start == 0 or end - start < 10:
                continue
            onset = points[start, body]
            if not np.isfinite(onset).all():
                continue
            lo = max(previous, start - round(1.5 / dt))
            mask[lo:start, body] = True
            locations[lo:start, body] = onset
            previous = start
            events.append((int(start), CONTACT_NAMES[body]))
    return mask, locations, sorted(events)


def grid_xy(reference):
    x, y = scan_axes(reference)
    return torch.cartesian_prod(x[::2], y[::2])


def contact_candidates(scan, xy):
    valid = (scan > MISSING_HEIGHT + 0.01) & torch.isfinite(scan)
    z = torch.where(valid, scan, 0.0)
    return torch.cat((xy.expand(len(scan), -1, -1), z[:, :, None]), 2), valid


def reachable_labels(scan, targets, labels):
    """Suppress targets outside the sensor and surfaces absent from the observed grid."""
    xy = grid_xy(scan)
    for i in range(0, len(scan), 64):
        part = slice(i, i + 64)
        points, valid = contact_candidates(scan[part], xy)
        dist = torch.cdist(targets[part], points).masked_fill(~valid[:, None], torch.inf).amin(2)
        inside = (
            (targets[part][:, :, 0] >= xy[:, 0].min())
            & (targets[part][:, :, 0] <= xy[:, 0].max())
            & (targets[part][:, :, 1] >= xy[:, 1].min())
            & (targets[part][:, :, 1] <= xy[:, 1].max())
        )
        labels[part].masked_fill_((labels[part] > 0) & (~inside | (dist > 0.13)), -1)
    return labels


class PointContactActor(EdgeGaitContactActor):
    plan_yaw_frame = True

    def __init__(self, body_frame_input=False):
        super().__init__()
        self.body_frame_input = body_frame_input
        self.register_buffer(
            "point_spec", torch.tensor(2 if body_frame_input else 1, dtype=torch.int64)
        )
        self.register_load_state_dict_post_hook(self._read_input_frame)
        self.register_buffer("point_xy", grid_xy(torch.zeros(1)))
        self.obs_mean = torch.zeros(POINT_OBS_DIM)
        self.obs_scale = torch.ones(POINT_OBS_DIM)
        self.history_embedding = nn.Linear(HISTORY_DIM, 128, bias=False)
        self.query = nn.Linear(128, 8 * 8)
        nn.init.zeros_(self.history_embedding.weight)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def _read_input_frame(self, module, incompatible):
        self.body_frame_input = self.point_spec.item() == 2

    def point_state(self, obs):
        core = obs[:, : EDGE_OBS_DIM + HISTORY_DIM]
        if self.body_frame_input:
            from .contact_anchor import full_body_landmarks

            core = full_body_landmarks(core)
        normalized = (
            (core - self.obs_mean[: EDGE_OBS_DIM + HISTORY_DIM])
            / self.obs_scale[: EDGE_OBS_DIM + HISTORY_DIM]
        ).clamp(-8, 8)
        return self.encoder(normalized[:, :EDGE_OBS_DIM]) + self.history_embedding(
            normalized[:, EDGE_OBS_DIM:]
        )

    def controls(self, obs, labels=None, targets=None, teacher_probability=0.0, rng=None):
        state = self.point_state(obs)
        proposals = self.planner(state).reshape(-1, 8, 4)
        logits, seed = proposals[:, :, 0], proposals[:, :, 1:]
        points, seen = contact_candidates(obs[:, -N_POINTS:], self.point_xy)
        x, y, z = points.unbind(2)
        keys = torch.stack(
            (x, y, z, x.square(), y.square(), z.square(), x * y, torch.ones_like(x)), -1
        )
        score = torch.bmm(self.query(state).reshape(-1, 8, 8), keys.transpose(1, 2))
        # A learned per-body query attends to observed cells using both their content and
        # squared distance to a continuous query center. The center never goes to the motors.
        score = score + (
            2 * torch.bmm(seed, points.transpose(1, 2)) - points.square().sum(2)[:, None]
        ) / (2 * 0.085**2)
        score = score.masked_fill(~seen[:, None], -1e4)
        distribution = score.softmax(2)
        weighted = torch.bmm(distribution, points)
        index = score.argmax(2)
        selected = points.gather(1, index[:, :, None].expand(-1, -1, 3))
        selected = weighted + (selected - weighted).detach() if self.training else selected
        logits = torch.where(seen.any(1)[:, None], logits, -100.0)
        prob = logits.sigmoid()
        hard = (prob >= 0.5).float()
        valid = prob + (hard - prob).detach() if self.training else hard
        motor_valid = valid
        motor_pos = selected * valid[:, :, None]
        if labels is not None and teacher_probability > 0:
            chosen = torch.rand((len(obs), 1), generator=rng).to(obs.device) < teacher_probability
            use = (labels >= 0) & chosen
            motor_valid = torch.where(use, labels.clamp_min(0), motor_valid)
            motor_pos = torch.where(
                use[:, :, None], targets * labels.clamp_min(0)[:, :, None], motor_pos
            )
        context = torch.cat((state, motor_valid, (motor_pos).flatten(1)), 1)
        return self.motor_target(obs, context), logits, selected, score, points

    def action_and_plan(self, obs):
        a, logits, points, _, _ = self.controls(obs)
        return a, logits, torch.where((logits >= 0)[:, :, None], points, 0.0)

    def imitation_components(self, obs, labels, targets, probability, rng):
        action, logits, pts, scores, candidates = self.controls(
            obs, labels, targets, probability, rng
        )
        # Categorical supervision targets the nearest *measured* cell in 3D. Masking
        # unknown cells also covers future real scans and missing depth returns.
        with torch.no_grad():
            dist = torch.cdist(targets, candidates)
            dist.masked_fill_(obs[:, None, -N_POINTS:] < MISSING_HEIGHT + 0.01, torch.inf)
            nearest = dist.argmin(2)
        cls = F.cross_entropy(scores.flatten(0, 1), nearest.flatten(), reduction="none").view_as(
            labels
        )
        pos = labels > 0
        ce = (cls * pos).sum() / pos.sum().clamp_min(1)
        return action, logits, pts, 0.035 * ce


@torch.no_grad()
def evaluate_points(actor, validation):
    """Measure where the planner places each real future contact on the held-out rollouts."""
    if validation is None:
        return None
    observed, _, occupancy, target = validation
    device = actor.point_xy.device
    counts = torch.zeros(8, 5, dtype=torch.float64, device=device)
    for obs, mask, true in zip(observed.split(128), occupancy.split(128), target.split(128)):
        obs, true, positive = obs.to(device), true.to(device), mask.to(device) > 0
        _, logits, selected, scores, candidates = actor.controls(obs)
        nearest_distances = torch.cdist(true, candidates)
        scan = obs[:, None, -N_POINTS:]
        hidden = ~torch.isfinite(scan) | (scan < MISSING_HEIGHT + 0.01)
        nearest_distances.masked_fill_(hidden, torch.inf)
        oracle_error, nearest = nearest_distances.min(2)
        present = logits >= 0
        error = (selected - true).norm(dim=2)
        counts[:, 0] += positive.sum(0)
        counts[:, 1] += (positive & present).sum(0)
        counts[:, 2] += (positive & present & (scores.argmax(2) == nearest)).sum(0)
        counts[:, 3] += torch.where(positive & present, error, 0.0).sum(0)
        counts[:, 4] += torch.where(positive, oracle_error, 0.0).sum(0)

    def metrics(row):
        n, called, exact, distance, oracle = row.tolist()
        return {
            "observed_future_contacts": int(n),
            "contact_recall": called / n if n else None,
            "predicted_nearest_visible_cell_fraction": exact / n if n else None,
            "position_error_when_predicted_m": distance / called if called else None,
            "nearest_visible_cell_oracle_error_m": oracle / n if n else None,
        }

    return {
        "overall": metrics(counts.sum(0)),
        "by_body": {name: metrics(row) for name, row in zip(CONTACT_NAMES, counts)},
    }
