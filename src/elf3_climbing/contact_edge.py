"""An egocentric elevation scan and an edge estimate. The simulator samples horizontal boxes;
a real sensor can supply its own gridded heights and missing-cell mask to edge_from_scan().
"""

import torch
from torch import nn

from .contact_gait_actor import GaitContactActor
from .contact_learning import CONTACT_OBS_DIM, ContactActor


EDGE_FEATURES = (
    "normal_distance_m",
    "normal_x",
    "normal_y",
    "rise_m",
    "visible_span_m",
    "edge_visible",
    "center_tangent_m",
    "center_visible",
)
EDGE_OBS_DIM = CONTACT_OBS_DIM + len(EDGE_FEATURES)
NX, NY = 39, 37


def scan_axes(reference):
    return (
        torch.linspace(-0.50, 1.40, NX, device=reference.device, dtype=reference.dtype),
        torch.linspace(-0.90, 0.90, NY, device=reference.device, dtype=reference.dtype),
    )


def sample_height_scan(root_pos, root_quat, centers, halves, quats):
    """Known simulation geometry sampled on a 5 cm yaw-aligned grid. No ray occlusion model."""
    x, y = scan_axes(root_pos)
    w, qx, qy, qz = root_quat.T
    yaw = torch.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))[:, None, None]
    wx = root_pos[:, 0, None, None] + yaw.cos() * x[None, :, None] - yaw.sin() * y[None, None, :]
    wy = root_pos[:, 1, None, None] + yaw.sin() * x[None, :, None] + yaw.cos() * y[None, None, :]
    heights = torch.zeros_like(wx)
    for center, half, quat in zip(centers, halves, quats):
        cw, cx, cy, cz = quat
        heading = torch.atan2(2 * (cw * cz + cx * cy), 1 - 2 * (cy * cy + cz * cz))
        dx, dy = wx - center[0], wy - center[1]
        inside = ((heading.cos() * dx + heading.sin() * dy).abs() <= half[0]) & (
            (-heading.sin() * dx + heading.cos() * dy).abs() <= half[1]
        )
        heights = torch.where(inside, torch.maximum(heights, center[2] + half[2]), heights)
    return heights - root_pos[:, 2, None, None], torch.ones_like(heights, dtype=torch.bool)


def _fit_line(x, y, valid):
    count = valid.sum(1).clamp_min(1)
    cx = (x * valid).sum(1) / count
    cy = (y * valid).sum(1) / count
    slope = ((x - cx[:, None]) * (y - cy[:, None]) * valid).sum(1) / (
        (y - cy[:, None]).square() * valid
    ).sum(1).clamp_min(1e-6)
    offset = cx - slope * cy
    return slope, offset


def _directional_feature(heights, valid, x, y, legacy=False):
    nx, ny = heights.shape[1:]
    valid = valid.bool() & torch.isfinite(heights)
    height = torch.where(valid, heights, 0.0)
    # Require two adjacent samples on the higher face, guarding against isolated noise.
    rise = height[:, 1:-1] - height[:, :-2]
    transition = (
        (rise > 0.20)
        & ((height[:, 2:] - height[:, 1:-1]).abs() < 0.10)
        & valid[:, :-2]
        & valid[:, 1:-1]
        & valid[:, 2:]
    )
    hit = transition.any(1)
    idx = transition.int().argmax(1)
    xm = (x[idx] + x[idx + 1]) * 0.5
    ym = y.expand_as(xm)
    slope, offset = _fit_line(xm, ym, hit)
    close = hit & ((xm - offset[:, None] - slope[:, None] * ym).abs() < 0.09)
    slope, offset = _fit_line(xm, ym, close)
    denom = (1 + slope.square()).sqrt()
    count = close.sum(1)
    lo = torch.where(close, ym, 2.0).amin(1)
    hi = torch.where(close, ym, -2.0).amax(1)
    span = (hi - lo + 0.05) * denom
    residual = (xm - offset[:, None] - slope[:, None] * ym).square()
    visible = (
        (count >= 5)
        & (span >= 0.23)
        & (slope.abs() < 1.6)
        & ((residual * close).sum(1) / count.clamp_min(1) < 0.045**2)
    )
    # Reject disparate edges whose supporting points do not make a mostly contiguous face.
    coverage = 0.70 if legacy else 0.35
    visible &= count * 0.05 >= coverage * (hi - lo + 0.05)
    jump = rise.gather(1, idx[:, None]).squeeze(1)
    rise_m = (jump * close).sum(1) / count.clamp_min(1)
    # Both neighboring columns must show observed continuous free surface across the projected
    # edge; a cropped/occluded boundary cannot localize a physical edge endpoint.
    iy = torch.arange(ny, device=height.device).expand_as(idx)
    ilo = torch.where(close, iy, ny).amin(1)
    ihi = torch.where(close, iy, -1).amax(1)

    def end_is_free(j):
        col = j.clamp(0, ny - 1)
        px = offset + slope * y[col]
        k = ((px - x[0]) / 0.05).floor().long().clamp(0, nx - 2)
        lane = torch.arange(len(col), device=height.device)
        if not legacy:
            # Verify an observed low surface beyond the endpoint. Absence of a jump in an
            # unmeasured column or a continuous high plateau is insufficient evidence.
            base = (height.gather(1, idx[:, None]).squeeze(1) * close).sum(1) / count.clamp_min(1)
            rear = (k - 1).clamp(0, nx - 1)
            front = (k + 2).clamp(0, nx - 1)
            no_high = (
                valid[lane, rear, col]
                & valid[lane, front, col]
                & ((height[lane, front, col] - base).abs() < 0.12)
            )
        else:
            no_high = torch.ones_like(col, dtype=torch.bool)
        return (
            no_high
            & (j >= 0)
            & (j < ny)
            & (px > x[1])
            & (px < x[-2])
            & valid[lane, k, col]
            & valid[lane, k + 1, col]
            & ((height[lane, k + 1, col] - height[lane, k, col]).abs() < 0.10)
            & ~hit[lane, col]
        )

    center_ok = visible & end_is_free(ilo - 1) & end_is_free(ihi + 1)
    center_y = (lo + hi) * 0.5
    center_x = offset + slope * center_y
    tangent = (slope * center_x + center_y) / denom
    result = torch.stack(
        (
            offset / denom,
            1 / denom,
            -slope / denom,
            rise_m,
            span,
            visible.float(),
            torch.where(center_ok, tangent, 0.0),
            center_ok.float(),
        ),
        1,
    )
    tangent_lo = (slope * (offset + slope * lo) + lo) / denom
    tangent_hi = (slope * (offset + slope * hi) + hi) / denom
    out_of_segment = torch.maximum(tangent_lo.clamp_min(0), (-tangent_hi).clamp_min(0))
    distance = (offset / denom).square() + out_of_segment.square()
    score = torch.where(visible, distance, torch.inf)
    return torch.where(visible[:, None], result, 0.0), score


def edge_from_scan(heights, valid, version=2):
    if heights.shape != valid.shape or heights.shape[1:] != (NX, NY):
        raise ValueError("Expected one height and measured validity per scan cell")
    x, y = scan_axes(heights)
    if version == 1:
        return _directional_feature(heights, valid, x, y, legacy=True)[0]
    if version != 2:
        raise ValueError("Unknown elevation scan geometry version")
    alternatives = [
        (heights, valid, x, y, 1, 0),
        (
            heights.transpose(1, 2).flip((2,)),
            valid.transpose(1, 2).flip((2,)),
            y,
            -x.flip((0,)),
            0,
            1,
        ),
        (heights.flip((1, 2)), valid.flip((1, 2)), -x.flip((0,)), -y.flip((0,)), -1, 0),
        (
            heights.transpose(1, 2).flip((1,)),
            valid.transpose(1, 2).flip((1,)),
            -y.flip((0,)),
            x,
            0,
            -1,
        ),
    ]
    choices = []
    scores = []
    for h, m, gx, gy, c, si in alternatives:
        feature, score = _directional_feature(h, m, gx, gy)
        a, b = feature[:, 1].clone(), feature[:, 2].clone()
        feature[:, 1], feature[:, 2] = c * a - si * b, si * a + c * b
        choices.append(feature)
        scores.append(score)
    best = torch.stack(scores, 1).argmin(1)
    return torch.stack(choices, 1)[torch.arange(len(best), device=best.device), best]


def simulated_edge_features(root_pos, root_quat, centers, halves, quats, version=2):
    return edge_from_scan(
        *sample_height_scan(root_pos, root_quat, centers, halves, quats), version=version
    )


def _expand_encoder(actor):
    old = actor.encoder[0]
    new = nn.Linear(EDGE_OBS_DIM, old.out_features)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :CONTACT_OBS_DIM].copy_(old.weight)
        new.bias.copy_(old.bias)
    actor.encoder[0] = new
    actor.obs_mean = torch.zeros(EDGE_OBS_DIM)
    actor.obs_scale = torch.ones(EDGE_OBS_DIM)
    actor.register_buffer("edge_spec", torch.tensor(2, dtype=torch.int64))


class EdgeContactActor(ContactActor):
    def __init__(self):
        super().__init__()
        _expand_encoder(self)


class EdgeGaitContactActor(GaitContactActor):
    def __init__(self):
        super().__init__()
        _expand_encoder(self)


def extend_weights(actor, source):
    """Preserve all 254-D motor outputs before training the new localization features."""
    state = actor.state_dict()
    for k, v in source.items():
        if k in ("obs_mean", "obs_scale"):
            state[k][: len(v)].copy_(v)
        elif k == "encoder.0.weight":
            state[k][:, : v.shape[1]].copy_(v)
        else:
            state[k].copy_(v)
    actor.load_state_dict(state)
