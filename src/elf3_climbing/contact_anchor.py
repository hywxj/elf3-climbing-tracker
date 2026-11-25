"""Offline stability distillation. These geometric weights are never used by the deployed actor."""

import torch


def far_ground_weight(obs):
    """Confidence that the upright robot still has open walking space in front of it."""
    height = obs[:, 96:159].reshape(-1, 9, 7)
    valid = obs[:, 159:222].reshape(-1, 9, 7) > 0.5
    floor = height[:, :3, 2:5].flatten(1).median(1).values
    forward = torch.linspace(-0.8, 1.2, 9, device=obs.device)
    high = ((height[:, :, 2:5] - floor[:, None, None] > 0.30) & valid[:, :, 2:5]).any(2)
    distance = torch.where(high, forward[None, :], 2.0).amin(1)
    # A sample already behind the machine or under its body releases the stability anchor.
    room = ((distance - 0.20) / 0.50).clamp(0, 1)
    if obs.shape[1] >= 262:
        # The denser scan resolves whether the robot is still in the clear.
        room = torch.where(obs[:, 259] > 0.5, ((obs[:, 254] - 0.22) / 0.45).clamp(0, 1), room)
    upright = ((-obs[:, 95] - 0.82) / 0.12).clamp(0, 1)
    tall = ((-floor - 0.65) / 0.25).clamp(0, 1)
    unbraced = obs[:, 248:254].amax(1) < 0.15
    seen = valid[:, :3, 2:5].all(2).any(1)
    return room * upright * tall * unbraced * seen


def anchor_observation(actor, anchor, obs):
    """Map current observations into the walking anchor's original full-body frame."""
    anchor_obs = obs[:, : len(anchor.obs_mean)]
    if not getattr(actor, "plan_yaw_frame", False) or getattr(anchor, "plan_yaw_frame", False):
        return anchor_obs
    return full_body_landmarks(anchor_obs)


def full_body_landmarks(obs):
    """Convert yaw-aligned measured landmarks to the original motor encoder's body frame."""
    gravity = torch.nn.functional.normalize(obs[:, 93:96], dim=1)
    gx, gy, gz = gravity.T
    cos_pitch = (gy.square() + gz.square()).sqrt().clamp_min(1e-6)
    sin_roll, cos_roll = -gy / cos_pitch, -gz / cos_pitch
    x, y, z = obs[:, 222:246].reshape(-1, 8, 3).unbind(2)
    tx = cos_pitch[:, None] * x - gx[:, None] * z
    tz = gx[:, None] * x + cos_pitch[:, None] * z
    body = torch.stack(
        (
            tx,
            cos_roll[:, None] * y + sin_roll[:, None] * tz,
            -sin_roll[:, None] * y + cos_roll[:, None] * tz,
        ),
        2,
    )
    anchor_obs = obs.clone()
    anchor_obs[:, 222:246] = body.flatten(1)
    return anchor_obs


def perturbed_feedback(obs, rng):
    """Approximate small tracking disturbances around actual roll-in states for distillation."""
    x = obs.clone()
    for lo, hi, noise in (
        (0, 29, 0.025),
        (29, 58, 0.025),
        (58, 87, 0.02),
        (87, 90, 0.035),
        (90, 93, 0.025),
        (93, 95, 0.015),
    ):
        x[:, lo:hi] += torch.randn((len(x), hi - lo), generator=rng).to(x.device) * noise
    return x
