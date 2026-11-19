"""Flat-ground walking commands, physical step detection and reward rates (SI units)."""

import math
import torch

WALK_STEPS = 300  # 6 s at the tracker's 50 Hz
STOP_STEPS = 125  # 2.5 s, including braking
HALF = WALK_STEPS // 2
FEATURES = 4  # speed, gait clock sin/cos, residual enable


def walk_command(tick, speeds, first_side, dt):
    active = tick >= 0
    moving = active & (tick < WALK_STEPS)
    speed = torch.where(tick < HALF, speeds[:, 0], speeds[:, 1]) * moving
    angle = (2 * math.pi / 1.10) * tick * dt + first_side * math.pi
    sine, cosine = angle.sin() * moving, angle.cos() * moving
    return speed, torch.stack((speed * 4, sine, cosine, active.float()), dim=1)


class Footsteps:
    """Credit a foot once after supported lift, forward swing and load acceptance."""

    def __init__(self, count, device):
        self.flight = torch.zeros((count, 2), dtype=torch.long, device=device)
        self.lifted = torch.zeros_like(self.flight, dtype=torch.bool)
        self.takeoff = torch.zeros_like(self.flight, dtype=torch.float)
        self.last_side = torch.full_like(self.flight[:, 0], -1)
        self.counts = torch.zeros_like(self.flight)

    def reset(self, ids):
        self.flight[ids] = 0
        self.lifted[ids] = False
        self.last_side[ids] = -1
        self.counts[ids] = 0

    def update(self, moving, load, height, forward_pos):
        loaded = load > 20
        swing = (load < 5) & loaded.flip([1]) & moving[:, None]
        self.takeoff = torch.where(swing & (self.flight == 0), forward_pos, self.takeoff)
        self.flight = torch.where(swing, self.flight + 1, 0)
        self.lifted |= (self.flight >= 3) & (height > 0.025)
        landed = self.lifted & loaded & (forward_pos - self.takeoff > 0.045)
        side = landed.long().argmax(1)
        alternating = moving & (landed.sum(1) == 1) & (side != self.last_side)
        self.counts += (landed & alternating[:, None]).long()
        self.last_side = torch.where(alternating, side, self.last_side)
        # Losing both supports invalidates the pending swing (hopping).
        self.lifted &= ~loaded & loaded.flip([1]) & moving[:, None]
        return alternating


def reward_rates(
    speed,
    forward,
    lateral,
    tilt,
    angular_xy,
    yaw_error,
    wrist_error,
    foot_height,
    load,
    slip,
    phase_sine,
    smooth,
    effort,
):
    moving = speed > 0
    # Same full credit across a small speed-error band; separate stop shaping.
    error = ((forward - speed).abs() - 0.035).clamp_min(0)
    tracking = torch.exp(-(error / 0.11).square())
    lobe = torch.stack((phase_sine, -phase_sine), dim=1).clamp_min(0)
    swing_shape = 1 - ((foot_height - 0.060 * lobe).abs() / 0.060).clamp(max=1)
    opposite_support = (load.flip([1]) / 35).clamp(0, 1)
    # During double support near phase boundaries this cue is deliberately weak.
    swing = (lobe * swing_shape * opposite_support).sum(1)
    return {
        "forward_velocity": 3.0 * tracking,
        "lateral_velocity": 0.7 * torch.exp(-(lateral / 0.14).square()),
        "heading": 0.45 * torch.exp(-(yaw_error / 0.26).square()),
        "upright": 0.65 * torch.exp(-(tilt / 0.35).square()),
        "wrist": 0.55 * torch.exp(-(wrist_error / 0.12).square()),
        "swing_and_support": 1.0 * swing * moving,
        "stopped_double_support": 0.65 * ~moving * (load > 20).all(1) * tracking,
        "stance_slip": -1.1 * slip.clamp(max=1),
        "flight": -0.8 * (load < 5).all(1),
        "trunk_angular_velocity": -0.12 * angular_xy.square().clamp(max=4),
        "action_change": -0.06 * smooth,
        "effort_ratio": -0.08 * effort,
    }
