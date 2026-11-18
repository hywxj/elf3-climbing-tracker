"""ELF3-specific gripper sequencing and CSI; independent of simulator imports."""

from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class GripperRules:
    target_angle: float = -1.309
    angle_tolerance: float = 0.07
    velocity_tolerance: float = 0.25
    settle_steps: int = 10  # 0.2 s at 50 Hz
    ready_clearance: float = 0.01
    penetration_tolerance: float = 0.001
    premature_advance: float = 0.03
    retain_angle_tolerance: float = 0.18


def advance_gate(angle_error, angular_velocity, clearance, settle, latched, rules):
    valid = (
        (angle_error.abs() <= rules.angle_tolerance)
        & (angular_velocity.abs() <= rules.velocity_tolerance)
        & (clearance >= rules.ready_clearance)
    )
    settle = torch.where(valid, settle + 1, torch.zeros_like(settle))
    ready = valid & (settle >= rules.settle_steps)
    # Latch records that a valid flip was observed; maintaining it is a separate requirement.
    latched = latched | ready
    maintained = (angle_error.abs() <= rules.retain_angle_tolerance) & (
        clearance >= -rules.penetration_tolerance
    )
    return settle, latched, latched & maintained, ready


def obb_separation(center, axes, half, other_center, other_axes, other_half):
    """SAT signed separating-axis margin for boxes; sign is exact, distance is not Euclidean.

    axes matrices store unit local axes in columns. Negative means overlap on all
    tested axes. Positive is a conservative lower bound on separation.
    """
    n = center.shape[0]
    other_axes = other_axes.expand(n, 3, 3)
    a = axes.transpose(-1, -2)
    b = other_axes.transpose(-1, -2)
    cross = torch.cross(
        a[:, :, None, :].expand(-1, 3, 3, -1), b[:, None, :, :].expand(-1, 3, 3, -1), dim=-1
    ).reshape(n, 9, 3)
    directions = torch.cat([a, b, cross], dim=1)
    norm = torch.linalg.vector_norm(directions, dim=-1)
    directions = directions / norm.clamp(min=1e-8)[..., None]
    delta = (center - other_center)[:, None, :]
    radius_a = (torch.matmul(directions, axes).abs() * half).sum(-1)
    radius_b = (torch.matmul(directions, other_axes).abs() * other_half).sum(-1)
    gap = (delta * directions).sum(-1).abs() - radius_a - radius_b
    gap = torch.where(norm > 1e-7, gap, torch.full_like(gap, -torch.inf))
    return gap.amax(-1)


class BackwardCurriculum:
    """Paper-inspired CSI; thresholds here are ELF3 trial choices, not paper defaults."""

    def __init__(self, phase=0.9, window=128, threshold=0.8, step=0.1):
        self.phase = float(phase)
        self.window = window
        self.threshold = threshold
        self.step = step
        self.samples = deque(maxlen=window)
        self.promotions = []
        self.episodes = 0

    def observe(self, successes, start_phases):
        # Do not promote on easy late resets that were in flight before a promotion.
        for success, start in zip(successes, start_phases):
            self.episodes += 1
            if abs(float(start) - self.phase) <= 0.03:
                self.samples.append(bool(success))
        if (
            len(self.samples) == self.window
            and sum(self.samples) / self.window >= self.threshold
            and self.phase > 0
        ):
            old = self.phase
            rate = sum(self.samples) / self.window
            self.phase = max(0.0, round(self.phase - self.step, 6))
            self.promotions.append(
                {
                    "episodes": self.episodes,
                    "from": old,
                    "to": self.phase,
                    "window_success_rate": rate,
                }
            )
            self.samples.clear()

    def state(self):
        return {
            "phase": self.phase,
            "window": self.window,
            "threshold": self.threshold,
            "step": self.step,
            "window_completed": len(self.samples),
            "window_success_rate": sum(self.samples) / max(1, len(self.samples)),
            "episodes": self.episodes,
            "promotions": self.promotions,
        }


def premature_step(feet, initial_feet, direction, latched, before_approach, rules):
    """Allow torso balance corrections, but reject forward foot motion before flip."""
    advance = ((feet[..., :2] - initial_feet[..., :2]) * direction).sum(-1).amax(-1)
    return before_approach & ~latched & (advance > rules.premature_advance), advance


def completion_bonus(remaining_steps, dt):
    # 6.5 exceeds the maximum positive dense reward rate (6.0). Completing now
    # must not pay less than waiting in an unfinished state for the same horizon.
    return 2.0 + 6.5 * remaining_steps.clamp(min=0) * dt
