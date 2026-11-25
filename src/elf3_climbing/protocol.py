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


def knee_training_phases(env_ids, phase, step, push_phase=None, *, assisted=False):
    """Rehearse late starts and reserve grounded trials for the push-off task."""
    phases = torch.full(env_ids.shape, phase, dtype=torch.float64, device=env_ids.device)
    phases[env_ids % 8 == 3] = 0.9
    phases[env_ids % 8 == 7] = min(0.9, round(phase + step, 6))
    if assisted:
        # Half rehearse the ground sequence, including starts before the extension.
        phases[env_ids % 4 == 0] = 0.0
        phases[env_ids % 8 == 2] = push_phase / 2
        phases[env_ids % 8 == 6] = push_phase
    elif push_phase is not None:
        phases[env_ids % 16 == 2] = 0.0
        phases[env_ids % 16 == 6] = push_phase
    return phases


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
            if abs(float(start) - self.phase) <= 1e-6:
                self.samples.append(bool(success))

    def promote(self, evaluated_phase, evaluation_success, min_evaluation_success=0.9, *, sampled_counts=None):
        """Advance once, only after an explicit matching-phase policy evaluation."""
        if abs(float(evaluated_phase) - self.phase) > 1e-6:
            return False
        count, total = (sum(self.samples), len(self.samples)) if sampled_counts is None else sampled_counts
        if not 0 <= count <= total:
            raise ValueError("Invalid sampled episode counts")
        rate = count / max(1, total)
        if (
            total >= self.window
            and rate >= self.threshold
            and evaluation_success >= min_evaluation_success
            and self.phase > 0
        ):
            old = self.phase
            self.phase = max(0.0, round(self.phase - self.step, 6))
            self.promotions.append(
                {
                    "episodes": self.episodes,
                    "from": old,
                    "to": self.phase,
                    "window_success_rate": rate,
                    "sampled_episodes": total,
                    "evaluation_success_rate": evaluation_success,
                }
            )
            self.samples.clear()
            return True
        return False

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
    # 10.5 exceeds the maximum positive dense reward rate (9.8). Completing now
    # must not pay less than waiting in an unfinished state for the same horizon.
    return 2.0 + 10.5 * remaining_steps.clamp(min=0) * dt


def velocity_tracking_errors(linear, angular, target_linear, target_angular):
    """Track commanded motion; absolute forward speed is not a balance violation."""
    return (
        torch.linalg.vector_norm(linear - target_linear, dim=-1),
        torch.linalg.vector_norm(angular - target_angular, dim=-1),
    )


def held_velocity_target(reference_velocity, held):
    """A position frozen at the segment boundary must have a zero velocity target."""
    return torch.where(held[:, None], 0.0, reference_velocity)


def tolerant_velocity_reward(error, tolerance):
    """Equal reward throughout the small-error band, smooth decay beyond it."""
    excess = (error - tolerance).clamp(min=0)
    return torch.exp(-excess.square() / 0.25)


def flip_retention_failures(angle_error, clearance, latched, rules):
    """Separate loss of wrist angle from a gripper collision; retain the same failure union."""
    lost_angle = latched & (angle_error.abs() > rules.retain_angle_tolerance)
    collision = clearance < -rules.penetration_tolerance
    return lost_angle, collision
