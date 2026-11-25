"""Physical completion criteria for the single-ledge contact student (SI units)."""

import math

import torch


def direct_joint_targets(action, nominal, previous_target, limits, rates, dt):
    desired = (nominal + action).clamp(limits[..., 0], limits[..., 1])
    delta = (desired - previous_target).clamp(-rates * dt, rates * dt)
    return (previous_target + delta).clamp(limits[..., 0], limits[..., 1])


def ground_reset_offsets(ground_start, forward, generator=None):
    """Horizontal launch uncertainty along and across the route, in meters."""
    u = torch.rand((len(ground_start), 2), device=forward.device, generator=generator)
    distances = (2 * u - 1) * forward.new_tensor([0.025, 0.015])
    left = torch.stack((-forward[1], forward[0]))
    return (distances[:, :1] * forward + distances[:, 1:] * left) * ground_start[:, None]


class FrontApproach:
    """Bound walking rewards by new net foot progress toward the platform's finite front face."""

    def __init__(self, count, center, half, quat, start_xy):
        w, x, y, z = quat.unbind()
        axes = torch.stack((
            torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z))),
            torch.stack((2 * (x*y - w*z), 1 - 2 * (x*x + z*z))),
        ))
        offset = axes @ (start_xy - center[:2])
        face = int((offset.abs() / half[:2]).argmax())
        self.outward = axes[face] * torch.sign(offset[face])
        self.side = axes[1 - face]
        self.edge = center[:2] + self.outward * half[face]
        self.half_width = half[1 - face]
        self.start_gap = center.new_zeros(count)
        self.best = center.new_zeros(count)
        self.last_step = center.new_zeros(count)
        self.last_plan = center.new_zeros((count, 2))

    def update(self, feet_xy, first, switched):
        relative = feet_xy - self.edge
        gap = relative @ self.outward
        lateral = (relative @ self.side).abs()
        # Both foot landmarks must still approach the front, with clearance for the shoe
        # at the ends of the physical top. Neither side nor the back is a walking target.
        corridor = (lateral <= self.half_width - 0.06).all(1) & (gap >= 0).all(1)
        distance = gap.clamp_min(0).mean(1)
        self.start_gap.copy_(torch.where(first, distance, self.start_gap))
        self.best.masked_fill_(first, 0)
        self.last_step.masked_fill_(first, 0)
        self.last_plan.masked_fill_(first[:, None], 0)
        progress = (self.start_gap - distance).clamp_min(0)
        gain = (progress - self.best).clamp_min(0) * corridor
        self.best.copy_(torch.maximum(self.best, progress))
        # Do not bank off-course distance to cash it in when re-entering the corridor.
        self.last_step.copy_(torch.where(corridor, self.last_step, self.best))
        self.last_plan.copy_(torch.where(corridor[:, None], self.last_plan, self.best[:, None]))
        near_frontier = corridor & (progress >= self.best - 0.02)
        # A new left/right switch must buy at least one detector-length (4.5 cm)
        # of *net* progress. Repeating a previously paid piece of ground earns nothing.
        qualified = switched & near_frontier & (progress >= self.last_step + 0.045)
        self.last_step.copy_(torch.where(qualified, progress, self.last_step))
        return gain, qualified, progress, near_frontier

    def foot_plan_reward(self, reward, matches, complete, braced, progress, near_frontier):
        """Gate only pre-brace foot-plan matches, independently of arm and knee tracking."""
        valid = near_frontier[:, None] & (progress[:, None] >= self.last_plan + 0.045)
        paid = complete & valid & ~braced[:, None]
        self.last_plan.copy_(torch.where(paid, progress[:, None], self.last_plan))
        denied = complete & ~valid & ~braced[:, None]
        return reward - (matches * denied).sum(1)


class ContactMilestones:
    """Record events in order. Seeded episodes can earn knee credit but never full-course credit."""

    def __init__(self, count, device, step_dt):
        self.dt = step_dt
        self.hold_steps = math.ceil(0.20 / step_dt - 1e-8)
        self.walk_switches = torch.zeros(count, device=device, dtype=torch.long)
        self.hold = torch.zeros_like(self.walk_switches)
        self.brace_steps = torch.zeros_like(self.walk_switches)
        self.braced = torch.zeros_like(self.walk_switches, dtype=torch.bool)
        self.walked_to_edge = self.braced.clone()
        self.push_observed = self.braced.clone()
        self.pushed_after_walk = self.braced.clone()
        self.airborne = self.braced.clone()

    def reset(self, ids):
        for value in (
            self.walk_switches,
            self.hold,
            self.brace_steps,
            self.braced,
            self.walked_to_edge,
            self.push_observed,
            self.pushed_after_walk,
            self.airborne,
        ):
            value[ids] = 0

    def update(
        self, switch, foot_advance, four_point, pushed, airborne, knee_contact, ground_start
    ):
        self.walk_switches += (switch & ~self.braced).long()
        self.brace_steps.copy_(torch.where(four_point, self.brace_steps + 1, 0))
        # A supported stand at the platform must happen before launch. Extra steps after
        # the robot has already jumped cannot retroactively satisfy the walking task.
        brace_now = self.brace_steps >= 3
        self.braced |= brace_now
        self.walked_to_edge |= (
            brace_now & ~self.airborne & (self.walk_switches >= 1) & (foot_advance >= 0.25)
        )
        self.push_observed |= self.braced & pushed
        self.pushed_after_walk |= self.walked_to_edge & pushed
        self.airborne |= self.push_observed & airborne
        self.hold.copy_(torch.where(self.airborne & knee_contact, self.hold + 1, 0))
        knee = self.hold >= self.hold_steps
        return knee, ground_start & self.pushed_after_walk & knee


def accepted_knee(metrics, floor_load, footprint_margin, body_weight):
    """A real accepted transfer and loaded knee on the top while both feet clear the ground."""
    return (
        metrics["right_knee_supported"]
        & ~metrics["wrist_forbidden"]
        & (metrics["right_knee_top_load_N"] > 8)
        & (metrics["right_knee_top_load_N"] / body_weight >= 0.15)
        & (metrics["right_knee_off_patch_load_N"] <= 8)
        & (metrics["right_knee_inside_edge_m"] >= 0)
        & (footprint_margin >= 0)
        & (metrics["min_foot_clearance_m"] > 0.01)
        & (floor_load < 3).all(1)
    )
