"""Reward transferring weight to the feet and raising the hips after knee entry."""

import torch


class SupportReward:
    def __init__(self, task):
        self.task = task
        self.env = task.env
        self.best = torch.zeros_like(task.body_weight)

    def reset(self, ids):
        self.task.reset(ids)
        self.best[ids] = 0

    def measure(self):
        result = self.task.measure()
        feet = result["foot_bearing"]
        # Count only soles that have stayed loaded and inside the upper footprint.
        supported = (result["foot_top_N"] * feet).sum(dim=1) / self.task.body_weight
        supported = supported.clamp(0, 1) * self.env.mount_active
        upright = result["torso_up"].clamp(0, 1)
        hip_lift = ((result["hip_above_top_m"] - 0.10) / 0.45).clamp(0, 1)
        bilateral = feet.all(dim=1)
        quality = supported * (0.5 + 0.5 * bilateral) * (0.3 * upright + 0.7 * upright * hip_lift)
        gain = (quality - self.best).clamp_min(0)
        self.best.copy_(torch.maximum(self.best, quality))
        # A transient impact can raise the peak once. Quieter, nearly upright
        # bilateral support continues earning credit while the robot stands up.
        quiet = (1 - result["root_speed_m_s"] / 1.0).clamp(0, 1)
        held = quality * bilateral * quiet * result["ground_clear"] * (upright > 0.7)
        result["reward"] = result["reward"] + 6.0 * gain + self.env.step_dt * held
        return result
