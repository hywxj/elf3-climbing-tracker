"""Score a predicted contact against the later, sustained physical center of pressure."""

import math

import torch


class ContactPlanReward:
    def __init__(self, count, device, dt):
        self.minimum = torch.tensor([15, 15, 3, 3, 3, 3, 8, 8], device=device)
        self.horizon = round(1.5 / dt)
        self.dwell = math.ceil(0.20 / dt - 1e-8)
        self.ttl = torch.zeros((count, 8), device=device, dtype=torch.long)
        self.empty = torch.zeros_like(self.ttl)
        self.loaded = torch.zeros_like(self.ttl)
        self.target = torch.zeros((count, 8, 3), device=device)
        self.match = torch.zeros_like(self.target[..., 0])

    def reset(self, ids):
        for field in (self.ttl, self.empty, self.loaded, self.target, self.match):
            field[ids] = 0

    def propose(self, valid, points):
        # Commit before touchdown. A new prediction cannot move a goal under a loaded limb.
        new = valid & (self.empty >= 3) & (self.ttl == 0)
        self.ttl.masked_fill_(new, self.horizon)
        self.target.copy_(torch.where(new[..., None], points, self.target))

    def measure(self, force, centers):
        bearing = (force >= self.minimum) & torch.isfinite(centers).all(2)
        self.empty.copy_(torch.where(bearing, 0, self.empty + 1))
        active = bearing & (self.ttl > 0)
        self.loaded.copy_(torch.where(active, self.loaded + 1, 0))
        error_sq = (centers.nan_to_num() - self.target).square().sum(2)
        similarity = torch.exp(-error_sq / 0.10**2)
        self.match.copy_(torch.where(active, self.match + similarity, 0.0))
        complete = self.loaded == self.dwell
        reward = ((self.match / self.dwell) * complete).sum(1)
        self.ttl.sub_(1).clamp_min_(0).masked_fill_(complete, 0)
        return reward
