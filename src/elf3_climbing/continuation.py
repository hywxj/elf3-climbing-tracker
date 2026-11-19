"""Retain the accepted standing/flip actor and learn a bounded leg correction."""

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic


class StandingLegResidual(nn.Module):
    def __init__(self, base, leg_indices, bound=0.15):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.correction = nn.Sequential(nn.Linear(632, 128), nn.ELU(), nn.Linear(128, 29))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        mask = torch.zeros(29)
        mask[leg_indices] = bound
        self.register_buffer("mask", mask)

    def forward(self, obs):
        # Column 631 was already a task-phase flag; the original flip actor has
        # zero weights on this column. Normalizer stays fixed at mean=0,std=1.
        gate = (obs[:, 631:632] > 0.5).to(obs.dtype)
        correction = torch.tanh(self.correction(obs.clamp(-5.0, 5.0)))
        return self.base(obs) + gate * self.mask * correction


class StandingContinuationActorCritic(ActorCritic):
    def __init__(self, *args, leg_indices, residual_bound=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = StandingLegResidual(self.actor, leg_indices, residual_bound)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        # Protect the learned flip/stand prefix from large leg exploration.
        # This is part of the policy distribution, so PPO sees the actual std.
        moving = observations[:, 631:632] > 0.5
        std = torch.where(moving, std, std.clamp(max=0.001))
        self.distribution = Normal(mean, std)


def inherit_standing_checkpoint(checkpoint):
    """Map old actor keys without reinitializing any learned parameter."""
    state = checkpoint["model_state_dict"]
    if any(k.startswith("actor.base.") for k in state):
        return state, False
    # Keep all critic and exploration tensors, preserve every actor tensor.
    mapped = {("actor.base." + k[6:] if k.startswith("actor.") else k): v for k, v in state.items()}
    return mapped, True


def freeze_prior_statistics(runner):
    runner.obs_normalizer.until = int(runner.obs_normalizer.count)
    mean = runner.obs_normalizer._mean[0, 631]
    std = runner.obs_normalizer._std[0, 631]
    if abs(float(mean)) > 1.0e-6 or abs(float(std) - 1.0) > 1.0e-6:
        raise ValueError("Continuation requires the accepted flip phase-channel normalization")
