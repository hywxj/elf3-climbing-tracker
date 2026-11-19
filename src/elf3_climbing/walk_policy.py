"""Walking branch with the accepted actor and its exact input transform embedded."""

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from .walk import FEATURES


class WalkActor(nn.Module):
    def __init__(self, base, base_size, leg_indices):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.base_norm = EmpiricalNormalization(base_size, until=0)
        self.legs = nn.Sequential(
            nn.Linear(5 * 93 + FEATURES, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, len(leg_indices)),
        )
        nn.init.zeros_(self.legs[-1].weight)
        nn.init.zeros_(self.legs[-1].bias)
        self.register_buffer("leg_ids", torch.tensor(leg_indices, dtype=torch.long))

    def forward(self, obs):
        features = obs[:, -FEATURES:]
        base = self.base(self.base_norm(obs[:, :-FEATURES]))
        inputs = torch.cat((obs[:, : 5 * 93], features), dim=1)
        delta = 0.9 * self.legs(inputs).tanh()
        # Downstream actuators clip actions to +/-3. At the gate, avoid a saturated
        # prior consuming the entire correction budget while away from its training poses.
        legs = base[:, self.leg_ids]
        target = torch.where(features[:, -1:] > 0.5, legs.clamp(-3, 3) + delta, legs)
        return base.index_copy(1, self.leg_ids, target)


class WalkActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, leg_indices, **kwargs):
        super().__init__(num_actor_obs - FEATURES, num_critic_obs, num_actions, **kwargs)
        self.actor = WalkActor(self.actor, num_actor_obs - FEATURES, leg_indices)
        self.log_std.requires_grad_(False)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(
            mean, torch.where(observations[:, -1:] > 0.5, std, std.clamp(max=0.001))
        )

    @torch.no_grad()
    def inherit_flip(self, checkpoint, leg_std, wrist_std):
        state = checkpoint["model_state_dict"]
        self.actor.base.load_state_dict(
            {k[6:]: v for k, v in state.items() if k.startswith("actor.")}
        )
        self.actor.base_norm.load_state_dict(checkpoint["obs_norm_state_dict"])
        self.log_std.fill_(float(torch.tensor(wrist_std).log()))
        self.log_std[self.actor.leg_ids] = float(torch.tensor(leg_std).log())
