import unittest

import torch
from torch import nn

from elf3_climbing.continuation import StandingLegResidual, StandingContinuationActorCritic


class ContinuationTests(unittest.TestCase):
    def test_zero_residual_preserves_every_prior_output(self):
        torch.manual_seed(42)
        base = nn.Sequential(nn.Linear(632, 16), nn.ELU(), nn.Linear(16, 29))
        module = StandingLegResidual(base, list(range(15)))
        obs = torch.randn(20, 632)
        obs[:10, 631], obs[10:, 631] = 0.0, 1.0
        torch.testing.assert_close(module(obs), base(obs), rtol=0, atol=0)

    def test_learning_legs_cannot_change_prefix_or_arms_or_prior(self):
        torch.manual_seed(7)
        base = nn.Sequential(nn.Linear(632, 16), nn.ELU(), nn.Linear(16, 29))
        module = StandingLegResidual(base, list(range(15)))
        original = {k: v.clone() for k, v in base.state_dict().items()}
        obs = torch.randn(8, 632)
        obs[:, 631] = 1.0
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            module(obs).sum().backward()
            optimizer.step()
        delta = module(obs) - base(obs)
        self.assertGreater(float(delta[:, :15].abs().max()), 0.0)
        self.assertLessEqual(float(delta[:, :15].abs().max()), 0.150001)
        torch.testing.assert_close(delta[:, 15:], torch.zeros_like(delta[:, 15:]), rtol=0, atol=0)
        obs[:, 631] = 0.0
        torch.testing.assert_close(module(obs), base(obs), rtol=0, atol=0)
        for k, v in original.items():
            torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)

    def test_ppo_distribution_uses_actual_phase_dependent_noise(self):
        policy = StandingContinuationActorCritic(
            632,
            697,
            29,
            leg_indices=list(range(15)),
            actor_hidden_dims=[16],
            critic_hidden_dims=[16],
            noise_std_type="log",
            init_noise_std=0.02,
        )
        obs = torch.zeros(2, 632)
        obs[1, 631] = 1.0
        policy.update_distribution(obs)
        torch.testing.assert_close(policy.action_std[0], torch.full((29,), 0.001))
        torch.testing.assert_close(policy.action_std[1], torch.full((29,), 0.02))
