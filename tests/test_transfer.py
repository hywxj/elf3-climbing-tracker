import unittest
from types import SimpleNamespace

import torch
from torch import nn

from elf3_climbing.transfer import standing_to_flip


class StandingTransfer(unittest.TestCase):
    def test_other_outputs_preserved_at_old_channel_means_and_new_inputs_are_bounded(self):
        torch.manual_seed(13)
        policy = SimpleNamespace(
            actor=nn.Sequential(nn.Linear(632, 32), nn.ELU(), nn.Linear(32, 29)),
            critic=nn.Sequential(nn.Linear(697, 32), nn.ELU(), nn.Linear(32, 1)),
        )
        norms = [
            SimpleNamespace(
                _mean=torch.randn(1, n), _var=torch.ones(1, n), _std=torch.ones(1, n) * 0.001
            )
            for n in (632, 697)
        ]
        wrist = 20
        columns = [h * 93 + o + wrist for h in range(5) for o in (6, 35, 64)]
        columns += [626, 627, 628, 629, 631]
        raw = torch.randn(4, 632) * 0.01 + norms[0]._mean
        raw[:, columns] = norms[0]._mean[:, columns]
        before = policy.actor((raw - norms[0]._mean) / (norms[0]._std + 0.01)).detach()
        remaining = [i for i in range(632) if i not in columns]
        old_means = norms[0]._mean.clone()
        standing_to_flip(policy, *norms, wrist, 0.0)
        raw[:, columns] = -1.309
        after = policy.actor((raw - norms[0]._mean) / (norms[0]._std + 0.01)).detach()
        body = [i for i in range(29) if i != wrist]
        torch.testing.assert_close(after[:, body], before[:, body])
        torch.testing.assert_close(after[:, wrist], torch.zeros(4))
        torch.testing.assert_close(norms[0]._mean[:, remaining], old_means[:, remaining])
        self.assertTrue(torch.isfinite(after).all())
