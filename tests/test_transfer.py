import unittest
from types import SimpleNamespace

import torch
from torch import nn

from elf3_climbing.transfer import standing_to_flip, preserve_standing_feedback


class StandingTransfer(unittest.TestCase):
    def test_wrist_updates_preserve_body_feedback_and_normalization(self):
        from rsl_rl.modules.normalizer import EmpiricalNormalization

        torch.manual_seed(7)
        actor = nn.Sequential(nn.Linear(12, 16), nn.ELU(), nn.Linear(16, 29))
        policy = SimpleNamespace(actor=actor)
        normalizer = EmpiricalNormalization(12)
        samples = torch.randn(32, 12)
        normalizer(samples)
        old_stats = {k: v.clone() for k, v in normalizer.state_dict().items()}
        before = actor(normalizer(samples)).detach().clone()
        # Capture after this final statistics update, then freeze for training.
        old_stats = {k: v.clone() for k, v in normalizer.state_dict().items()}
        handles = preserve_standing_feedback(policy, normalizer, 20)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            actor(normalizer(samples)).sum().backward()
            optimizer.step()
        normalizer(samples * 100)
        after = actor(normalizer(samples)).detach()
        body = [i for i in range(29) if i != 20]
        torch.testing.assert_close(after[:, body], before[:, body], rtol=0, atol=0)
        self.assertFalse(torch.equal(after[:, 20], before[:, 20]))
        for k, value in normalizer.state_dict().items():
            torch.testing.assert_close(value, old_stats[k], rtol=0, atol=0)
        for handle in handles:
            handle.remove()

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
        standing_to_flip(policy, *norms, wrist, -1.309)
        raw[:, columns] = -1.309
        after = policy.actor((raw - norms[0]._mean) / (norms[0]._std + 0.01)).detach()
        body = [i for i in range(29) if i != wrist]
        torch.testing.assert_close(after[:, body], before[:, body])
        torch.testing.assert_close(after[:, wrist], torch.full((4,), -1.309))
        torch.testing.assert_close(norms[0]._mean[:, remaining], old_means[:, remaining])
        self.assertTrue(torch.isfinite(after).all())
