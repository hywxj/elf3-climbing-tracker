import unittest
import torch
from torch import nn

from elf3_climbing.walk import Footsteps, HALF, WALK_STEPS, reward_rates, walk_command
from elf3_climbing.walk_policy import WalkActor
from elf3_climbing.walk_training import step_summary, walk_checkpoint_score


class WalkTaskTests(unittest.TestCase):
    def test_flip_prefix_and_arm_outputs_survive_nonzero_leg_corrections(self):
        actor = WalkActor(nn.Linear(632, 29), 632, list(range(15)))
        x = torch.zeros(2, 636)
        x[1, -1] = 1
        with torch.no_grad():
            actor.legs[-1].bias.fill_(0.3)
        prior = actor.base(actor.base_norm(x[:, :632]))
        output = actor(x)
        torch.testing.assert_close(output[0], prior[0], atol=0, rtol=0)
        torch.testing.assert_close(output[1, 15:], prior[1, 15:], atol=0, rtol=0)
        self.assertTrue((output[1, :15] > prior[1, :15]).all())
        self.assertEqual(actor.base_norm.count.item(), 0)
        self.assertFalse(any(p.requires_grad for p in actor.base.parameters()))

    def test_schedule_has_two_complete_windows_and_a_stop(self):
        t = torch.tensor([-1, 0, HALF - 1, HALF, WALK_STEPS - 1, WALK_STEPS])
        s, f = walk_command(
            t, torch.tensor([[0.12, 0.32]]).repeat(len(t), 1), torch.zeros(len(t)), 0.02
        )
        torch.testing.assert_close(s, torch.tensor([0, 0.12, 0.12, 0.32, 0.32, 0]))
        self.assertEqual(f[:, -1].tolist(), [0, 1, 1, 1, 1, 1])
        self.assertEqual(f[-1, 1:3].tolist(), [0, 0])

    def test_physical_alternation(self):
        feet = Footsteps(1, "cpu")
        moving = torch.tensor([True])
        x = torch.zeros(1, 2)

        def update(force, height):
            return feet.update(moving, torch.tensor([force]), torch.tensor([height]), x).item()

        # Sliding forward while loaded never earns a step.
        x += 0.20
        self.assertFalse(update([40, 40], [0.0, 0.0]))
        for side, expect_credit in ((0, False), (0, False), (1, True)):
            force, height = [50, 50], [0.0, 0.0]
            force[side], height[side] = 0, 0.05
            for _ in range(4):
                update(force, height)
                x[0, side] += 0.025
            self.assertEqual(update([50, 50], [0.0, 0.0]), expect_credit)
        for _ in range(4):
            update([0, 0], [0.05, 0.05])
            x += 0.04
        self.assertFalse(update([50, 50], [0.0, 0.0]))
        self.assertEqual(feet.counts.tolist(), [[1, 1]])
        feet.reset(torch.tensor([0]))
        self.assertEqual(feet.counts.tolist(), [[0, 0]])

    def test_checkpoint_prioritizes_completed_walking_over_one_foot_then_falling(self):
        def result(steps, fallen):
            episode = {"left_steps": steps[0], "right_steps": steps[1], "terminated": fallen}
            return dict(
                step_summary([episode]),
                walk_success_fraction=0,
                survived_fraction=int(not fallen),
                halt_hold_s=0,
                forward_distance_m=0,
            )

        stationary = result((0, 0), False)
        first_then_fall = result((1, 0), True)
        both_then_fall = result((2, 1), True)
        safe_pair = result((1, 1), False)
        self.assertEqual(first_then_fall["alternating_steps"], 0)
        self.assertEqual(both_then_fall["alternating_steps"], 2)
        self.assertEqual(both_then_fall["surviving_step_pairs_per_episode"], 0)
        self.assertGreater(
            walk_checkpoint_score(stationary), walk_checkpoint_score(first_then_fall)
        )
        self.assertGreater(walk_checkpoint_score(stationary), walk_checkpoint_score(both_then_fall))
        self.assertGreater(walk_checkpoint_score(safe_pair), walk_checkpoint_score(stationary))

    def test_velocity_band_and_stop_scoring(self):
        cmd = torch.tensor([0.30, 0.30, 0.30, 0.0, 0.0])
        forward = torch.tensor([0.30, 0.32, 0.0, 0.0, 0.30])
        z = torch.zeros_like(cmd)
        pairs = torch.zeros(len(z), 2)
        scores = reward_rates(cmd, forward, z, z, z, z, z, pairs, 40 + pairs, z, z, z, z)
        t = scores["forward_velocity"]
        self.assertEqual(t[0], t[1])
        self.assertGreater(t[0], 5 * t[2])
        self.assertEqual(scores["stopped_double_support"][:3].tolist(), [0, 0, 0])
        self.assertGreater(scores["stopped_double_support"][3], scores["stopped_double_support"][4])
