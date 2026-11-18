import unittest
import torch
from elf3_climbing.protocol import GripperRules, advance_gate, obb_separation, BackwardCurriculum


class GateTests(unittest.TestCase):
    def test_requires_ten_consecutive_samples(self):
        r = GripperRules()
        settle = torch.tensor([0])
        latched = torch.tensor([False])
        for i in range(10):
            settle, latched, allowed, ready = advance_gate(
                torch.tensor([0.01]), torch.tensor([0.1]), torch.tensor([0.02]), settle, latched, r
            )
            self.assertEqual(bool(allowed[0]), i == 9)
        self.assertTrue(bool(ready[0]))

    def test_velocity_or_clearance_resets_hold(self):
        r = GripperRules()
        settle, latch, allowed, ready = advance_gate(
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.3, 0.0]),
            torch.tensor([0.03, 0.005]),
            torch.tensor([9, 9]),
            torch.tensor([False, False]),
            r,
        )
        self.assertEqual(settle.tolist(), [0, 0])
        self.assertFalse(allowed.any())

    def test_latch_does_not_allow_loss_of_flip(self):
        r = GripperRules()
        _, latched, allowed, _ = advance_gate(
            torch.tensor([0.3, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.02, -0.002]),
            torch.tensor([10, 10]),
            torch.tensor([True, True]),
            r,
        )
        self.assertTrue(latched.all())
        self.assertFalse(allowed.any())


class GeometryTests(unittest.TestCase):
    def test_signed_boxes(self):
        axes = torch.eye(3).expand(3, -1, -1)
        c = torch.tensor([[3.0, 0, 0], [2.0, 0, 0], [1.5, 0, 0]])
        d = obb_separation(c, axes, torch.ones(3), torch.zeros(3), torch.eye(3), torch.ones(3))
        torch.testing.assert_close(d, torch.tensor([1.0, 0.0, -0.5]))

    def test_rotation_invariant(self):
        a = torch.tensor(0.63)
        R = torch.tensor([[a.cos(), -a.sin(), 0], [a.sin(), a.cos(), 0], [0, 0, 1.0]])
        h = torch.tensor([1.0, 0.3, 0.2])
        c = torch.tensor([[2.5, 0.1, 0.0]])
        x = obb_separation(c, torch.eye(3)[None], h, torch.zeros(3), torch.eye(3), h)
        y = obb_separation(c @ R.T, R[None], h, torch.zeros(3), R, h)
        torch.testing.assert_close(x, y)


class CurriculumTests(unittest.TestCase):
    def test_failure_does_not_promote(self):
        c = BackwardCurriculum(window=4)
        c.observe([False] * 4, [0.9] * 4)
        self.assertEqual(c.phase, 0.9)

    def test_promotion_rejects_stale_easy_episodes(self):
        c = BackwardCurriculum(window=4)
        c.observe([True] * 4, [0.9] * 4)
        c.promote(0.9, 1.0)
        self.assertEqual(c.phase, 0.8)
        c.observe([True] * 4, [0.9] * 4)
        self.assertEqual(c.phase, 0.8)
        self.assertEqual(len(c.samples), 0)
        c.observe([True] * 4, [0.8] * 4)
        c.promote(0.8, 1.0)
        self.assertEqual(c.phase, 0.7)

    def test_requires_matching_phase_evaluation(self):
        c = BackwardCurriculum(window=4)
        c.observe([True] * 4, [0.9] * 4)
        self.assertEqual(c.phase, 0.9)
        self.assertFalse(c.promote(0.8, 1.0))
        self.assertFalse(c.promote(0.9, 0.5))
        self.assertTrue(c.promote(0.9, 1.0))
        self.assertEqual(c.phase, 0.8)

    def test_phase_has_floor(self):
        c = BackwardCurriculum(phase=0.1, window=2)
        c.observe([True, True], [0.1, 0.1])
        c.promote(0.1, 1.0)
        self.assertEqual(c.phase, 0.0)
        c.observe([True, True], [0.0, 0.0])
        c.promote(0.0, 1.0)
        self.assertEqual(c.phase, 0.0)


class TaskSemanticsTests(unittest.TestCase):
    def test_fixed_feet_do_not_block_balance(self):
        from elf3_climbing.protocol import premature_step

        feet = torch.zeros(1, 2, 3)
        r = GripperRules()
        early, _ = premature_step(
            feet, feet, torch.tensor([1.0, 0.0]), torch.tensor([False]), torch.tensor([True]), r
        )
        self.assertFalse(early.any())
        moved = feet.clone()
        moved[0, 0, 0] = 0.04
        early, _ = premature_step(
            moved, feet, torch.tensor([1.0, 0.0]), torch.tensor([False]), torch.tensor([True]), r
        )
        self.assertTrue(early.any())
        early, _ = premature_step(
            moved, feet, torch.tensor([1.0, 0.0]), torch.tensor([True]), torch.tensor([True]), r
        )
        self.assertFalse(early.any())

    def test_completion_beats_waiting_for_dense_reward(self):
        from elf3_climbing.protocol import completion_bonus

        now = completion_bonus(torch.tensor([40]), 0.02)
        later = completion_bonus(torch.tensor([0]), 0.02) + 9.8 * 40 * 0.02
        self.assertGreater(float(now), float(later))


class VelocityTrackingTests(unittest.TestCase):
    def test_forward_movement_and_turning_are_allowed(self):
        from elf3_climbing.protocol import velocity_tracking_errors

        linear = torch.tensor([[0.7, 0.0, 0.0]])
        angular = torch.tensor([[0.0, 0.0, 0.8]])
        le, ae = velocity_tracking_errors(linear, angular, linear, angular)
        self.assertEqual(le.item(), 0.0)
        self.assertEqual(ae.item(), 0.0)

    def test_uncommanded_motion_has_error(self):
        from elf3_climbing.protocol import velocity_tracking_errors

        zero = torch.zeros(1, 3)
        le, ae = velocity_tracking_errors(torch.tensor([[0.7, 0.0, 0.0]]), zero, zero, zero)
        self.assertGreater(le.item(), 0.25)
        self.assertEqual(ae.item(), 0.0)

    def test_neighboring_small_curriculum_steps_do_not_mix_samples(self):
        c = BackwardCurriculum(phase=0.875, step=0.025)
        c.observe([True, False], [0.9, 0.875])
        self.assertEqual(list(c.samples), [False])


class HeldReferenceTests(unittest.TestCase):
    def test_terminal_hold_zeros_velocity_without_stopping_motion_phase(self):
        from elf3_climbing.protocol import held_velocity_target

        reference = torch.tensor([[0.7, 0.0, 0.2], [0.025, -0.03, 0.01]])
        target = held_velocity_target(reference, torch.tensor([False, True]))
        torch.testing.assert_close(target[0], reference[0])
        torch.testing.assert_close(target[1], torch.zeros(3))
        self.assertNotEqual(float(reference[1, 0]), 0.0)


class TolerantVelocityRewardTests(unittest.TestCase):
    def test_small_nonzero_errors_receive_full_reward(self):
        from elf3_climbing.protocol import tolerant_velocity_reward

        for tolerance in (0.05, 0.1):
            errors = torch.tensor([0.0, tolerance / 2, tolerance], requires_grad=True)
            scores = tolerant_velocity_reward(errors, tolerance)
            torch.testing.assert_close(scores, torch.ones(3))
            scores.sum().backward()
            torch.testing.assert_close(errors.grad, torch.zeros(3))

    def test_larger_errors_reduce_reward_smoothly(self):
        from elf3_climbing.protocol import tolerant_velocity_reward

        errors = torch.tensor([0.05, 0.05001, 0.1, 0.3, 0.8])
        scores = tolerant_velocity_reward(errors, 0.05)
        self.assertTrue(torch.all(scores[1:] <= scores[:-1]))
        self.assertAlmostEqual(float(scores[1]), 1.0, places=6)
        self.assertTrue(torch.all((scores > 0) & (scores <= 1)))

    def test_moving_reference_allows_small_tracking_error(self):
        from elf3_climbing.protocol import velocity_tracking_errors, tolerant_velocity_reward

        target = torch.tensor([[0.7, 0.0, 0.0]])
        actual = torch.tensor([[0.67, 0.0, 0.0]])
        zero = torch.zeros(1, 3)
        error, _ = velocity_tracking_errors(actual, zero, target, zero)
        self.assertGreater(float(error), 0.0)
        self.assertEqual(float(tolerant_velocity_reward(error, 0.05)), 1.0)


if __name__ == "__main__":
    unittest.main()
