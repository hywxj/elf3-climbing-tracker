import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from elf3_climbing.training_control import (
    EvaluationGuard,
    BalanceExploration,
    initial_balance_std,
    finalize_checkpoint,
    save_checkpoint_atomically,
)


class GuardTests(unittest.TestCase):
    def test_stops_sustained_regression(self):
        guard = EvaluationGuard(patience=2, best_start=1, best_tail=1)
        self.assertIsNone(guard.inspect(0, 0, 0.8, 0, 0))
        self.assertEqual(guard.inspect(0, 0, 0.8, 0, 0), "evaluation_regression")

    def test_harder_phase_does_not_reset_retention_requirement(self):
        guard = EvaluationGuard(patience=2, best_tail=1)
        self.assertIsNone(guard.inspect(0, 0, 0.8, 0, 0))
        self.assertEqual(guard.inspect(0, 0, 0.7, 0, 0), "evaluation_regression")

    def test_stops_unproductive_stage_without_false_convergence(self):
        guard = EvaluationGuard(patience=2)
        self.assertIsNone(guard.inspect(0, 0, 0.9, 0, 0))
        self.assertEqual(guard.inspect(0, 0, 0.9, 0, 0), "no_task_success")

    def test_recovery_clears_regression(self):
        guard = EvaluationGuard(patience=2, best_tail=1)
        self.assertIsNone(guard.inspect(0, 0, 0.8, 0, 0))
        self.assertIsNone(guard.inspect(0, 1, 0.8, 1, 1))
        self.assertEqual(guard.bad_streak, 0)

    def test_measured_hold_progress_defers_stall_but_not_forever(self):
        guard = EvaluationGuard(patience=2, min_iterations_before_stall=200)
        for step, progress in [(200, 1.0), (220, 1.04), (240, 1.11), (260, 1.12)]:
            self.assertIsNone(
                guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=step, progress=progress)
            )
        self.assertEqual(
            guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=280, progress=1.12),
            "no_task_success",
        )

    def test_hold_progress_does_not_hide_regression_of_accepted_skill(self):
        guard = EvaluationGuard(patience=2, best_tail=1)
        self.assertIsNone(guard.inspect(0, 0, 0.8, 0, 0, progress=1.0))
        self.assertEqual(guard.inspect(0, 0, 0.8, 0, 0, progress=2.0), "evaluation_regression")


class ExplorationTests(unittest.TestCase):
    def test_warm_start_preserves_precision_and_explicit_overrides(self):
        self.assertEqual(initial_balance_std(0.08, None, 1), 0.08)
        self.assertEqual(initial_balance_std(0.08, 0.08, 0), 0.08)
        self.assertEqual(initial_balance_std(0.08, 0.08, 1), 0.02)
        self.assertEqual(initial_balance_std(0.08, 0.01, 1), 0.01)
        self.assertEqual(initial_balance_std(0.08, 0.08, 1, explicit=True), 0.08)

    def test_no_success_does_not_shrink_exploration(self):
        schedule = BalanceExploration()
        for _ in range(10):
            self.assertFalse(schedule.observe(0.9, 0.0, 0.0))
        self.assertEqual(schedule.std, 0.08)

    def test_repeated_mean_success_reduces_noise_until_floor(self):
        schedule = BalanceExploration()
        for expected in [0.04, 0.02, 0.01]:
            self.assertFalse(schedule.observe(0.9, 1.0, 0.0))
            self.assertTrue(schedule.observe(0.9, 1.0, 0.0))
            self.assertEqual(schedule.std, expected)
        self.assertFalse(schedule.observe(0.9, 1.0, 0.0))
        self.assertFalse(schedule.observe(0.9, 1.0, 0.0))

    def test_phase_change_requires_fresh_evidence(self):
        schedule = BalanceExploration()
        schedule.observe(0.9, 1.0, 0.0)
        self.assertFalse(schedule.observe(0.875, 1.0, 0.0))
        self.assertEqual(schedule.std, 0.08)

    def test_warmup_delays_no_success_stop_but_not_regression(self):
        guard = EvaluationGuard(patience=2, min_iterations_before_stall=200)
        for step in [20, 40, 60, 180]:
            self.assertIsNone(guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=step))
        self.assertIsNone(guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=200))
        self.assertEqual(
            guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=220), "no_task_success"
        )
        guard = EvaluationGuard(patience=2, best_tail=1, min_iterations_before_stall=200)
        self.assertIsNone(guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=20))
        self.assertEqual(
            guard.inspect(0, 0, 0.9, 0, 0, completed_iterations=40), "evaluation_regression"
        )


class CheckpointTests(unittest.TestCase):
    def test_interrupt_preserves_last_assessed_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "best.pt").write_bytes(b"best-assessed")
            (output / "latest.pt").write_bytes(b"last-assessed")
            runner = Mock()
            runner.save.side_effect = lambda path, **kw: Path(path).write_bytes(b"unassessed")
            finalize_checkpoint(runner, output, 20, {}, interrupted=True)
            self.assertEqual((output / "checkpoint.pt").read_bytes(), b"last-assessed")
            self.assertEqual((output / "best.pt").read_bytes(), b"best-assessed")
            self.assertFalse(runner.save.call_args.kwargs["infos"]["evaluated"])
            self.assertEqual((output / "interrupted.pt").read_bytes(), b"unassessed")

    def test_interrupt_before_first_block_retains_evaluated_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "best.pt").write_bytes(b"initial-assessed")
            runner = Mock()
            runner.save.side_effect = lambda path, **kw: Path(path).write_bytes(b"unassessed")
            finalize_checkpoint(runner, output, 0, {}, interrupted=True)
            self.assertEqual((output / "checkpoint.pt").read_bytes(), b"initial-assessed")

    def test_interrupt_during_serialization_keeps_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "latest.pt"
            destination.write_bytes(b"validated")

            def interrupted_write(path, **kwargs):
                Path(path).write_bytes(b"truncated")
                raise KeyboardInterrupt()

            runner = Mock()
            runner.save.side_effect = interrupted_write
            with self.assertRaises(KeyboardInterrupt):
                save_checkpoint_atomically(runner, destination, {})
            self.assertEqual(destination.read_bytes(), b"validated")


if __name__ == "__main__":
    unittest.main()
