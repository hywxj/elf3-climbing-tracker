import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from elf3_climbing.training_control import (
    EvaluationGuard,
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
