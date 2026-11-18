import unittest
from elf3_climbing.training import score


def result(error, speed=0.02, failed=False, success=0.0, hold=0.0):
    entry = {
        "success_fraction": success,
        "episodes": [
            {
                "max_stable_hold_s": hold,
                "terminated": failed,
                "terminal_gripper_error_rad": error,
                "terminal_root_velocity_error_m_s": speed,
                "terminal_root_angular_velocity_error_rad_s": 0.0,
            }
        ],
    }
    return {"from_start": entry, "tail": entry}


class CheckpointSelection(unittest.TestCase):
    def test_flip_progress_beats_stationarity_when_both_survive(self):
        self.assertGreater(score(result(0.1, 0.05), True), score(result(1.3, 0.0), True))

    def test_angle_during_failure_does_not_beat_survival(self):
        self.assertLess(score(result(0.0, failed=True), True), score(result(1.3), True))

    def test_completion_and_hold_remain_primary(self):
        self.assertGreater(score(result(0.04, success=1, hold=5), True), score(result(0.0), True))

    def test_balance_keeps_velocity_selection(self):
        self.assertGreater(score(result(1.3, 0.0)), score(result(0.1, 0.05)))
