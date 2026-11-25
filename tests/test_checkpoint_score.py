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

    def test_knee_hold_beats_a_stationary_timeout(self):
        reached, parked = result(0.0, speed=0.2), result(0.0, speed=0.0)
        for batch, hold in ((reached, 0.14), (parked, 0.0)):
            for evaluation in batch.values():
                evaluation["required_hold_s"] = 0.2
                evaluation["episodes"][0].update(right_knee_supported_max_s=hold, takeoff_observed=False)
        self.assertGreater(score(reached), score(parked))

    def test_knee_policy_covering_an_earlier_start_beats_a_tail_only_policy(self):
        policies = []
        for phase, rate in [(0.9, 1.0), (0.8, 0.95)]:
            results = result(0.0)
            for evaluation in results.values():
                evaluation["required_hold_s"] = 0.2
                evaluation["episodes"][0].update(right_knee_supported_max_s=0.2, takeoff_observed=False)
            results["tail"] = dict(results["tail"], success_fraction=1.0)
            results["current"] = dict(results["tail"], start_phase=phase, success_fraction=rate)
            policies.append(results)
        self.assertGreater(score(policies[1]), score(policies[0]))
        policies[1]["tail"]["success_fraction"] = 0.0
        self.assertLess(score(policies[1]), score(policies[0]))

    def test_balance_keeps_velocity_selection(self):
        self.assertGreater(score(result(1.3, 0.0)), score(result(0.1, 0.05)))
