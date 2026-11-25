from types import SimpleNamespace

import torch

from elf3_climbing.contact_runner import evaluate


def test_autoreset_lanes_do_not_contribute_to_ground_score(tmp_path):
    class Env:
        device, num_envs, max_episode_length, step_dt = "cpu", 2, 4, 0.02
        knee_contacts = SimpleNamespace(total_mass=torch.tensor([40.0, 40.0]))

        def reset(self, seed):
            self.step_number = 0
            return {"policy": torch.zeros(2, 4)}, {}

        def step(self, actions):
            self.step_number += 1
            flag = torch.tensor([self.step_number > 1, False])
            self.last_metrics = {
                key: flag
                for key in (
                    "walked_to_edge",
                    "bilateral_braced",
                    "bilateral_push",
                    "push_observed",
                    "takeoff_observed",
                    "right_knee_patch_touched",
                    "knee_reached",
                    "success",
                )
            }
            self.last_metrics["ground_start"] = torch.ones(2, dtype=torch.bool)
            self.last_metrics.update(
                {
                    key: flag.float()
                    for key in (
                        "walk_switches",
                        "advance_m",
                        "knee_hold_s",
                        "right_knee_top_N",
                        "right_knee_load_bodyweight",
                    )
                }
            )
            done = torch.tensor([self.step_number == 1, self.step_number == 3])
            return {"policy": actions}, None, done, torch.zeros_like(done), {}

    output = evaluate(Env(), torch.nn.Identity(), 33, tmp_path / "ground.json")
    assert output["counts"]["ground_start"] == 2
    assert output["counts"]["success"] == 0
    assert output["after_walk"] == {"push": 0, "takeoff": 0, "knee_touch_after_takeoff": 0}
    assert output["peak"]["right_knee_top_N"] == 0


def test_plan_reward_commits_before_loading_and_requires_unbroken_support():
    from elf3_climbing.contact_plan_reward import ContactPlanReward

    scorer = ContactPlanReward(2, "cpu", 0.02)
    load = torch.zeros(2, 8)
    centers = torch.zeros(2, 8, 3)
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[:, 7] = True
    for _ in range(3):
        scorer.measure(load, centers)
    planned = centers.clone()
    planned[1, 7, 0] = 0.4
    scorer.propose(valid, planned)
    load[:, 7] = 100
    for _ in range(9):
        scorer.propose(valid, centers)
        assert scorer.measure(load, centers).sum() == 0
    result = scorer.measure(load, centers)
    assert result[0] == 1
    assert result[1] < 1e-5
    assert scorer.measure(load, centers).sum() == 0
    scorer.reset(torch.tensor([0, 1]))
    scorer.propose(valid, centers)
    for _ in range(12):
        assert scorer.measure(load, centers).sum() == 0
