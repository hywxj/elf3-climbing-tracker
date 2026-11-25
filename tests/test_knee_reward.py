from types import SimpleNamespace as Box
import unittest

import torch

from elf3_climbing.knee_reward import KneeEntryReward


def make_env():
    count = 3
    zeros = torch.zeros(count)
    flags = torch.zeros(count, dtype=torch.bool)
    sites = torch.zeros(4, 8, 3)
    sites[:2, 7, 0] = -0.5
    m = dict(
        forearm_top_load_N=torch.full((count, 2), 20.0),
        forearm_target_error_m=torch.zeros(count, 2),
        right_knee_top_load_N=zeros.clone(),
        foot_ground_load_N=torch.zeros(count, 2),
        max_foot_load_N=zeros.clone(),
        min_foot_clearance_m=zeros.clone() + 0.05,
        right_knee_goal_distance_m=zeros.clone(),
        wrist_loaded=flags.clone(),
        terminated=flags.clone(),
        expired=flags.clone(),
        success=flags.clone(),
        gripper_ready=~flags,
        stable_hold_steps=zeros.clone(),
        takeoff_observed=flags.clone(),
        right_knee_supported=flags.clone(),
        root_velocity_error_m_s=zeros.clone(),
        root_angular_velocity_error_rad_s=zeros.clone(),
    )
    return Box(
        starts=torch.full((count,), 3),
        segment_end=3,
        step_dt=0.02,
        required_hold_steps=10,
        ref=dict(site_pos=sites, root_lin_vel=torch.zeros(4, 3), root_pos=torch.zeros(4, 3)),
        last_metrics=m,
        gate_allowed=~flags,
        knee_contacts=Box(takeoff=2, flight=~flags),
        actions=torch.zeros(count, 29),
        previous_actions=torch.zeros(count, 29),
        robot=Box(
            data=Box(
                root_link_lin_vel_w=torch.zeros(count, 3), root_link_pos_w=torch.zeros(count, 3)
            )
        ),
        scene=Box(env_origins=torch.zeros(count, 3)),
        cfg=Box(reward_stage="prior", knee_wrist_contact="full", knee_support_rule="continuous"),
        outcome_counts=torch.zeros(2),
        curriculum=Box(phase=0.9),
        device="cpu",
        extras={},
    )


class KneeRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.task = KneeEntryReward(self.env)
        self.ids = torch.arange(3)
        self.task.reset(self.ids)
        self.zero = torch.zeros(3)

    def step(self, frame=3):
        return self.task.compute(torch.full((3,), frame), self.zero, self.zero, self.zero)

    def test_idle_and_wrist_support_cannot_outscore_a_settled_hold(self):
        m = self.env.last_metrics
        m["right_knee_supported"][:] = True
        m["right_knee_supported"][0] = False
        m["max_foot_load_N"][0] = 80
        m["wrist_loaded"][2] = True
        total = torch.zeros(3)
        for n in range(1, 11):
            m["stable_hold_steps"][1] = n
            total += self.step()
        self.assertLess(total[0], 0)
        self.assertGreater(total[1], 4.9)
        self.assertLess(total[2], 0)
        m["right_knee_supported"][:] = False
        self.step()
        m["right_knee_supported"][:] = True
        for _ in range(10):
            self.assertLessEqual(self.step()[1], 0)  # regaining the same hold pays no second bonus
        self.task.reset(torch.tensor([1]))
        self.assertGreater(self.step()[1], 0)

    def test_phased_reward_prefers_planted_push_and_allows_airborne_arm_release(self):
        e = self.env
        m = e.last_metrics
        e.cfg.knee_support_rule = "push-off"
        e.knee_contacts.push_start = 1
        m.update(foot_ground_patch_N=torch.full((3, 2), 60.), foot_ground_slip_m_s=torch.zeros(3, 2),
                 bilateral_push=torch.zeros(3, dtype=torch.bool),
                 weaker_foot_bodyweight=torch.full((3,), 0.16), com_vz_m_s=torch.full((3,), 0.3),
                 forearm_assist_recent=torch.ones(3, dtype=torch.bool),
                 takeoff_assumed=torch.zeros(3, dtype=torch.bool))
        self.task = KneeEntryReward(e)
        self.task.reset(self.ids)
        e.knee_contacts.flight[:] = False
        m["forearm_top_load_N"][:] = 0
        m["foot_ground_slip_m_s"][1, 0] = 0.4
        m["com_vz_m_s"][2] = 0
        ground = self.step(1)
        self.assertGreater(ground[0], ground[2])
        self.assertGreater(ground[2], ground[1])
        m["foot_ground_slip_m_s"][1, 0] = 2.4
        self.assertLess(self.step(1)[1], ground[1])
        e.knee_contacts.flight[:] = True
        m["takeoff_observed"][:] = True
        m["foot_ground_patch_N"].zero_()
        m["forearm_top_load_N"][2] = 30
        # Unloaded feet can move sideways and airborne arms can unload freely.
        air = self.step(2)
        torch.testing.assert_close(air, air[0].expand_as(air))
        self.assertTrue((air > 0).all())

    def test_assisted_push_rewards_current_support_and_releases_after_takeoff(self):
        e, m = self.env, self.env.last_metrics
        e.cfg.knee_support_rule = "assisted-push"
        e.knee_contacts.push_start = 1
        e.knee_contacts.flight[:] = False
        flags = torch.zeros(3, dtype=torch.bool)
        m.update(foot_ground_patch_N=torch.full((3, 2), 60.), foot_ground_slip_m_s=torch.zeros(3, 2),
                 bilateral_push=flags.clone(), push_release_window=flags.clone(),
                 simultaneous_support=~flags, support_streak=torch.full((3,), 3),
                 knee_bearing=flags.clone(), transfer_lost=flags.clone(), push_started=flags.clone(),
                 arm_carry_fraction=torch.ones(3),
                 weaker_foot_bodyweight=torch.full((3,), 0.16), com_vz_m_s=torch.full((3,), 0.3),
                 hip_vz_m_s=torch.full((3,), 0.3), hip_forward_m_s=torch.full((3,), 0.25),
                 takeoff_assumed=flags.clone())
        self.task = KneeEntryReward(e)
        self.task.reset(self.ids)
        m["forearm_top_load_N"][1, 0] = 0
        m["foot_ground_patch_N"][2, 1] = 0
        m["weaker_foot_bodyweight"][2] = 0
        self.step(1)  # acquire once
        reward = self.step(1)
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[0], reward[2])
        m["hip_forward_m_s"][0] = -0.25
        self.assertLess(self.step(1)[0], reward[0])
        # While one foot is still on the ground after a measured push, unloading it
        # with arm support earns more than keeping it loaded or losing the arms.
        m["push_started"][:] = True
        m["hip_forward_m_s"][:] = 0.25
        m["arm_carry_fraction"][:] = torch.tensor([1., 1., 0.])
        m["max_foot_load_N"][:] = torch.tensor([20., 120., 20.])
        self.step(2)
        terms = e.extras["log"]
        self.assertGreater(terms["KneeReward/assisted_release"], 0)
        self.assertLess(terms["KneeReward/unsupported_transfer"], 0)
        m["arm_carry_fraction"][:] = 1
        m["max_foot_load_N"].zero_()
        m["takeoff_observed"][:] = True
        e.knee_contacts.flight[:] = True
        m["foot_ground_patch_N"].zero_()
        air = self.step(2)
        torch.testing.assert_close(air, air[0].expand_as(air))

    def test_fast_motion_during_knee_contact_earns_less_than_settling(self):
        m = self.env.last_metrics
        m["right_knee_supported"][:] = True
        m["root_angular_velocity_error_rad_s"][0] = 2.0
        reward = self.step()
        self.assertGreater(reward[1], reward[0])
        m["success"][1] = True
        self.assertGreater(self.step()[1], 8)

    def test_pre_endpoint_steps_do_not_fill_the_stable_hold(self):
        self.env.last_metrics["right_knee_supported"][:] = True
        for _ in range(10):
            self.step(2)
        self.step(3)
        self.assertEqual(self.task.endpoint_hold.tolist(), [1, 1, 1])
        self.assertTrue((self.task.best_settled_hold <= 0.1).all())

    def test_only_observed_release_earns_one_takeoff_bonus(self):
        step1 = self.step(2)
        self.env.last_metrics["takeoff_observed"][1] = True
        self.env.last_metrics["min_foot_clearance_m"][1] = 0
        self.step(2)
        self.assertFalse(self.task.credited_takeoff[1])
        self.env.last_metrics["min_foot_clearance_m"][1] = 0.05
        step2 = self.step(2)
        self.assertAlmostEqual(float(step2[1] - step1[1]), 1.0)
        self.assertAlmostEqual(float(self.step(2)[1]), float(step1[1]))

    def test_ground_phase_tracks_preload_while_feet_and_forearms_are_loaded(self):
        m = self.env.last_metrics
        m["foot_ground_load_N"][0, :] = 40
        m["foot_ground_load_N"][1, :] = 40
        self.env.robot.data.root_link_pos_w[1, 2] = 0.10
        reward = self.step(1)
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[1], reward[2])
