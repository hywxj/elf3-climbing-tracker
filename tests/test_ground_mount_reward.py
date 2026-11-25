import unittest
from types import SimpleNamespace

import torch

from elf3_climbing.ground_mount_reward import GroundMountReward


class GroundMountRewardTests(unittest.TestCase):
    def setUp(self):
        n = 5
        feet = torch.zeros(n, 2, 3)
        feet[:, 0, 1] = 0.16
        feet[:, 1, 1] = -0.16
        self.feet = feet
        self.vel = torch.zeros(n, 3)
        self.omega = torch.zeros_like(self.vel)
        self.env = SimpleNamespace(
            num_envs=n,
            step_dt=0.02,
            advance_direction=torch.tensor([1.0, 0.0]),
            cfg=SimpleNamespace(sim=SimpleNamespace(gravity=[0, 0, -10])),
            knee_contacts=SimpleNamespace(total_mass=torch.full((n,), 40.0), floor_z=0.0),
            robot=SimpleNamespace(
                data=SimpleNamespace(root_link_lin_vel_w=self.vel, root_link_ang_vel_w=self.omega)
            ),
            scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
            site_positions=lambda: feet,
            course_time_s=torch.full((n,), 2.0),
        )
        self.reward = GroundMountReward(self.env)
        self.m = {
            "heading_error_deg": torch.zeros(n),
            "root_angular_speed_rad_s": torch.zeros(n),
            "foot_ground_patch_N": torch.full((n, 2), 200.0),
            "foot_ground_slip_m_s": torch.zeros(n, 2),
            "foot_tilt_deg": torch.zeros(n, 2),
            "torso_up": torch.ones(n),
            "forearm_top_load_N": torch.full((n, 2), 30.0),
            "arm_support_top_N": torch.full((n, 2), 30.0),
            "forearm_target_error_m": torch.zeros(n, 2),
            "wrist_platform_force_N": torch.zeros(n, 2),
            "wrist_forbidden": torch.zeros(n, dtype=torch.bool),
            "foot_top_N": torch.full((n, 2), 200.0),
            "foot_bearing": torch.ones(n, 2, dtype=torch.bool),
            "foot_patch_major_std_m": torch.full((n, 2), 0.03),
            "foot_patch_minor_std_m": torch.full((n, 2), 0.01),
            "foot_patch_supported": torch.ones(n, 2, dtype=torch.bool),
            "hip_above_top_m": torch.ones(n),
            "root_speed_m_s": torch.zeros(n),
            "ground_clear": torch.ones(n, dtype=torch.bool),
            "reward": torch.zeros(n),
            "approach_accepted": torch.zeros(n, dtype=torch.bool),
            "entry_accepted": torch.zeros(n, dtype=torch.bool),
            "terminated": torch.zeros(n, dtype=torch.bool),
            "knee_top_N": torch.zeros(n, 2),
            "com_vz_m_s": torch.zeros(n),
            "com_inside_edge_m": torch.full((n,), 0.3),
            "standing": torch.zeros(n, dtype=torch.bool),
            "success": torch.zeros(n, dtype=torch.bool),
        }
        self.stage = torch.zeros(n, dtype=torch.long)

    def measure(self):
        return self.reward.measure(self.m, self.stage)

    def test_slip_and_tilt_are_measured_only_on_loaded_feet(self):
        self.vel[:, 0] = 0.24
        self.m["foot_tilt_deg"][:3, 1] = 50
        self.m["foot_ground_slip_m_s"][:3, 1] = 0.6
        self.m["foot_ground_patch_N"][0] = torch.tensor([400, 0])
        self.m["foot_ground_patch_N"][1] = torch.tensor([0, 400])
        self.m["foot_ground_patch_N"][3] = 0
        self.vel[4, 0] = 0
        m = self.measure()
        torch.testing.assert_close(m["walk_tilt_deg"], torch.tensor([0, 50, 25, 0, 0.0]))
        torch.testing.assert_close(m["walk_slip_m_s"], torch.tensor([0, 0.6, 0.3, 0, 0]))
        self.assertEqual(m["walk_loaded"].tolist(), [True, True, True, False, True])
        self.assertGreater(m["walk_quality"][0], m["walk_quality"][2])
        self.assertGreater(m["walk_quality"][2], m["walk_quality"][1])
        self.assertTrue((m["walk_quality"][3:] == 0).all())

    def test_walking_stability_excludes_bracing_and_frozen_entry(self):
        self.vel[:, 0] = 0.24
        self.m["heading_error_deg"][1] = 40
        self.m["torso_up"][2] = 0.8
        self.m["root_angular_speed_rad_s"][1:3] = 3.0
        self.env.course_time_s[3:] = 4.2
        self.m["torso_up"][3] = 0.3
        self.stage[4] = 1
        m = self.measure()
        self.assertTrue((m["walk_quality"][0] > 5 * m["walk_quality"][1:3]).all())
        self.assertTrue((m["walk_quality"][3:] == 0).all())
        self.assertGreater(m["course_reward"][3], 0)
        self.assertEqual(m["course_reward"][4], 0)
        self.m["entry_accepted"][4] = True
        self.assertGreater(self.measure()["course_reward"][4], 0)

    def test_only_new_planted_footholds_earn_distance_and_reset_is_local(self):
        idle = self.measure()["course_reward"][0].item()
        self.assertLess(idle, 0)
        self.m["foot_ground_patch_N"][0, 1] = 0
        self.feet[0, 1, 0] += 0.2
        self.assertLess(self.measure()["course_reward"][0], 0)
        self.m["foot_ground_patch_N"][0, 1] = 200
        self.assertGreater(self.measure()["course_reward"][0], 0.7)
        self.feet[0, 1, 0] -= 0.15
        self.measure()
        self.feet[0, 1, 0] += 0.15
        self.assertAlmostEqual(self.measure()["course_reward"][0].item(), idle, places=5)
        self.reward.reset(torch.tensor([0]))
        self.feet[:2, 1, 0] += 0.2
        m = self.measure()
        self.assertAlmostEqual(m["course_reward"][0].item(), idle, places=5)
        self.assertGreater(m["course_reward"][1], 0.7)

    def test_brace_progress_accepts_tabletop_wrists_and_rejects_off_face_contact(self):
        self.env.course_time_s.fill_(4.5)
        self.m["forearm_top_load_N"].zero_()
        self.m["forearm_target_error_m"][0] = 0.35
        self.m["forearm_top_load_N"][2, 0] = 12
        self.m["forearm_top_load_N"][3] = 12
        self.m["arm_support_top_N"] = self.m["forearm_top_load_N"].clone()
        # At 0.80 s in the successful 2300 trace both shells were unloaded
        # while the upper faces carried 73.09 / 85.82 N through the wrists.
        self.m["wrist_platform_force_N"][4] = torch.tensor([73.09, 85.82])
        self.m["arm_support_top_N"][4] = self.m["wrist_platform_force_N"][4]
        r = self.measure()["course_reward"]
        self.assertTrue((r[1:4] > r[:3]).all())
        self.assertGreater(r[4], r[2])
        self.assertLess(r[4], r[3])  # shell loading is a small additional gain
        self.m["wrist_forbidden"][4] = True
        self.assertLess(self.measure()["course_reward"][4], 0)

    def test_gradual_improvements_and_an_alternating_weight_bearing_landing(self):
        self.m["foot_ground_slip_m_s"].fill_(0.48)
        self.m["foot_tilt_deg"].fill_(49)
        self.m["heading_error_deg"].fill_(60)
        self.m["torso_up"].fill_(0.25)
        self.m["root_angular_speed_rad_s"].fill_(3)
        self.vel[:, 0] = 0.2
        self.m["foot_tilt_deg"][1] = 25
        self.m["foot_ground_slip_m_s"][2] = 0.25
        self.m["heading_error_deg"][3] = 35
        self.m["torso_up"][4] = 0.88
        m = self.measure()
        self.assertTrue((m["course_reward"][1:] > m["course_reward"][0] + 0.0004).all())
        for side in (0, 1):
            self.m["foot_ground_patch_N"][0, side] = 0
            self.feet[0, side, 2] = 0.05
            for _ in range(3):
                self.assertFalse(self.measure()["walk_step_switch"].any())
            self.feet[0, side, 2] = 0
            self.feet[0, side, 0] += 0.12
            self.m["foot_ground_patch_N"][0, side] = 200
            self.assertEqual(self.measure()["walk_step_switch"][0].item(), side == 1)
        self.assertFalse(self.measure()["walk_step_switch"].any())

    def test_rise_prefers_real_area_on_both_feet_stable_heading_and_width(self):
        self.stage.fill_(2)
        self.m["foot_patch_minor_std_m"][1] = 0
        self.m["foot_patch_supported"][1] = False
        self.m["foot_top_N"][2, 1] = 0
        self.m["foot_bearing"][2, 1] = False
        self.m["heading_error_deg"][3] = 80
        self.feet[4, 1, 1] = -0.5
        self.m["standing"][0] = True
        self.m["foot_top_N"] *= 0
        self.measure()
        self.m["foot_top_N"][:] = 200
        self.m["foot_top_N"][2, 1] = 0
        first = self.measure()
        quality = first["rise_quality"]
        self.assertLess(quality[1], 0.14)
        self.assertLess(0, quality[2])
        self.assertLess(quality[2], quality[0])
        self.assertTrue((quality[3:] < 0.5 * quality[0]).all())
        second = self.measure()
        self.assertGreater(second["course_reward"][0], 0)
        self.assertTrue((second["course_reward"][1:4] == 0).all())
        self.assertLess(second["course_reward"][0], first["course_reward"][0])
        self.m["reward"] += 100  # Legacy point-contact and termination rewards are discarded.
        self.stage[1] = 1
        self.m["success"][:] = True
        third = self.measure()
        self.assertEqual(third["course_reward"][1], 0)
        self.assertAlmostEqual((third["course_reward"] - second["course_reward"])[0].item(), 12)
        self.assertAlmostEqual(self.measure()["course_reward"][0], second["course_reward"][0])
        for values in third.values():
            self.assertEqual(values.shape, (5,))

    def test_loaded_knee_catches_fall_before_soling_and_keeps_signal_through_contact_gap(self):
        self.stage.fill_(2)
        self.m["foot_top_N"].zero_()
        self.m["foot_bearing"].fill_(False)
        self.m["knee_top_N"][:] = torch.tensor([0.0, 60.0])
        self.m["forearm_top_load_N"][:] = 95
        self.measure()
        self.m["knee_top_N"][0, 0] = 70
        self.vel[1, 2] = -1.2
        self.vel[2, 1] = 0.75
        self.omega[3, 0] = 4.5
        self.m["com_vz_m_s"][4] = -1.1
        self.m["forearm_top_load_N"][4] = 0
        self.m["knee_top_N"][4] = 0
        out = self.measure()
        self.assertGreater(out["course_reward"][0], 0.2)
        self.assertTrue((out["course_reward"][1:] < 0).all())
        self.assertGreater(out["rise_quality"][0], out["rise_quality"][4])
        self.assertEqual(self.measure()["course_reward"][0], 0)
        self.stage.fill_(1)
        self.assertTrue((self.measure()["course_reward"] == 0).all())
