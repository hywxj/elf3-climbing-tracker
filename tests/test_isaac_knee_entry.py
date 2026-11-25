from types import SimpleNamespace
import unittest

import numpy as np
import torch

from elf3_climbing.isaac_knee_entry import KneeContacts, foot_patch_motion, tabletop_loads
from elf3_climbing.reference import SITE_BODIES


class KneeContactTests(unittest.TestCase):
    def make_task(self, rule="continuous"):
        sites = torch.zeros((2, 8, 3))
        sites[:, :2, 2] = 0.15
        mask = torch.zeros((1, 8), dtype=torch.bool)
        mask[:, 7] = True
        names = [*SITE_BODIES, "l_wrist_y_link", "r_wrist_y_link", "l_wrist_z_link", "r_wrist_z_link"]
        forces = {n: torch.zeros((2, 1, 2, 3)) for n in names}
        sensors = {
            n: SimpleNamespace(data=SimpleNamespace(force_matrix_w=f)) for n, f in forces.items()
        }
        env = SimpleNamespace(
            segment_start=0,
            segment_end=0,
            step_dt=0.02,
            cfg=SimpleNamespace(knee_wrist_contact="full", knee_support_rule=rule),
            starts=torch.zeros(2, dtype=torch.long),
            ref_steps=torch.zeros(2, dtype=torch.long),
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
            robot=SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(2, 3),
                default_mass=torch.full((2, 1), 38.), body_com_lin_vel_w=torch.zeros(2, 1, 3))),
            reference_np={
                "contact_mask": mask.numpy(),
                "site_bodies": np.array(SITE_BODIES),
                "root_pos": np.zeros((1, 3)),
                "terrain_box_centers_w": np.array([[0, 0, 0.35], [0, 0, 0]]),
                "terrain_box_half_sizes": np.array([[1, 1, 0.35], [2, 2, 0]]),
            },
            ref={"site_pos": sites[:1], "contact_mask": mask},
            site_positions=lambda: sites,
            support_sensors=sensors,
        )
        env.advance_direction = torch.tensor([1.0, 0.0])
        env.robot.find_bodies = lambda names, preserve_order: ([0, 1], names)
        env.robot.data.body_link_lin_vel_w = torch.zeros(2, 2, 3)
        task = KneeContacts(env)
        # This test supplies top-face loads directly. The patch filter is exercised below.
        task.surface_loads = lambda name, knee_pivots=None: (
            forces[name][:, 0, 0, 2],
            torch.zeros(2),
        )
        task.ground_motion = lambda: (torch.stack([forces[n][:, 0, 1, 2] for n in SITE_BODIES[:2]], 1),
                                      torch.zeros(2, 2), torch.zeros(2, 2))
        task.reset(torch.arange(2))
        return task, env, sites, forces

    def test_playback_geometry_uses_the_rotated_approach_edge_and_clone_origin(self):
        task, env, sites, _ = self.make_task()
        # Approach along world +Y, facing the box's rotated local -X face.
        env.reference_np["root_pos"][0, 1] = -2
        env.reference_np["terrain_box_quat_wxyz"] = np.array(
            [[2**-0.5, 0, 0, 2**-0.5], [1, 0, 0, 0]])
        env.scene.env_origins[:, 1] = torch.tensor([1.0, 10.0])
        sites[:, 7, 1] = env.scene.env_origins[:, 1] + torch.tensor([-0.8, -1.1])
        env.robot.data.body_link_pos_w = torch.zeros(2, 2, 3)
        env.robot.data.body_link_pos_w[:, :, 2] = torch.tensor([0.75, 0.65])
        task.enable_pose_trace()
        task.measure(env.starts, sites[:, :2])
        torch.testing.assert_close(task.metrics["right_knee_inside_edge_m"], torch.tensor([0.2, -0.1]))
        torch.testing.assert_close(task.metrics["hip_height_above_platform_m"], torch.zeros(2))

    def test_playback_summary_keeps_wrist_tabletop_load_separate_and_counts_contact_loss(self):
        from elf3_climbing.playback import knee_load_summary

        sample = dict(reference_frame=4, both_feet_airborne=False, takeoff_assumed=False,
                      left_forearm_N=0., right_forearm_N=12., right_knee_top_N=0.,
                      left_foot_patch_N=15., right_foot_patch_N=15., left_wrist_tabletop_N=15.,
                      right_wrist_tabletop_N=0., hip_height_above_platform_m=-0.1,
                      right_knee_inside_edge_m=-0.15)
        rows = [sample, dict(sample, reference_frame=5),
                dict(sample, reference_frame=6, both_feet_airborne=True),
                dict(sample, reference_frame=7, both_feet_airborne=True, right_knee_top_N=150.,
                     right_knee_inside_edge_m=0.1, hip_height_above_platform_m=0.2),
                dict(sample, reference_frame=8)]
        summary = knee_load_summary(rows, 5, 0.02, 300)
        self.assertEqual(summary["peak_right_knee_inside_edge_m"], 0.1)
        self.assertAlmostEqual(summary["max_hip_rise_from_prior_low_m"], 0.3)
        phases = summary["phases"]
        self.assertEqual([stage["samples"] for stage in phases.values()], [1, 1, 1, 2])
        self.assertEqual(phases["transfer"]["both_forearms_above_3N_fraction"], 0.)
        self.assertEqual(phases["transfer"]["mean_load_N"]["left_wrist_tabletop_N"], 15.)
        self.assertEqual(phases["knee"]["mean_right_knee_body_weight_fraction"], 0.25)

    def test_brief_touch_recovers_and_sustained_load_fails(self):
        task, env, sites, forces = self.make_task()
        for n in [*SITE_BODIES[2:4], SITE_BODIES[7]]:
            forces[n][:, 0, 0, 2] = 20
        forces["r_wrist_z_link"][0, 0, 0, 2] = 8
        forces["l_wrist_z_link"][1, 0, 0, 2] = 8
        for step in range(1, 6):
            supported, failed = task.measure(env.starts, sites[:, :2])
            self.assertEqual(supported.tolist(), [step > 1, False])
            self.assertEqual(failed.tolist(), [False, step == 5])
            forces["r_wrist_z_link"][0, 0, 0, 2] = 0
        task.reset(torch.tensor([1]))
        supported, failed = task.measure(env.starts, sites[:, :2])
        self.assertEqual(supported.tolist(), [True, False])
        self.assertFalse(failed.any())

    def test_wrist_y_support_is_visible_in_legacy_playback_and_rejected_in_new_runs(self):
        task, env, sites, forces = self.make_task()
        for body in [*SITE_BODIES[2:4], SITE_BODIES[7]]:
            forces[body][:, 0, 0, 2] = 20
        forces["l_wrist_y_link"][0, 0, 0, 2] = 8
        forces["r_wrist_y_link"][1, 0, 0, 0] = 8
        for _ in range(5):
            supported, failed = task.measure(env.starts, sites[:, :2])
        self.assertFalse(supported.any())
        self.assertTrue(failed.all())
        env.cfg.knee_wrist_contact = "distal-only"
        task.reset(torch.arange(2))
        supported, failed = task.measure(env.starts, sites[:, :2])
        self.assertTrue(supported.all())
        self.assertFalse(failed.any())
        self.assertTrue((task.metrics["wrist_y_platform_force_N"].amax(1) == 8).all())

    def test_release_requires_bilateral_clearance_and_a_recent_supported_push(self):
        task, env, sites, forces = self.make_task()
        task.takeoff = 1
        task.reset(torch.arange(2))
        env.robot.data.root_link_lin_vel_w[:, 2] = 0.1
        sites[:, :2, 2] = 0
        for body in SITE_BODIES[2:4]:
            forces[body][:, 0, 0, 2] = 20
        for body in SITE_BODIES[:2]:
            forces[body][:, 0, 1, 2] = 20
        task.measure(env.starts, sites[:, :2])
        for body in SITE_BODIES[:2]:
            forces[body][:, 0, 1, 2] = 0
        env.ref_steps += 1
        task.measure(env.starts, sites[:, :2])
        self.assertFalse(task.observed.any())
        sites[0, :2, 2] = 0.02
        sites[1, 0, 2] = 0.02
        env.ref_steps += 1
        task.measure(env.starts, sites[:, :2])
        self.assertEqual(task.observed.tolist(), [True, False])
        env.ref_steps += 10
        sites[1, 1, 2] = 0.02
        task.measure(env.starts, sites[:, :2])
        self.assertEqual(task.observed.tolist(), [True, False])

    def test_push_allows_arm_release_but_requires_assistance_and_planted_feet(self):
        task, env, sites, forces = self.make_task()
        env.cfg.knee_support_rule = "push-off"
        task.takeoff = 10
        slip = torch.zeros(2, 2)
        task.ground_motion = lambda: (
            torch.stack([forces[n][:, 0, 1, 2] for n in SITE_BODIES[:2]], 1), slip, slip)
        for mode in ("clean", "slip", "late_slip", "no_assist", "no_upward_velocity"):
            task.reset(torch.arange(2))
            env.ref_steps.zero_()
            slip.zero_()
            sites[:, :2, 2] = 0
            forces[SITE_BODIES[7]].zero_()
            for n in SITE_BODIES[2:4]:
                forces[n][:, 0, 0, 2] = 0 if mode == "no_assist" else 20
            for n in SITE_BODIES[:2]:
                forces[n][:, 0, 1, 2] = 120
            env.robot.data.body_com_lin_vel_w.zero_()
            task.measure(env.starts, sites[:, :2])
            # Both arms can unload between setup and the propulsive foot contact.
            for n in SITE_BODIES[2:4]:
                forces[n].zero_()
            if mode == "slip":
                slip[:, 0] = 0.5
            if mode != "no_upward_velocity":
                env.robot.data.body_com_lin_vel_w[:, :, 2] = 0.23
            env.ref_steps += 1
            task.measure(env.starts, sites[:, :2])
            if mode == "late_slip":
                slip[:, 1] = 0.4
                env.ref_steps += 1
                task.measure(env.starts, sites[:, :2])
            for n in SITE_BODIES[:2]:
                forces[n].zero_()
            env.ref_steps += 1
            sites[:, :2, 2] = 0.03
            task.measure(env.starts, sites[:, :2])
            self.assertEqual(task.observed.all().item(), mode == "clean", mode)
            forces[SITE_BODIES[7]][:, 0, 0, 2] = 30
            supported, _ = task.measure(env.starts, sites[:, :2])
            self.assertEqual(supported.all().item(), mode == "clean", mode)

    def test_assisted_push_requires_concurrent_support_and_forward_upward_hips(self):
        task, env, sites, forces = self.make_task("assisted-push")
        task.takeoff = 10
        for mode in ("valid", "arms_unloaded", "one_foot", "hips_back", "hips_down", "stale"):
            task.reset(torch.arange(2))
            env.ref_steps.zero_()
            sites[:, :2, 2] = 0
            for name in SITE_BODIES[2:4]:
                forces[name][:, 0, 0, 2] = 20
            for name in SITE_BODIES[:2]:
                forces[name][:, 0, 1, 2] = 100
            env.robot.data.body_link_lin_vel_w.zero_()
            env.robot.data.body_com_lin_vel_w.zero_()
            task.measure(env.starts, sites[:, :2])
            if mode == "arms_unloaded":
                forces[SITE_BODIES[2]].zero_()
            if mode == "one_foot":
                forces[SITE_BODIES[1]].zero_()
            v = env.robot.data.body_link_lin_vel_w
            v[:, :, 0] = -0.2 if mode == "hips_back" else 0.2
            v[:, :, 2] = -0.2 if mode == "hips_down" else 0.2
            env.robot.data.body_com_lin_vel_w[:, :, 2] = 0.2
            env.ref_steps += 1
            task.measure(env.starts, sites[:, :2])
            self.assertEqual(bool(task.metrics["bilateral_push"].all()), mode in ("valid", "stale"), mode)
            for name in SITE_BODIES[:4]:
                forces[name].zero_()
            env.ref_steps += 4 if mode == "stale" else 1
            sites[:, :2, 2] = 0.04
            task.measure(env.starts, sites[:, :2])
            self.assertEqual(bool(task.observed.all()), mode == "valid", mode)

    def test_tabletop_wrist_support_bridges_release_until_knee_is_bearing(self):
        task, env, sites, forces = self.make_task("assisted-push")
        env.cfg.knee_wrist_contact = "tabletop"
        task.takeoff = 0
        task.reset(torch.arange(2))
        for side in ("l", "r"):
            forces[side + "_wrist_y_link"][0, 0, 0, 2] = 25
        for _ in range(5):
            task.measure(env.starts, sites[:, :2])
        self.assertEqual(task.transfer_lost.tolist(), [False, True])
        self.assertTrue(task.metrics["wrist_loaded"][0])
        torch.testing.assert_close(task.metrics["wrist_tabletop_load_N"][0], torch.tensor([25., 25.]))
        self.assertFalse(task.metrics["wrist_support_failure"].any())
        forces[SITE_BODIES[7]][:, 0, 0, 2] = 30
        supported, _ = task.measure(env.starts, sites[:, :2])
        self.assertFalse(supported.any())  # require a second physical knee load
        for n in ("l_wrist_y_link", "r_wrist_y_link"):
            forces[n].zero_()
        supported, _ = task.measure(env.starts, sites[:, :2])
        self.assertEqual(supported.tolist(), [True, False])
        task.reset(torch.tensor([1]))
        self.assertFalse(task.transfer_lost.any())

    def test_tabletop_touch_requires_both_forearms_to_push_lift_and_carry(self):
        task, env, sites, forces = self.make_task("assisted-push")
        env.cfg.knee_wrist_contact = "tabletop-touch"
        task.takeoff = 10
        task.reset(torch.arange(2))
        for name in ("l_wrist_y_link", "r_wrist_z_link"):
            forces[name][:, 0, 0, 2] = 100
        for name in SITE_BODIES[:2]:
            forces[name][:, 0, 1, 2] = 100
        forces[SITE_BODIES[2]][:, 0, 0, 2] = 25
        forces[SITE_BODIES[3]][0, 0, 0, 2] = 25
        env.robot.data.body_link_lin_vel_w[:, :, (0, 2)] = 0.2
        env.robot.data.body_com_lin_vel_w[:, :, 2] = 0.2
        for _ in range(2):
            env.ref_steps += 1
            task.measure(env.starts, sites[:, :2])
        torch.testing.assert_close(task.metrics["arm_support_top_N"], torch.tensor([[25., 25.], [25., 0.]]))
        torch.testing.assert_close(task.metrics["wrist_tabletop_load_N"], torch.full((2, 2), 100.))
        self.assertFalse(task.metrics["wrist_forbidden"].any())
        self.assertEqual(task.metrics["bilateral_push"].tolist(), [True, False])
        for name in SITE_BODIES[:2]:
            forces[name].zero_()
        env.ref_steps += 1
        task.measure(env.starts, sites[:, :2])
        self.assertEqual(task.observed.tolist(), [True, False])
        forces[SITE_BODIES[3]].zero_()
        for _ in range(4):
            env.ref_steps += 1
            task.measure(env.starts, sites[:, :2])
        self.assertEqual(task.metrics["arm_carry_fraction"].tolist(), [0., 0.])
        self.assertEqual(task.transfer_lost.tolist(), [True, False])

    def test_tabletop_touch_latches_sustained_ground_reload_before_landing(self):
        task, env, sites, forces = self.make_task("assisted-push")
        env.cfg.knee_wrist_contact = "tabletop-touch"
        task.transfer_guide = True
        task.takeoff = 10
        task.reset(torch.arange(2))
        for name in SITE_BODIES[2:4]:
            forces[name][:, 0, 0, 2] = 25
        for name in SITE_BODIES[:2]:
            forces[name][:, 0, 1, 2] = 100
        env.robot.data.body_link_lin_vel_w[:, :, (0, 2)] = 0.2
        env.robot.data.body_com_lin_vel_w[:, :, 2] = 0.2
        def step():
            env.ref_steps += 1
            return task.measure(env.starts, sites[:, :2])[0]
        step()
        step()
        for name in SITE_BODIES[:2]:
            forces[name].zero_()
        step()
        self.assertTrue(task.observed.all())
        foot = forces[SITE_BODIES[0]][:, 0, 1, 2]
        foot[:] = 20
        step()
        self.assertFalse(task.transfer_lost.any())
        foot[0] = 0
        step()
        self.assertEqual(task.transfer_lost.tolist(), [False, True])
        foot.zero_()
        forces[SITE_BODIES[7]][:, 0, 0, 2] = 30
        step()
        supported = step()
        self.assertEqual(supported.tolist(), [True, False])
        task.reset(torch.tensor([1]))
        self.assertEqual(task.reground_steps.tolist(), [0, 0])
        self.assertFalse(task.transfer_lost.any())

    def test_transfer_can_recover_a_brief_ground_brush_after_measured_lift(self):
        self.check_lift_recovery(guided=True)
        self.check_lift_recovery(guided=False)

    def check_lift_recovery(self, guided):
        task, env, sites, forces = self.make_task("assisted-push")
        task.takeoff = 10
        task.transfer_guide = guided

        def measure():
            return task.measure(env.starts, sites[:, :2], recover_supported_lift=not guided)

        for invalid in ("late", "no_arm"):
            task.reset(torch.arange(2))
            env.ref_steps.zero_()
            sites[:, :2, 2] = 0
            for n in SITE_BODIES[2:4]:
                forces[n][:, 0, 0, 2] = 25
            for n in SITE_BODIES[:2]:
                forces[n][:, 0, 1, 2] = 100
            env.robot.data.body_link_lin_vel_w[:, :, (0, 2)] = 0.2
            env.robot.data.body_com_lin_vel_w[:, :, 2] = 0.2
            measure()
            env.ref_steps += 1
            measure()
            for n in SITE_BODIES[:2]:
                forces[n].zero_()
            sites[:, :2, 2] = 0.05
            env.ref_steps += 1
            measure()
            self.assertTrue(task.flight.all())

            forces[SITE_BODIES[0]][:, 0, 1, 2] = 15
            env.ref_steps += 3
            measure()
            self.assertFalse(task.flight.any())
            forces[SITE_BODIES[0]].zero_()
            env.ref_steps += 1
            if invalid == "late":
                env.ref_steps[1] += 8
            else:
                forces[SITE_BODIES[3]][1].zero_()
            measure()
            self.assertEqual(task.flight.tolist(), [True, False])
            self.assertTrue(task.observed.all())

    def test_foot_slip_uses_contact_point_velocity_and_top_normal_load(self):
        # For a rolling foot the contact point can be stationary while its COM moves.
        heights = torch.zeros(3)
        centers = torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
        linear = torch.tensor([[0., 1., 0.], [0., 0.4, 0.], [0., 1., 0.]])
        angular = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        data = (torch.tensor([30., 40., 9.]), torch.zeros(3, 3),
                torch.tensor([[0.,0.,1.], [0.,0.,1.], [1.,0.,0.]]), None,
                torch.ones(3, 1, dtype=torch.int), torch.arange(3)[:,None])
        loads, speed, outward = foot_patch_motion(data, 0, heights, centers, linear, angular,
                                                  torch.tensor([0., -1.]))
        torch.testing.assert_close(loads, torch.tensor([30., 40., 0.]))
        torch.testing.assert_close(speed, torch.tensor([0., 0.4, 0.]))
        torch.testing.assert_close(outward, torch.tensor([0., -0.4, 0.]))

    def test_only_proximal_top_contacts_contribute_force(self):
        heights = torch.tensor([0.7, 1.0, 1.2])
        pivots = torch.tensor([[0.0, 0.0, 0.75], [4.0, 0.0, 1.05], [8.0, 0.0, 1.25]])
        points = torch.tensor(
            [[0.0, 0.0, 0.7], [0.2, 0.0, 0.7], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [4.0, 0.0, 1.0]]
        )
        normals = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        data = (
            torch.tensor([9.0, 30.0, 7.0, 12.0, 15.0]),
            points,
            normals,
            None,
            torch.tensor([[4, 0], [1, 0], [0, 0]]),
            torch.tensor([[0, 0], [4, 0], [0, 0]]),
        )
        top, other = tabletop_loads(data, 0, heights, pivots)
        torch.testing.assert_close(top, torch.tensor([9.0, 15.0, 0.0]))
        torch.testing.assert_close(other, torch.tensor([49.0, 0.0, 0.0]))
        self.assertEqual(float(tabletop_loads(data, 1, heights, pivots)[0].sum()), 0.0)
