from types import SimpleNamespace
import math

import numpy as np
import torch
import unittest

from elf3_climbing.isaac_knee_entry import KneeContacts
from elf3_climbing.mount_support import SupportReward
from elf3_climbing.mount_task import MountTask


class MountTaskTests(unittest.TestCase):
    def setUp(self):
        n = 2
        ori = torch.tensor([[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]])
        # Approach along +Y, with a yawed platform, in two translated clones.
        center = np.array([[0.0, 1.0, 0.35], [0.0, 0.0, 0.0]])
        size = np.array([[0.8, 0.4, 0.35], [2.0, 2.0, 0.0]])
        quat = np.array([[2**-0.5, 0.0, 0.0, 2**-0.5], [1.0, 0.0, 0.0, 0.0]])
        sites = ori[:, None] + torch.tensor([0.0, 0.55, 0.70]).expand(n, 8, 3)
        bodies = ori[:, None] + torch.tensor([0.0, 0.55, 1.54]).expand(n, 3, 3)
        data = SimpleNamespace(
            body_com_pos_w=bodies.clone(),
            body_link_pos_w=bodies.clone(),
            body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 3, 1),
            root_link_pos_w=bodies[:, 0],
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4),
            root_link_lin_vel_w=torch.zeros(n, 3),
            root_link_ang_vel_w=torch.zeros(n, 3),
        )
        self.env = env = SimpleNamespace(
            starts=torch.zeros(n, dtype=torch.long),
            num_envs=n,
            step_dt=0.02,
            scene=SimpleNamespace(env_origins=ori),
            robot=SimpleNamespace(data=data),
            cfg=SimpleNamespace(sim=SimpleNamespace(gravity=[0.0, 0.0, -9.81], dt=0.005)),
            site_positions=lambda: sites,
            site_ids=[1, 2],
            segment_start=0,
            reference_np=dict(
                terrain_box_centers_w=center,
                terrain_box_half_sizes=size,
                terrain_box_quat_wxyz=quat,
                root_pos=np.array([[0.0, -0.2, 1.0]]),
            ),
            ref=dict(
                terrain_box_centers_w=torch.tensor(center).float(),
                terrain_box_half_sizes=torch.tensor(size).float(),
                terrain_box_quat_wxyz=torch.tensor(quat).float(),
            ),
        )
        self.sites = sites
        self.loads = {k: torch.zeros(n) for k in ("lf", "rf", "lk", "rk")}
        self.floor = torch.zeros(n, 2)
        env.knee_contacts = k = SimpleNamespace(
            env=env,
            upper=0,
            lower=1,
            top_heights=ori[:, 2] + 0.7,
            floor_heights=ori[:, 2],
            hip_ids=[1, 2],
            total_mass=torch.full((n,), 38.0),
            mass_fraction=torch.full((n, 3), 1 / 3),
            foot_names=("lf", "rf"),
            knee_names=("lk", "rk"),
            surface_loads=lambda name, pivot=None: (self.loads[name], torch.zeros(n)),
            loads=lambda names, box: self.floor,
        )
        k.enable_pose_trace = lambda: KneeContacts.enable_pose_trace(k)
        self.patch_offsets = torch.tensor(
            [[-0.06, -0.025, 0.0], [-0.06, 0.025, 0.0], [0.06, -0.025, 0.0], [0.06, 0.025, 0.0]]
        ).repeat(n, 2, 1, 1)
        self.patch_fractions = torch.full((n, 2, 4), 0.25)

        def sensor(side, name):
            def contacts(dt):
                points = (sites[:, side, None] + self.patch_offsets[:, side]).reshape(-1, 3)
                normals = torch.zeros_like(points)
                normals[:, 2] = 1
                weights = self.patch_fractions[:, side] * self.loads[name][:, None]
                return (
                    weights.reshape(-1, 1),
                    points,
                    normals,
                    None,
                    torch.tensor([[4, 0], [4, 0]]),
                    torch.tensor([[0, 0], [4, 0]]),
                )

            return SimpleNamespace(contact_physx_view=SimpleNamespace(get_contact_data=contacts))

        env.support_sensors = {n: sensor(i, n) for i, n in enumerate(k.foot_names)}
        self.mount = MountTask(env)

    def test_flat_stance_rejects_edge_support_and_one_tilted_foot(self):
        self.loads["lf"].fill_(185)
        self.loads["rf"].fill_(185)
        # Strong dispersed forces along a diagonal still provide a line, not an area.
        self.patch_offsets[0, 0, :, 1] = self.patch_offsets[0, 0, :, 0] * 0.5
        angle = math.radians(21) / 2
        self.env.robot.data.body_link_quat_w[1, 2] = torch.tensor(
            [math.cos(angle), math.sin(angle), 0, 0]
        )
        for _ in range(50):
            metrics = self.mount.measure()
        self.assertTrue(metrics["foot_bearing"].all())
        self.assertFalse(metrics["standing"].any())
        self.assertFalse(metrics["foot_patch_supported"][0, 0])
        self.assertTrue(metrics["foot_patch_supported"][1, 1])
        self.assertAlmostEqual(metrics["foot_tilt_deg"][1, 1].item(), 21, places=4)
        self.assertEqual(metrics["stand_hold_s"].tolist(), [0.0, 0.0])
        self.mount.stance = "legacy"
        self.mount.reset(torch.arange(2))
        for _ in range(50):
            metrics = self.mount.measure()
        self.assertTrue(metrics["success"].all())

    def test_support_reward_requires_loaded_feet_and_upwards_torso(self):
        self.env.mount_active = torch.ones(2, dtype=torch.bool)
        shaped = SupportReward(self.mount)
        self.loads["lf"].fill_(185)
        self.loads["rf"].fill_(185)
        self.env.robot.data.root_link_quat_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        for _ in range(4):
            m = shaped.measure()
        self.assertGreater(shaped.best[0].item(), 0.9)
        self.assertEqual(shaped.best[1].item(), 0.0)
        self.assertGreater(m["reward"][0].item(), 0.0)
        self.assertEqual(m["reward"][1].item(), 0.0)
        shaped.reset(torch.tensor([0]))
        self.env.mount_active[0] = False
        for _ in range(4):
            shaped.measure()
        self.assertEqual(shaped.best[0].item(), 0.0)

    def test_stand_hold_restarts_after_concentrated_sole_load_or_ground_contact(self):
        self.loads["lf"].fill_(183)
        self.loads["rf"].fill_(184)
        for cause in ("concentrated_sole_load", "ground_contact"):
            with self.subTest(cause=cause):
                self.mount.reset(torch.arange(2))
                for _ in range(49):
                    self.assertFalse(self.mount.measure()["success"].any())
                if cause == "concentrated_sole_load":
                    self.patch_fractions[1, 1] = torch.tensor([0.9997, 0.0001, 0.0001, 0.0001])
                else:
                    self.floor[1, 0] = 12
                m = self.mount.measure()
                self.assertTrue(m["foot_bearing"].all())
                self.assertEqual(m["success"].tolist(), [True, False])
                torch.testing.assert_close(m["stand_hold_s"], torch.tensor([1.0, 0.0]))
                torch.testing.assert_close(m["foot_inside_edge_m"], torch.full((2, 2), 0.35))
                self.patch_fractions[1, 1] = 0.25
                self.floor.zero_()
                self.mount.reset(torch.tensor([0]))
                m = self.mount.measure()
                self.assertEqual(m["reward"][0], 0.0)
                torch.testing.assert_close(m["stand_hold_s"], torch.full((2,), 0.02))

    def test_any_leg_can_step_first_and_stalls_do_not_earn_progress(self):
        self.loads["rf"][0] = 185
        self.loads["lf"][1] = 185
        self.assertEqual(self.mount.measure()["reward"].tolist(), [0.0, 0.0])
        self.mount.measure()
        m = self.mount.measure()
        self.assertEqual(m["stage"].tolist(), [3, 3])
        self.assertEqual(m["foot_bearing"].tolist(), [[False, True], [True, False]])
        self.assertEqual(self.mount.measure()["reward"].tolist(), [0.0, 0.0])
        self.loads["lf"].zero_()
        self.loads["rf"].zero_()
        self.mount.measure()
        self.loads["rf"][0] = self.loads["lf"][1] = 185
        for i in range(4):
            self.assertEqual(self.mount.measure()["reward"].tolist(), [0.0, 0.0])

    def test_false_standing_from_rim_or_light_contact(self):
        self.loads["lf"].fill_(30)
        self.loads["rf"].fill_(30)
        self.assertFalse(self.mount.measure()["standing"].any())
        self.loads["lf"].fill_(185)
        self.loads["rf"].fill_(185)
        # Positive insertion from the front edge, but outside the side of the box.
        self.sites[0, 0, 0] = 0.8
        self.env.robot.data.body_link_pos_w[1, 2, 2] = 1.1
        m = self.mount.measure()
        self.assertTrue((m["foot_inside_edge_m"] > 0).all())
        self.assertFalse(m["standing"].any())
