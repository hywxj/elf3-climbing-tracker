import unittest

import torch

from elf3_climbing.contact_tracker_task import (
    ContactMilestones,
    FrontApproach,
    accepted_knee,
    direct_joint_targets,
    ground_reset_offsets,
)


class ContactTrackerTaskTests(unittest.TestCase):
    def test_only_fresh_progress_toward_finite_rotated_front_earns_walking_rewards(self):
        # The long local X axis is rotated 90 degrees. The robot approaches its negative-Y face.
        center = torch.tensor([2., 3., .35])
        task = FrontApproach(4, center, torch.tensor([.8, .55, .35]),
                             torch.tensor([2**-.5, 0., 0., 2**-.5]), torch.tensor([2., 0.]))
        def tick(forward, lateral, first=False):
            xy = torch.stack((torch.tensor(lateral) + 2., torch.tensor(forward) + 1.), 1)
            feet = xy[:, None] + torch.tensor([[-.10, 0.], [.10, 0.]])
            return task.update(feet, torch.full((4,), first), torch.ones(4, dtype=torch.bool))

        self.assertFalse(tick([0.]*4, [0.]*4, True)[1].any())
        gain, switch, progress, valid = tick([.10,.10,.10,.0], [0.,.70,.42,.2])
        torch.testing.assert_close(gain, torch.tensor([.1,0.,0.,0.]))
        self.assertEqual(switch.tolist(), [True, False, False, False])
        # Each pre-brace foot may earn a correctly predicted contact once at this frontier.
        matches = torch.tensor([[.7, .5]]).repeat(4, 1)
        complete = torch.ones(4,2,dtype=torch.bool)
        braced = torch.tensor([False,False,True,False])
        total = torch.full((4,), 2.2)  # Includes one unit of independent arm/knee credit.
        pay = lambda: task.foot_plan_reward(total, matches, complete, braced, progress, valid)
        torch.testing.assert_close(pay(), torch.tensor([2.2,1.,2.2,1.]))
        torch.testing.assert_close(pay(), torch.tensor([1.,1.,2.2,1.]))
        # Retreat/reclaim, returning to the lane, and additional in-place swings are all unpaid.
        tick([.01,.05,.01,.0], [0.,.70,.0,.2])
        gain, switch, _, _ = tick([.10,.10,.10,.0], [0.,0.,0.,.2])
        self.assertLess(gain.abs().max(), 1e-6)
        self.assertFalse(switch.any())
        gain, switch, _, _ = tick([.17,.17,.17,1.4], [0.,0.,0.,0.])
        self.assertEqual(switch.tolist(), [True,True,True,False])
        self.assertEqual(float(gain[-1]), 0.)

    def test_seeded_ground_offsets_preserve_reference_starts_and_follow_route(self):
        ground = torch.arange(600) % 3 > 0
        fwd = torch.tensor([0.6, 0.8])

        def draw(seed):
            return ground_reset_offsets(ground, fwd, torch.Generator().manual_seed(seed))

        offset = draw(42)
        torch.testing.assert_close(offset, draw(42))
        self.assertFalse(torch.equal(offset, draw(73)))
        self.assertEqual(offset[~ground].abs().max(), 0)
        self.assertLessEqual((offset @ fwd).abs().max(), 0.025)
        self.assertLessEqual((offset @ torch.tensor([-0.8, 0.6])).abs().max(), 0.015)

    def test_direct_actions_control_all_joints_with_only_physical_limits(self):
        nominal = torch.linspace(-0.3, 0.3, 29).repeat(2, 1)
        previous = nominal.clone()
        action = torch.full_like(nominal, 0.08)
        action[1] *= -1
        limits = torch.stack((nominal - 1, nominal + 1), dim=-1)
        rates = torch.full((29,), 12.0)
        # The left gripper is also learned and has the same actuator slew limit as the teacher.
        rates[20] = 1.5
        target = direct_joint_targets(action, nominal, previous, limits, rates, 0.02)
        self.assertTrue(torch.allclose(target[:, 20] - nominal[:, 20], torch.tensor([0.03, -0.03])))
        self.assertTrue(torch.allclose(target[:, :20] - nominal[:, :20], action[:, :20]))
        for _ in range(2):
            target = direct_joint_targets(action, nominal, target, limits, rates, 0.02)
        torch.testing.assert_close(target, nominal + action)
        torque_request = torch.full_like(action, 50.0)
        for _ in range(60):
            target = direct_joint_targets(torque_request, nominal, target, limits, rates, 0.02)
        torch.testing.assert_close(target, limits[..., 1])

    def test_right_knee_requires_top_surface_footprint_and_clear_feet_at_once(self):
        n = 9
        m = dict(
            right_knee_supported=torch.ones(n, dtype=torch.bool),
            wrist_forbidden=torch.zeros(n, dtype=torch.bool),
            right_knee_top_load_N=torch.full((n,), 60.0),
            right_knee_off_patch_load_N=torch.zeros(n),
            right_knee_inside_edge_m=torch.full((n,), 0.09),
            min_foot_clearance_m=torch.full((n,), 0.2),
        )
        load, margin = torch.zeros(n, 2), torch.full((n,), 0.08)
        m["right_knee_top_load_N"][1] = 0
        m["right_knee_off_patch_load_N"][2] = 9
        m["right_knee_inside_edge_m"][3] = -0.01
        m["min_foot_clearance_m"][4] = 0
        m["wrist_forbidden"][5] = True
        load[6, 1] = 15
        margin[7] = -0.01
        m["right_knee_top_load_N"][8] = 59.9
        self.assertEqual(
            accepted_knee(m, load, margin, torch.full((n,), 400.0)).tolist(), [True] + [False] * 8
        )

    def test_full_ground_walk_then_push_then_uninterrupted_hold(self):
        # row 0 completes the task, row 1 skips walking, row 2 started at a near-edge sample.
        task = ContactMilestones(3, "cpu", 0.02)
        no = torch.zeros(3, dtype=torch.bool)
        yes = ~no
        ground = torch.tensor([True, True, False])
        advances = torch.full((3,), 0.35)

        def tick(switch=no, brace=no, push=no, air=no, contact=no):
            return task.update(switch, advances, brace, push, air, contact, ground)

        self.assertFalse(tick(air=yes, contact=yes)[0].any())
        tick(switch=torch.tensor([True, False, True]))
        for _ in range(3):
            tick(brace=yes)
        tick(push=yes)
        tick(air=yes)
        for _ in range(9):
            self.assertFalse(tick(contact=yes)[0].any())
        tick()
        for _ in range(9):
            self.assertFalse(tick(contact=yes)[0].any())
        knee, success = tick(contact=yes)
        self.assertTrue(knee.all())
        self.assertEqual(success.tolist(), [True, False, False])
        task.reset(torch.tensor([0]))
        self.assertFalse(task.braced[0])
        self.assertTrue(task.braced[1])
        self.assertEqual(int(task.hold[0]), 0)

    def test_full_credit_requires_a_push_after_the_qualified_approach(self):
        task = ContactMilestones(1, "cpu", 0.02)
        yes = torch.ones(1, dtype=torch.bool)
        no = ~yes
        short, far = torch.tensor([0.20]), torch.tensor([0.30])
        task.update(yes, short, no, no, no, no, yes)
        for _ in range(3):
            task.update(no, short, yes, yes, no, no, yes)
        task.update(no, far, yes, no, no, no, yes)
        for _ in range(10):
            local, full = task.update(no, far, no, no, yes, yes, yes)
        assert local.item() and not full.item()
