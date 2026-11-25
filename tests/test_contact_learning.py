import unittest

import numpy as np
import torch

from elf3_climbing.contact_learning import (
    CONTACT_OBS_DIM,
    ContactActor,
    contact_imitation_loss,
    encode_contact_observation,
    upcoming_contacts,
    plan_label_mask,
)


class ContactLearningTests(unittest.TestCase):
    def test_load_hysteresis_and_physical_knee_touchdown(self):
        # Start with two feet on the ground. Left foot unloads and lands before the right knee.
        f = np.zeros((90, 8))
        f[:, :2] = 150
        f[8:21, 0] = 0
        f[45:, 7] = 40
        f[28, 3] = 14  # Single frame forearm impact is not a supported contact.
        p = np.full((90, 8, 3), np.nan)
        p[f > 0] = [1.0, 2.0, 0.7]
        val, pos, events = upcoming_contacts(f, p, 0.02)
        self.assertEqual(events, [(21, "left_foot"), (45, "right_knee")])
        self.assertTrue(val[17, 0])
        self.assertTrue(val[17, 7])
        self.assertFalse(val[21, 0])
        self.assertTrue(val[22, 7])
        self.assertFalse(val[45:].any())
        self.assertTrue(np.allclose(pos[22, 7], [1.0, 2.0, 0.7]))

    def test_per_body_window_and_repeated_touchdown(self):
        f = np.zeros((121, 8))
        f[20:30, 0] = 120
        f[92:97, 0] = 120
        f[81:90, 7] = 30
        p = np.ones((len(f), 8, 3))
        p[92:, 0, 0] = 2
        mask, coordinates, _ = upcoming_contacts(f, p, 0.02)
        self.assertFalse(mask[5, 7])
        self.assertTrue(mask[6, 7])
        self.assertTrue(mask[6, 0])
        self.assertEqual(coordinates[6, 0, 0], 1)
        self.assertEqual(coordinates[20, 0, 0], 2)
        self.assertFalse(mask[97:].any())

    def test_truncated_and_corrective_rollouts_hide_unknown_negative_labels(self):
        labels = torch.zeros(90, 8, dtype=torch.bool)
        labels[70, 7] = True
        mask = plan_label_mask(labels, 0.02, False)
        self.assertEqual(mask[0, 2], 0)
        self.assertEqual(mask[-1, 2], -1)
        self.assertEqual(mask[70, 7], 1)
        self.assertTrue((plan_label_mask(labels, 0.02, True) == -1).all())

    def test_rotation_no_world_origin_leak_and_planner_gradients(self):
        n = 3
        zeros = lambda *shape: torch.zeros(*shape)
        q = zeros(n, 4)
        q[:, 0] = q[:, 3] = 2**-0.5
        raw = {k: zeros(n, 29) for k in ("joint_pos", "joint_vel", "prev_target")}
        for k in ("root_pos", "root_lin_vel_b", "root_ang_vel_b", "projected_gravity_b"):
            raw[k] = zeros(n, 3)
        raw.update(
            root_quat=q, site_pos=zeros(n, 8, 3), terrain=zeros(n, 126), contact_force=zeros(n, 8)
        )
        raw["site_pos"][:, :, 1] = 1
        obs = encode_contact_observation(raw, zeros(29))
        self.assertEqual(obs.shape, (n, CONTACT_OBS_DIM))
        self.assertTrue(torch.allclose(obs[0, 222:225], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6))
        raw["root_pos"] += 32
        raw["site_pos"] += 32
        self.assertTrue(torch.equal(obs, encode_contact_observation(raw, zeros(29))))
        model = ContactActor()
        mask = zeros(n, 8)
        mask[:, 7] = 1
        loss = contact_imitation_loss(model, (obs, zeros(n, 29), mask, torch.ones(n, 8, 3)))
        loss.backward()
        self.assertGreater(model.planner.weight.grad.norm().item(), 0)
        self.assertEqual(model(obs).shape, (n, 29))


class GaitInitializationTests(unittest.TestCase):
    def test_unified_backbone_is_trainable_and_does_not_require_teacher_at_load(self):
        from elf3_climbing.contact_gait_actor import GaitContactActor

        actor = GaitContactActor()
        restored = GaitContactActor()
        restored.load_state_dict(actor.state_dict())
        obs = torch.randn(4, CONTACT_OBS_DIM)
        predicted, logits, xyz = restored.action_and_plan(obs)
        self.assertTrue(torch.equal(predicted, actor(obs)))
        (predicted.mean() + logits.mean() + xyz.mean()).backward()
        self.assertTrue(all(layer.weight.grad.norm() > 0 for layer in restored.backbone))
        self.assertGreater(restored.context.weight.grad.norm().item(), 0)


class OfflineAnchorTests(unittest.TestCase):
    def test_original_walking_anchor_receives_full_body_landmarks(self):
        from elf3_climbing.contact_anchor import anchor_observation
        from elf3_climbing.contact_learning import _local
        from elf3_climbing.contact_point_actor import POINT_OBS_DIM, PointContactActor, yaw_local

        student, walking = PointContactActor(), ContactActor()
        q = torch.nn.functional.normalize(torch.tensor([[0.71, 0.20, 0.34, 0.51]]), dim=1)
        world = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3) / 8
        obs = torch.zeros(1, POINT_OBS_DIM)
        obs[:, 93:96] = _local(q, torch.tensor([[[0.0, 0.0, -1.0]]]))[:, 0]
        obs[:, 222:246] = yaw_local(q, world).flatten(1)
        old_input = anchor_observation(student, walking, obs)
        self.assertTrue(
            torch.allclose(old_input[:, 222:246], _local(q, world).flatten(1), atol=1e-6)
        )

    def test_releases_near_edge_under_load_or_when_map_invalid(self):
        from elf3_climbing.contact_anchor import far_ground_weight

        obs = torch.zeros(4, CONTACT_OBS_DIM)
        obs[:, 95] = -1
        height = obs[:, 96:159].reshape(-1, 9, 7)
        height[:] = -1.1
        height[:, 7:, :] += 0.7
        obs[:, 159:222] = 1
        height[1, 4:, :] = -0.4
        obs[2, -3] = 1
        obs[3, 159:222] = 0
        self.assertTrue(
            torch.allclose(far_ground_weight(obs), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
        )


class VisibleEdgeTests(unittest.TestCase):
    def test_edge_and_independent_endpoint_visibility(self):
        from elf3_climbing.contact_edge import edge_from_scan, scan_axes

        x, y = scan_axes(torch.zeros(1))
        surface = ((x[:, None] > 0.52) & (y[None, :] > -0.4) & (y[None, :] < 0.4)).float() * 0.7
        heights = surface.expand(3, -1, -1).clone()
        valid = torch.ones_like(heights).bool()
        valid[1, :, y >= 0.35] = False
        valid[2] = False
        e = edge_from_scan(heights, valid)
        self.assertAlmostEqual(e[0, 0].item(), 0.52, delta=0.026)
        self.assertTrue(torch.allclose(e[0, 1:3], torch.tensor([1.0, 0.0]), atol=1e-4))
        self.assertTrue(torch.equal(e[:, 5], torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(e[:, 7], torch.tensor([1.0, 0.0, 0.0])))
        self.assertEqual(e[2].abs().sum().item(), 0.0)

    def test_warm_start_preserves_original_inputs_and_embeds_schema_marker(self):
        from elf3_climbing.contact_edge import EdgeGaitContactActor, extend_weights
        from elf3_climbing.contact_gait_actor import GaitContactActor

        old = GaitContactActor()
        new = EdgeGaitContactActor()
        extend_weights(new, old.state_dict())
        x = torch.randn(3, CONTACT_OBS_DIM)
        observed = torch.cat((x, torch.randn(3, 8)), 1)
        self.assertTrue(torch.allclose(old(x), new(observed), atol=1e-6))
        self.assertIn("edge_spec", new.state_dict())


class PointPlannerTests(unittest.TestCase):
    def test_direct_transfer_preserves_full_body_motor_encoder_and_reloads_frame(self):
        from elf3_climbing.contact_anchor import full_body_landmarks
        from elf3_climbing.contact_edge import extend_weights
        from elf3_climbing.contact_gait_actor import GaitContactActor
        from elf3_climbing.contact_point_actor import POINT_OBS_DIM, PointContactActor

        original = GaitContactActor()
        transferred = PointContactActor(body_frame_input=True)
        extend_weights(transferred, original.state_dict())
        reloaded = PointContactActor()
        reloaded.load_state_dict(transferred.state_dict())
        obs = torch.randn(4, POINT_OBS_DIM)
        expected = original.encoder(full_body_landmarks(obs[:, :254]).clamp(-8, 8))
        self.assertTrue(reloaded.body_frame_input)
        self.assertTrue(torch.allclose(reloaded.point_state(obs), expected, atol=1e-5))

    def test_strict_contact_and_world_to_yaw(self):
        from elf3_climbing.contact_point_actor import sustained_contacts, yaw_local

        f = np.zeros((120, 8))
        p = np.zeros((120, 8, 3))
        f[15:24, 7] = 80
        f[37:49, 7] = 54
        f[60:70, 7] = 80
        valid, _, events = sustained_contacts(f, p, 0.02)
        self.assertEqual(events, [(60, "right_knee")])
        self.assertTrue(valid[1, 7])
        self.assertFalse(valid[60, 7])
        q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
        v = torch.tensor([[[0.0, 1.0, 0.7]]])
        self.assertTrue(
            torch.allclose(yaw_local(q, v), torch.tensor([[[1.0, 0.0, 0.7]]]), atol=1e-6)
        )

    def test_online_history_and_episode_history_agree_at_reset_and_repeated_reads(self):
        from elf3_climbing.contact_point_actor import episode_observation, PointHistory

        n = 8
        base = torch.randn(n, 254)
        raw = {
            "root_pos": torch.zeros(n, 3),
            "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            "site_pos": torch.zeros(n, 8, 3),
        }
        geom = (
            torch.tensor([[0.9, 0.0, 0.35]]),
            torch.tensor([[0.45, 0.5, 0.35]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
        h = PointHistory(1, "cpu")
        samples = []
        for i in range(n):
            current = {k: v[i : i + 1] for k, v in raw.items()}
            sample = h.encode(base[i : i + 1], current, *geom, i)
            self.assertTrue(torch.equal(sample, h.encode(base[i : i + 1], current, *geom, i)))
            samples.append(sample)
        all_obs = torch.cat(samples)
        offline = episode_observation(all_obs[:, :262], all_obs[:, 646:])
        self.assertTrue(torch.equal(all_obs, offline))
        h.reset(torch.tensor([0]))
        first = h.encode(base[:1], {k: v[:1] for k, v in raw.items()}, *geom, n - 1)
        self.assertEqual(first[:, 262:646].abs().sum().item(), 0.0)

    def test_plan_returns_a_visible_cell_and_masks_missing_and_gradients(self):
        from elf3_climbing.contact_point_actor import PointContactActor, N_POINTS, POINT_OBS_DIM

        model = PointContactActor()
        obs = torch.zeros(2, POINT_OBS_DIM)
        obs[:, -N_POINTS:] = -5
        obs[0, -N_POINTS + 191] = -0.6
        with torch.no_grad():
            model.planner.bias.reshape(8, 4)[:, 0] = 3
        model.eval()
        _, logits, plan = model.action_and_plan(obs)
        self.assertTrue((logits[0] > 0).all())
        self.assertTrue((logits[1] < 0).all())
        self.assertFalse(plan[1].any())
        self.assertTrue(
            torch.allclose(
                plan[0, 0], torch.cat((model.point_xy[191], torch.tensor([-0.6]))), atol=1e-6
            )
        )
        model.train()
        obs[0, -N_POINTS + 200] = -0.6
        obs[0, -N_POINTS + 201] = -0.6
        targets = torch.zeros(2, 8, 3)
        targets[0] = torch.cat((model.point_xy[200], torch.tensor([-0.6])))
        labels = torch.full((2, 8), -1.0)
        labels[0] = 1.0
        a, l, p, ce = model.imitation_components(obs, labels, targets, 1.0, None)
        ce.backward()
        self.assertGreater(model.query.bias.grad.abs().sum().item(), 0)
