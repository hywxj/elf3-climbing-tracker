import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from elf3_climbing.ground_trajectory import GroundTrajectory, course_heading_error, forearm_gate


class GroundTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trajectory.npz"
        self.names = [f"joint_{i}" for i in range(29)]
        self.poses = np.zeros((2001, 36))
        self.poses[:, 0] = np.arange(2001) * 0.005
        self.poses[:, 3] = 1
        initial = np.r_[0, self.poses[0], np.zeros(35)]
        ctrl = np.broadcast_to(np.arange(2000)[:, None], (2000, 29))
        np.savez(self.path, ctrl=ctrl, initial_state=initial, dt=0.005, joint_names=self.names)
        np.savez(self.path.with_name("pose_trace.npz"), dt=0.005, qpos=self.poses)

    def course(self):
        return GroundTrajectory(self.path, self.names, 0.02, 250, "cpu")

    def test_exact_control_rows_and_zero_velocity_ground_start(self):
        c = self.course()
        self.assertEqual(c.controls[c.recovery_step, 0], 1432)
        self.assertAlmostEqual(c.poses[c.approach_step, 0].item(), 5.0)
        self.assertEqual(c.velocities[0].abs().max(), 0)
        self.assertAlmostEqual(c.velocities[130, 0].item(), 1)

    def test_clock_waits_for_actual_handoff_and_holds_final_command(self):
        c = self.course()
        stages = torch.tensor([0, 0, 1, 1, 2])
        step = torch.tensor([250, 247, 358, 310, 499])
        knee = torch.tensor([False, True, False, False, False])
        recovery = torch.tensor([False, False, False, True, False])
        self.assertEqual(
            c.advance(step, stages, knee, recovery).tolist(), [250, 250, 358, 358, 499]
        )

    def test_curriculum_and_ground_only_evaluation(self):
        c = self.course()
        ids = torch.arange(8)
        for epoch in range(8):
            counts = torch.full_like(ids, epoch)
            stage, steps = c.seeds(ids, counts, False)
            self.assertEqual(steps.max(), 0)
            self.assertEqual(stage.max(), 0)
            stage, steps = c.seeds(ids, counts, True)
            self.assertEqual((steps == 0).sum(), 4)
            self.assertEqual((stage == 0).sum(), 5)
            self.assertEqual(steps[stage == 1].tolist(), [250])
            self.assertEqual((stage == 2).sum(), 2)
            self.assertTrue((steps[stage == 2] >= c.recovery_step).all())

    def test_gate_needs_simultaneous_supported_pose_and_consecutive_steps(self):
        pose = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        m = dict(
            root_pose=pose.repeat(3, 1),
            root_speed_m_s=torch.zeros(3),
            wrist_forbidden=torch.tensor([False, False, True]),
            arm_support_top_N=torch.full((3, 2), 15.0),
            foot_ground_patch_N=torch.full((3, 2), 35.0),
        )
        m["root_pose"][1, 0] = 0.3
        streak = torch.zeros(3, dtype=torch.long)
        for _ in range(2):
            streak, ready = forearm_gate(m, pose, streak)
            self.assertFalse(ready.any())
        m["arm_support_top_N"][0, 1] = 0
        streak, ready = forearm_gate(m, pose, streak)
        self.assertEqual(streak.tolist(), [0, 0, 0])
        m["arm_support_top_N"][0, 1] = 12
        for _ in range(3):
            streak, ready = forearm_gate(m, pose, streak)
        self.assertEqual(ready.tolist(), [True, False, False])

    def test_gate_with_contact_trace_from_successful_knee2300_rollout(self):
        # Measured first three samples of inspect_stopped_knee2300_play_20261122_2215/
        # knee_contacts.csv. Hold the independent pose/speed checks satisfied.
        # Each row is left/right shell, left/right tabletop wrist, left/right sole (N).
        recorded = [
            (0.0, 24.90, 11.96, 0.0, 17.18, 22.28),
            (0.0, 24.39, 13.68, 0.0, 171.30, 177.34),
            (0.0, 36.20, 18.35, 0.0, 153.81, 139.24),
        ]
        pose = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        streak = torch.zeros(3, dtype=torch.long)
        for i, sample in enumerate(recorded):
            shells, wrists, soles = torch.tensor(sample).reshape(3, 2)
            # tabletop allows these top-face wrists, tabletop-touch requires
            # bearing on shells, and rejected side/rim wrist loading blocks both.
            m = dict(
                root_pose=pose.repeat(3, 1),
                root_speed_m_s=torch.zeros(3),
                arm_support_top_N=torch.stack((shells + wrists, shells, shells + wrists)),
                foot_ground_patch_N=soles.repeat(3, 1),
                wrist_forbidden=torch.tensor([False, False, True]),
            )
            streak, ready = forearm_gate(m, pose, streak)
            self.assertEqual(ready.tolist(), [i == 2, False, False])

    def test_mismatched_stream_and_yawed_axis(self):
        self.poses[0, 0] = 99
        np.savez(self.path.with_name("pose_trace.npz"), dt=0.005, qpos=self.poses)
        with self.assertRaisesRegex(ValueError, "Invalid continuous"):
            self.course()
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]])
        err = course_heading_error(q, torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(err[0].item(), 90, places=3)
        self.assertAlmostEqual(err[1].item(), 0, places=3)


class OfficialGaitTests(unittest.TestCase):
    def test_batched_actor_matches_vendor_onnx_and_rejects_swapped_joints(self):
        import importlib.util
        from elf3_climbing.ground_locomotion import GAIT_MODEL, GroundLocomotion

        if not GAIT_MODEL.is_file() or not all(
            importlib.util.find_spec(module) for module in ("onnx", "onnxruntime")
        ):
            self.skipTest("Official gait model or ONNX reference runtime unavailable")
        import onnxruntime as ort

        session = ort.InferenceSession(str(GAIT_MODEL), providers=["CPUExecutionProvider"])
        names = session.get_modelmeta().custom_metadata_map["joint_names"].split(",")
        actor = GroundLocomotion(names, "cpu")
        obs = np.random.default_rng(42).normal(size=(4, 96)).astype(np.float32)
        reference = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs])
        np.testing.assert_allclose(actor.actor(torch.from_numpy(obs)).numpy(), reference, atol=2e-5)
        with self.assertRaisesRegex(ValueError, "29 course joints"):
            GroundLocomotion(names[::-1], "cpu")
