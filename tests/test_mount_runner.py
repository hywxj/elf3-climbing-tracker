"""Regression tests for first-episode scoring and reproducible controller overrides."""

from types import SimpleNamespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from elf3_climbing.mount_runner import (
    GroundAssessment,
    PAIR_FIELDS,
    SCALAR_FIELDS,
    assess,
    checkpoint_score,
    controller_columns,
    controller_digest,
    normalize_contract,
    set_controller_parameters,
    validate_mount_contract,
)


class TestMountAssessment(unittest.TestCase):
    def test_ground_assessment_excludes_airborne_and_finished_clones_from_slip(self):
        env = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(2, 3))),
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
            advance_direction=torch.tensor([1.0, 0.0]),
        )
        score = GroundAssessment(env)
        m = dict(
            course_stage=torch.tensor([0, 0]),
            course_time_s=torch.full((2,), 2.0),
            approach_accepted=torch.tensor([False, False]),
            walk_step_switch=torch.tensor([True, False]),
            root_pose=torch.tensor([[0.4, 0.2], [0.0, -0.2]]),
            walk_active=torch.tensor([True, True]),
            walk_loaded=torch.tensor([True, False]),
            walk_quality=torch.tensor([0.4, 0.0]),
            walk_slip_m_s=torch.tensor([0.3, 0.0]),
            walk_tilt_deg=torch.tensor([15.0, 0.0]),
            rise_quality=torch.zeros(2),
            rise_active=torch.tensor([False, False]),
        )
        score.update(m, torch.tensor([True, True]))
        m.update(course_stage=torch.tensor([1, 2]), walk_active=torch.tensor([False, True]))
        score.update(m, torch.tensor([True, False]))
        s = score.summary()
        self.assertEqual(s["ground_braced"], 1)
        self.assertEqual(s["ground_with_alternating_steps"], 1)
        self.assertEqual(s["ground_mean_walk_advance_m"], 0)
        self.assertEqual(s["ground_walk_loaded_samples"], 1)
        self.assertAlmostEqual(s["ground_walk_loaded_slip_m_s"], 0.3)
        self.assertAlmostEqual(s["ground_mean_approach_advance_m"], 0.2)
        self.assertIsNone(s["ground_rise_quality"])
        s.update(standing_successes=0, mean_peak_stand_hold_s=0.0, entry_accepted=0)
        self.assertGreater(checkpoint_score(dict(s, standing_successes=1)), checkpoint_score(s))

    def test_finished_clone_cannot_contribute_success_or_support_from_a_later_episode(self):
        def zeros():
            return torch.zeros(2, dtype=torch.bool)

        obs = {"policy": torch.zeros(2, 3)}
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            max_episode_length=3,
            reset=lambda: (obs, {}),
        )
        steps = iter(range(3))

        def step(_):
            t = next(steps)
            fields = {key: torch.zeros(2, 2) for key in PAIR_FIELDS}
            fields.update({key: torch.zeros(2) for key in SCALAR_FIELDS})
            fields.update(
                mount_active=torch.tensor([t > 0, True]),
                entry_accepted=torch.tensor([t == 1, t == 0]),
                success=torch.tensor([t > 0, False]),
                terminated=zeros(),
                foot_bearing=torch.tensor([[t > 0, t > 0], [t == 2, False]]),
                knee_bearing=torch.tensor([[t > 0, t > 0], [False, True]]),
                reference_frame=torch.tensor([100, 200]) + t,
                root_pose=torch.full((2, 7), float(t)),
                joint_pos=torch.full((2, 29), float(t)),
            )
            # Stage three indicates one planted foot, independent of bilateral kneeling.
            fields["stage"] = torch.tensor([5 if t else 0, 3 if t == 2 else 1])
            env.last_metrics = fields
            return None, zeros().float(), torch.tensor([True, t == 2]), {}

        wrapped = SimpleNamespace(env=env, step=step)
        runner = SimpleNamespace(
            eval_mode=lambda: None,
            obs_normalizer=lambda x: x,
            alg=SimpleNamespace(policy=SimpleNamespace(act_inference=lambda x: x)),
        )
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = assess(runner, wrapped, directory, record_env=1)
            frames = [
                json.loads(row) for row in (directory / "trace.jsonl").read_text().splitlines()
            ]
        self.assertEqual([row["reference_frame"] for row in frames], [200, 201, 202])
        self.assertEqual([row["environment"] for row in frames], [1, 1, 1])
        s = report["summary"]
        self.assertEqual(s["entry_accepted"], 1)
        self.assertEqual(s["standing_successes"], 0)
        self.assertEqual(s["any_foot_bearing"], 1)
        self.assertEqual(s["both_feet_bearing"], 0)
        self.assertEqual(s["both_knees_bearing"], 0)
        self.assertIsNone(report["episodes"][0]["post_peak"])
        self.assertEqual(report["episodes"][1]["terminal"]["stage"], 3)

    def test_named_control_parameters_are_applied_to_all_clones(self):
        control = SimpleNamespace(
            parameter_names=("tempo", "forward_shift_m"), parameters=torch.ones(3, 2)
        )
        set_controller_parameters(control, {"forward_shift_m": -0.03})
        torch.testing.assert_close(control.parameters[:, 1], torch.full((3,), -0.03))
        torch.testing.assert_close(control.parameters[:, 0], torch.ones(3))
        for invalid in ({"speed": 2}, {"tempo": 0}, {"tempo": float("nan")}, {"tempo": True}):
            with self.assertRaises(ValueError):
                set_controller_parameters(control, invalid)

    def test_knee_transfer_and_recovery_parameters_replay_each_clone(self):
        control = SimpleNamespace(
            parameter_names=("tempo",),
            parameters=torch.ones(2, 1),
            transfer_parameter_names=("swing_duration_s",),
            transfer_parameters=torch.zeros(2, 1),
        )
        settings = {"names": ["tempo", "swing_duration_s"], "values": [[0.8, 1.2], [1.1, 0.6]]}
        set_controller_parameters(control, settings)
        self.assertEqual(list(controller_columns(control)), settings["names"])
        torch.testing.assert_close(control.parameters[:, 0], torch.tensor([0.8, 1.1]))
        torch.testing.assert_close(control.transfer_parameters[:, 0], torch.tensor([1.2, 0.6]))
        for malformed in (
            dict(settings, names=["swing_duration_s", "tempo"]),
            dict(settings, values=[[1.0, 1.0]]),
            dict(settings, values=[[1.0, 1.0], [2.0]]),
            dict(settings, values=[[True, 1.0], [1.0, 2.0]]),
            dict(settings, values=[[1.0, 1.0], [2.0, 0.0]]),
        ):
            with self.assertRaises(ValueError):
                set_controller_parameters(control, malformed)

    def test_only_audited_historical_standard_hash_migrates(self):
        from elf3_climbing.mount_control import MountControl

        contract = {
            "schema": "mount_residual_v1",
            "controller_sha256": "cbf919fc35b04523e5e268676be667bd4e70055d4f8d4ca1d2c14d9ade50ccb6",
            "mount_env_sha256": "unchanged",
        }
        restored = normalize_contract(contract)
        self.assertEqual(restored["controller"], "standard")
        self.assertEqual(restored["mount_env_sha256"], contract["mount_env_sha256"])
        self.assertEqual(
            restored["controller_sha256"], controller_digest(MountControl.__new__(MountControl))
        )
        unknown = dict(contract, controller_sha256="modified")
        self.assertEqual(normalize_contract(unknown), unknown)

    def test_flat_foot_opt_in_preserves_all_other_checkpoint_contract_fields(self):
        import elf3_climbing.mount_task as mount_task
        from elf3_climbing.reference import digest

        old = dict(
            schema="mount_residual_v2",
            mount_task_sha256="02d8552360a2c640847d7129d4600ea91e2aac5b58e4944fc310380085abcac3",
            entry_checkpoint_sha256="same frozen entry",
            control_parameters={"tempo": 1.2},
        )
        legacy = normalize_contract(old)
        expected = dict(old, stance="legacy", mount_task_sha256=digest(Path(mount_task.__file__)))
        self.assertEqual(legacy, expected)
        validate_mount_contract(legacy, expected, None)
        current = dict(expected, stance="flat-foot")
        validate_mount_contract(legacy, current, "flat-foot")
        with self.assertRaises(ValueError):
            validate_mount_contract(legacy, current, None)
        for changed in (
            dict(old, mount_task_sha256="unknown"),
            dict(legacy, entry_checkpoint_sha256="different"),
            dict(legacy, control_parameters={"tempo": 0.2}),
        ):
            with self.assertRaises(ValueError):
                validate_mount_contract(normalize_contract(changed), current, "flat-foot")


if __name__ == "__main__":
    unittest.main()
