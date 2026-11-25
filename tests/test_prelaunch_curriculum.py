"""Exercise the physical reset selection without opening an Isaac simulation."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import torch

from elf3_climbing import contact_learning


def load_curriculum():
    path = Path(contact_learning.__file__).with_name("contact_tracker_env.py")
    spec = importlib.util.spec_from_file_location("elf3_climbing._cpu_curriculum", path)
    module = importlib.util.module_from_spec(spec)
    direct = type("Direct", (), {"_reset_idx": lambda *_: None})
    math = SimpleNamespace(quat_apply=lambda _, x: x, quat_apply_inverse=None, yaw_quat=None)
    substitutes = {
        "isaaclab.envs": SimpleNamespace(DirectRLEnv=direct),
        "isaaclab.utils.math": math,
        "elf3_climbing.environment": SimpleNamespace(TrackerEnv=object),
    }
    with patch.dict(sys.modules, substitutes):
        spec.loader.exec_module(module)
    return module


def test_real_feet_and_verified_sequence_are_needed():
    module = load_curriculum()
    n = 22
    f = np.zeros((n, 8))
    f[:, :2] = 25
    z = np.zeros((n, 8, 3))
    d = dict(
        episode_id=np.ones(n, int),
        step=np.arange(n),
        student_contact_force=f,
        student_contact_point=z,
        terrain_box_centers_w=np.array([[0, 0, -0.1], [0, 0, 0.5]]),
        terrain_box_half_sizes=np.array([[2, 2, 0.1], [1, 1, 0.2]]),
        contact_observed=np.ones(n, bool),
        post_brace=np.arange(n) == 2,
        post_takeoff=np.arange(n) >= 13,
        post_bilateral_push=np.arange(n) == 11,
        post_knee_hold_steps=(np.arange(n) == 21) * 10,
        post_walk_switches=np.ones(n, int),
        post_foot_advance_m=np.full(n, 0.28),
        step_dt=0.02,
    )
    z[5, 0, 2] = 0.70
    f[6, 1] = 0
    f[7, 7] = 10
    assert module.grounded_prelaunch_rows(d, {1})[1].tolist() == [3, 4, 8, 9, 10, 11, 12, 13]
    assert not module.grounded_prelaunch_rows(d, {3})
    d["post_bilateral_push"][:] = False
    assert not module.grounded_prelaunch_rows(d, {1})


def test_split_matches_real_replay_and_excludes_cloning_holdout(tmp_path):
    module = load_curriculum()
    main, other = tmp_path / "complete.npz", tmp_path / "other.npz"

    def batches(path, edge=False, point=False):
        source = "complete" if path == main else "other"
        groups = []
        for i in range(10 if path == main else 17):
            q = torch.full((2, 1), float(i) if path == main else -5.0)
            groups.append(
                (
                    source,
                    str(i),
                    (q, q, q, q),
                    {"source": source, "episode": i, "reset_at_start": False},
                )
            )
        return ("a",), np.zeros(1), groups

    with patch.object(contact_learning, "_episode_batches", batches):
        pools = ([main, other], [other, main])
        raw = module.training_start_episodes(*pools)
        actual = [
            set(contact_learning.load_examples(p)["train"][0].flatten().tolist()) for p in pools
        ]
        assert raw[main] == (actual[0] & actual[1]) - {-5}
        assert 0 < len(raw[main]) < 8


def test_reset_restores_one_physical_row_and_only_ground_can_earn_full_credit():
    module = load_curriculum()
    env = module.ContactTrackerEnv.__new__(module.ContactTrackerEnv)
    n = 16
    env.device = "cpu"
    ids = torch.arange(n)
    pose = torch.zeros(n, 7)
    pose[:, 3] = 1
    env.ref = {
        "root_pos": pose[:, :3],
        "root_quat": pose[:, 3:],
        "root_ang_vel": torch.zeros(n, 3),
        "root_lin_vel": torch.zeros(n, 3),
        "joint_pos": torch.zeros(n, 29),
        "joint_vel": torch.zeros(n, 29),
    }
    env.nominal = torch.zeros(n, 29)
    env.joint_ids = list(range(29))
    env.robot = Mock()
    env.robot._ALL_INDICES = ids
    env.robot.data = SimpleNamespace(
        default_joint_pos=env.nominal, body_com_pos_b=torch.zeros(n, 1, 3)
    )
    env.scene = SimpleNamespace(env_origins=torch.zeros(n, 3))
    env.ref_steps = ids.clone().zero_()
    env.starts = ids.clone()
    env.reset_count = ids.clone().zero_()
    env.ground_start = torch.zeros(n, dtype=torch.bool)
    env.prelaunch_start = env.ground_start.clone()
    env.first_sample = env.ground_start.clone()
    env.ground_reset_offset_xy = torch.zeros(n, 2)
    env.advance_direction = torch.tensor([1.0, 0.0])
    env.progress_best = torch.zeros(n, 6)
    env.targets = torch.zeros(n, 29)
    env.actions, env.previous_actions = env.targets.clone(), env.targets.clone()
    env.knee_contacts, env.milestones, env.footsteps, env.plan_score = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    env.walk_start_frame, env.arm_approach_frame, env.segment_start = 1, 2, 4
    env.knee_contacts.push_start = 7
    env.diagnostic_start = None

    def bank(value):
        q = torch.full((1, 29), float(value))
        return dict(
            root_pose=torch.tensor([[value * 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            joint_pos=q,
            joint_vel=q + 1,
            prev_target=q + 2,
            root_lin_vel_w=q[:, :3] + 3,
            root_ang_vel_w=q[:, :3] + 4,
            frame=torch.tensor([-1]),
        )

    env.demonstration_starts, env.prelaunch_starts = [bank(3)], [bank(9)]
    env.stage_training = True
    env._reset_idx(ids)
    assert int(env.ground_start.sum()) == 8
    assert int(env.prelaunch_start.sum()) == 4
    chosen = env.prelaunch_start
    q, qd = env.robot.write_joint_state_to_sim.call_args.args
    velocity = env.robot.write_root_velocity_to_sim.call_args.args[0]
    assert torch.all(q[chosen] == 9) and torch.all(qd[chosen] == 10)
    assert torch.all(env.targets[chosen] == 11) and torch.all(velocity[chosen, :3] == 12)
    assert torch.all(velocity[chosen, 3:] == 13)
    assert not (chosen & env.ground_start).any()
    env.prelaunch_starts = []
    env._reset_idx(ids)
    assert env.ground_start.sum() == 8 and env.prelaunch_start.sum() == 0
    assert torch.all(env.targets[~env.ground_start] == 5)
    env.stage_training = False
    env._reset_idx(ids)
    assert env.ground_start.all() and not env.prelaunch_start.any()
    env.diagnostic_start = bank(7)
    env._reset_idx(ids)
    assert not env.ground_start.any() and not env.prelaunch_start.any()
    assert torch.all(env.targets == 9)
