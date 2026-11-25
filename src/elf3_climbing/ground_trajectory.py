"""Physical reset poses and joint motor commands for the continuous single-ledge course."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class GroundTrajectory:
    """Sample the archived actuated trajectory on the Isaac control clock."""

    # Time of the supported right-knee pose in the ground-to-steady-rise recording.
    recovery_start_s = 7.16

    def __init__(self, path, joint_names, step_dt, approach_steps, device):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            controls = data["ctrl"]
            initial = data["initial_state"]
            source_dt = float(data["dt"])
            if data["joint_names"].tolist() != list(joint_names):
                raise ValueError("Course motor order does not match the robot")
        with np.load(path.with_name("pose_trace.npz"), allow_pickle=False) as trace:
            poses = trace["qpos"]
            if not np.isclose(float(trace["dt"]), source_dt):
                raise ValueError("Course reset poses and commands have different timesteps")
        if (
            not np.isfinite(source_dt)
            or source_dt <= 0
            or controls.ndim != 2
            or controls.shape[1] != 29
            or poses.shape != (len(controls) + 1, 36)
            or initial.shape != (72,)
            or not all(np.isfinite(v).all() for v in (initial, controls, poses))
            or not np.allclose(poses[0], initial[1:37])
            or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Invalid continuous course commands or reset poses")
        stride = round(step_dt / source_dt)
        self.recovery_step = round(self.recovery_start_s / step_dt)
        if (
            stride < 1
            or not np.isclose(stride * source_dt, step_dt)
            or not np.isclose(self.recovery_step * step_dt, self.recovery_start_s)
        ):
            raise ValueError("Course and control timesteps must align")
        rows = np.arange(0, len(controls), stride)
        self.last_step = len(rows) - 1
        self.approach_step = int(approach_steps)
        if not 0 < self.approach_step < self.recovery_step < self.last_step:
            raise ValueError("Course is missing the ground, knee or recovery phase")
        next_pose = poses[rows + 1]
        pose = poses[rows]
        rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
        next_rotation = Rotation.from_quat(next_pose[:, [4, 5, 6, 3]])
        angular = (next_rotation * rotation.inv()).as_rotvec() / source_dt
        velocities = np.concatenate(
            (
                (next_pose[:, :3] - pose[:, :3]) / source_dt,
                angular,
                (next_pose[:, 7:] - pose[:, 7:]) / source_dt,
            ),
            axis=1,
        )
        # MuJoCo free-joint angular velocities are in the body frame. Isaac takes world velocities.
        velocities[0] = initial[37:]
        velocities[0, 3:6] = rotation[0].apply(initial[40:43])
        self.poses, self.velocities, self.controls = (
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in (pose, velocities, controls[rows])
        )

    def seeds(self, ids, counts, training):
        """Full-ground evaluation, plus walking, original knee and recovery samples in training."""
        stage = torch.zeros_like(ids)
        step = torch.zeros_like(ids)
        if training:
            bucket = (ids + counts) % 8
            walking = bucket == 4
            knee = bucket == 5
            recovery = bucket >= 6
            # Draw fresh times inside each stage. Interleaving reset buckets also works for 1 env.
            sampled = torch.rand(ids.shape, device=ids.device)
            step[walking] = ((0.30 + 0.67 * sampled[walking]) * self.approach_step).long()
            stage[knee], step[knee] = 1, self.approach_step
            stage[recovery] = 2
            # Half of the recovery seeds practice knee-to-foot transfer. The other half
            # span rising and the later double-foot stance in which the old soles tip.
            progress = torch.where(bucket == 6, 0.30 * sampled, 0.30 + 0.63 * sampled)
            step[recovery] = (
                self.recovery_step + progress[recovery] * (self.last_step - self.recovery_step)
            ).long()
        return stage, step

    def advance(self, step, stage_before, enter_knee, enter_recovery):
        limit = torch.where(
            stage_before == 0,
            self.approach_step,
            torch.where(stage_before == 1, self.recovery_step, self.last_step),
        )
        following = torch.minimum(step + 1, limit)
        following[enter_knee] = self.approach_step
        following[enter_recovery] = self.recovery_step
        return following


def forearm_gate(metrics, reference_pose, streak):
    pose = metrics["root_pose"]
    ready = (
        # Use exactly the support the frozen knee controller can accept. In
        # tabletop mode either forearm shell or verified top-face wrist counts.
        (metrics["arm_support_top_N"] > 3).all(1)
        & (metrics["foot_ground_patch_N"] > 8).all(1)
        & ~metrics["wrist_forbidden"]
        & ((pose[:, :3] - reference_pose[:3]).norm(dim=1) < 0.20)
        & ((pose[:, 3:7] * reference_pose[3:7]).sum(1).abs() > np.cos(np.deg2rad(35 / 2)))
        & (metrics["root_speed_m_s"] < 0.50)
    )
    streak = torch.where(ready, streak + 1, 0)
    return streak, streak >= 3


def course_heading_error(quat, direction):
    """Absolute angle between torso forward and the ground approach direction about world Z."""
    w, x, y, z = quat.unbind(dim=1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    error = yaw - torch.atan2(direction[1], direction[0])
    return torch.atan2(error.sin(), error.cos()).abs().rad2deg()
