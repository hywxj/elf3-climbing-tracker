"""Offline motor teacher for real four-limb support states reached by walking to the ledge."""

import numpy as np
import torch

from .ground_mount import GroundMountEnv
from .contact_tracker_task import ground_reset_offsets


def entry_profile(path):
    """Average physically settled prelaunch commands across successful teacher episodes."""
    with np.load(path, allow_pickle=False) as d:
        if str(d["source_kind"]) != "isaac_knee2300":
            raise ValueError("The entry profile requires previously accepted knee demonstrations")
        mask = (d["step"] >= 3) & (d["step"] <= 7)
        if not mask.any() or not np.all(d["contact_observed"][mask]):
            raise ValueError("Missing measured support states at 0.06 to 0.14 seconds")
        return {
            k: np.median(d[k][mask], axis=0)
            for k in ("joint_pos", "drive_target", "root_pos", "root_quat")
        }, d["joint_names"].tolist()


class BridgeTeacher(GroundMountEnv):
    """Training-data generation only. Every row starts on the ground and integrates continuously."""

    def attach_profile(self, path, settings):
        ref, names = entry_profile(path)
        if names != self.reference_np["joint_names"].tolist():
            raise ValueError("Motor profile and robot have different joints")
        self.bridge_profile = {k: self.actions.new_tensor(a) for k, a in ref.items()}
        self.bridge_profile["root_quat"] /= self.bridge_profile["root_quat"].norm()
        self.bridge_settings = self.actions.new_tensor([row[:4] for row in settings])
        self.walk_knee_boost = self.actions.new_tensor(
            [(row[6] if len(row) > 6 else 0.0, row[4] if len(row) > 4 else 0.0) for row in settings]
        )
        self.bridge_settle_fraction = self.actions.new_tensor(
            [row[5] if len(row) > 5 else 0.0 for row in settings]
        )
        self.swing_tick = self.starts.new_zeros((self.num_envs, 2))
        self.walk_pitch_joints = [
            [names.index(f"{side}_{part}_y_joint") for part in ("hip", "knee", "ankle")]
            for side in ("l", "r")
        ]
        self.bridge_active = self.gate_latched.clone().zero_()
        self.bridge_started = self.bridge_active.clone()
        self.bridge_age = torch.zeros_like(self.starts)
        self.bridge_good_streak = torch.zeros_like(self.starts)
        self.bridge_target0 = torch.zeros_like(self.targets)
        self.bridge_initial_cost = self.actions.new_full((self.num_envs,), float("inf"))
        self.bridge_right_knee = names.index("r_knee_y_joint")
        self.bridge_left_knee = names.index("l_knee_y_joint")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "bridge_active"):
            ids = self.robot._ALL_INDICES if env_ids is None else env_ids
            # All teacher roll-ins start on the ground with the same longitudinal/lateral
            # uncertainty as formal student evaluations. Only reset writes the root pose.
            root = self.course.poses[self.course_step[ids], :7].clone()
            root[:, :3] += self.scene.env_origins[ids]
            root[:, :2] += ground_reset_offsets(self.ground_stage[ids] == 0, self.advance_direction)
            self.robot.write_root_pose_to_sim(root, env_ids=ids)
            self.bridge_active[ids] = self.bridge_started[ids] = False
            self.bridge_age[ids] = self.bridge_good_streak[ids] = self.swing_tick[ids] = 0
            self.bridge_initial_cost[ids] = float("inf")

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.applied_drive_target = self.targets.clone()

    def _joint_command(self):
        command = super()._joint_command()
        if self.last_metrics and "foot_ground_patch_N" in self.last_metrics:
            floor = self.last_metrics["foot_ground_patch_N"]
            walking = (
                (self.ground_stage == 0) & (self.course_time_s > 1.5) & (self.course_time_s < 3.74)
            )
            released = walking[:, None] & (floor < 12) & (floor.flip([1]) > 40)
            self.swing_tick.copy_(torch.where(released, self.swing_tick + 1, 0))
            # The right foot in the recorded ground-to-knee positive advanced 11–14cm in the
            # air but cleared only 17–19mm. Shorten the unloaded leg during early swing.
            age = self.swing_tick * self.step_dt
            window = (age / 0.04).clamp(0, 1) * ((0.18 - age) / 0.06).clamp(0, 1)
            lift = self.walk_knee_boost * window
            for side, (hip, knee, ankle) in enumerate(self.walk_pitch_joints):
                command[:, hip] -= 0.5 * lift[:, side]
                command[:, knee] += lift[:, side]
                command[:, ankle] -= 0.5 * lift[:, side]
        if not self.bridge_active.any():
            return command
        profile, d = self.bridge_profile, self.robot.data
        ramp_s, qgain, xy_gain, load_gain = self.bridge_settings.unbind(1)
        t = ((self.bridge_age + 1) * self.step_dt / ramp_s).clamp(0, 1)
        blend = (t.square() * (3 - 2 * t))[:, None]
        q = d.joint_pos[:, self.joint_ids]
        qd = d.joint_vel[:, self.joint_ids]
        # The native knee's absolute motor targets include the static spring compression
        # of its loaded legs and arms. Interpolate the commands from the last real drive.
        servo = profile["drive_target"] + qgain[:, None] * (
            profile["joint_pos"] - q - 0.035 * qd
        ).clamp(-0.55, 0.55)
        root = d.root_link_pos_w - self.scene.env_origins
        err = root[:, :2] - profile["root_pos"][:2]
        vel = d.root_link_lin_vel_w[:, :2]
        body_pd = xy_gain[:, None] * (3.2 * err + 0.85 * vel)
        forward = self.advance_direction
        left = torch.stack((-forward[1], forward[0]))
        pitch = (body_pd @ forward).clamp(-0.48, 0.48)
        roll = -(body_pd @ left).clamp(-0.33, 0.33)
        for ankle_pitch, ankle_roll in ((7, 8), (13, 14)):
            servo[:, ankle_pitch] += pitch
            servo[:, ankle_roll] += roll
        feet = self.last_metrics["foot_ground_patch_N"].clamp_min(0)
        left_share = feet[:, 0] / feet.sum(1).clamp_min(1)
        transfer = load_gain * (left_share - 0.52).clamp(-0.35, 0.35) * (feet.sum(1) > 35)
        servo[:, self.bridge_left_knee] += transfer
        servo[:, self.bridge_right_knee] -= transfer
        servo[:, self.wrist_joint] = self.fixed_wrist_target
        bridged = torch.lerp(self.bridge_target0, servo, blend)
        return torch.where(self.bridge_active[:, None], bridged, command)

    def _bridge_metrics(self, m):
        ref = self.bridge_profile
        rp = m["root_pose"]
        pos = (rp[:, :3] - ref["root_pos"]).norm(dim=1)
        angle = 2 * (rp[:, 3:] * ref["root_quat"]).sum(1).abs().clamp(0, 1).acos().rad2deg()
        jrmse = (m["joint_pos"] - ref["joint_pos"]).square().mean(1).sqrt()
        speed = m["root_speed_m_s"]
        feet = m["foot_ground_patch_N"].clamp_min(0)
        right_share = feet[:, 1] / feet.sum(1).clamp_min(1)
        cost = (
            (pos / 0.10).square()
            + (angle / 20).square()
            + (jrmse / 0.22).square()
            + (speed / 0.25).square()
        ).sqrt()
        true_support = (
            (m["arm_support_top_N"] > 3).all(1) & (feet > 8).all(1) & ~m["wrist_forbidden"]
        )
        knee_flex = m["joint_pos"][:, self.bridge_right_knee]
        close = (
            true_support
            & (pos < 0.12)
            & (angle < 22)
            & (jrmse < 0.28)
            & (speed < 0.27)
            & (right_share > 0.27)
            & (right_share < 0.73)
            & (knee_flex > 0.30)
            & (m["gripper_error_rad"] < 0.16)
        )
        precision = self.bridge_settle_fraction
        # Loaded expert samples have ~43N on the right forearm, ~0.05m/s root
        # speed and near-even foot loading. Wait for this state before launch.
        right_shell = m["forearm_top_load_N"][:, 1]
        close &= (
            (right_shell >= 18 * precision)
            & (speed < 0.27 - 0.14 * precision)
            & ((right_share - 0.50).abs() < 0.23 - 0.09 * precision)
        )
        return cost, true_support, close

    def _get_dones(self):
        was_bridge = self.bridge_active.clone()
        failed, done = super()._get_dones()
        m = self.last_metrics
        start = m["approach_accepted"] & ~self.bridge_started & ~failed
        cost, support, close = self._bridge_metrics(m)
        self.bridge_initial_cost = torch.where(start, cost, self.bridge_initial_cost)
        self.bridge_target0[start] = self.targets[start]
        self.bridge_age += was_bridge.long()
        self.bridge_started |= start
        improved = was_bridge & support & close & (cost < 0.75 * self.bridge_initial_cost)
        self.bridge_good_streak.copy_(torch.where(improved, self.bridge_good_streak + 1, 0))
        ready = was_bridge & (self.bridge_good_streak >= 4)
        self.bridge_active.copy_((was_bridge | start) & ~ready & ~failed)
        # The parent's provisional entry at the old broad gate is deliberately held
        # in the data collector; no simulator state is assigned at any transition.
        self.ground_stage[self.bridge_active] = 0
        self.course_step[self.bridge_active | ready] = self.course.approach_step
        self.ground_stage[ready] = 1
        if ready.any():
            ids = ready.nonzero().flatten()
            self.knee_contacts.reset(ids)
            self.guided_targets.reset(ids)
            self.ref_steps[ids] = self.reference_wait_steps[ids] = self.stage_age[ids] = 0
            self.end_settle[ids] = self.gate_settle[ids] = 0
            self.actions[ids] = self.previous_actions[ids] = 0
            self.history_steps[ids] = -1
            self.gate_latched[ids] = self.gate_allowed[ids] = True
        timeout = self.bridge_active & (self.bridge_age >= round(2.8 / self.step_dt))
        m.update(
            bridge_start=start,
            bridge_active=was_bridge,
            bridge_cost=cost,
            bridge_initial_cost=self.bridge_initial_cost.clone(),
            bridge_support=support,
            bridge_qualified=ready,
            bridge_partial=(
                was_bridge
                & support
                & (cost < 0.68 * self.bridge_initial_cost)
                & (m["root_speed_m_s"] < 0.26)
                & (m["joint_pos"][:, self.bridge_right_knee] > 0.24)
            ),
            bridge_age=self.bridge_age.clone(),
            approach_accepted=start,
        )
        return failed, done | timeout
