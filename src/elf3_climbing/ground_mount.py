"""A single physical episode from ground standing to aligned flat-foot standing on the ledge."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .environment import TrackerEnv
from .ground_locomotion import GroundLocomotion
from .ground_mount_reward import GroundMountReward
from .ground_support import GroundSupport
from .ground_trajectory import GroundTrajectory, course_heading_error, forearm_gate
from .knee_transfer import point_jacobian
from .mount_env import MountEnv


class GroundMountEnv(MountEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self.ground_support = GroundSupport(self)

    def __init__(self, cfg, trajectory, **kwargs):
        if cfg.random_start or cfg.curriculum_enabled:
            raise ValueError("The ground course selects its own physical reset states")
        # Stage, clock, route direction, velocity and 4 measurements for each foot.
        cfg.observation_space += 17
        cfg.state_space += 17
        super().__init__(cfg, **kwargs)
        self.course = GroundTrajectory(
            trajectory,
            self.reference_np["joint_names"],
            self.step_dt,
            self.segment_start,
            self.device,
        )
        if (
            not torch.allclose(self.course.poses[0, :3], self.ref["root_pos"][0], atol=0.01)
            or not torch.allclose(
                self.course.poses[self.course.approach_step, :3],
                self.ref["root_pos"][self.segment_start],
                atol=0.08,
            )
            or not torch.allclose(
                self.course.poses[self.course.recovery_step, :3],
                self.guided_targets.landing_root_pose[:3],
                atol=0.14,
            )
        ):
            raise ValueError("Course phases are not aligned to this terrain and frozen knee guide")
        self.stage_training = False
        self.ground_stage = torch.zeros_like(self.starts)
        self.stage_age = torch.zeros_like(self.starts)
        self.course_step = torch.zeros_like(self.starts)
        self.entry_streak = torch.zeros_like(self.starts)
        self.aligned_stand = torch.zeros_like(self.starts)
        self.reset_count = torch.zeros_like(self.starts)
        self.balance_gains = self.actions.new_tensor([14.0, 2.8, 1.5, 0.15]).repeat(
            self.num_envs, 1
        )
        self.gait = GroundLocomotion(self.reference_np["joint_names"], self.device)
        self.gait_actions = torch.zeros_like(self.actions)
        self.training_stage_steps = self.starts.new_zeros(3)
        self.ground_contact_obs = self.actions.new_zeros((self.num_envs, 8))
        self.walk_dock_step = round(3.68 / self.step_dt)
        self.brace_settings = self.actions.new_tensor([-0.06, 0.01, 0.075, 15.0]).repeat(
            self.num_envs, 1
        )
        self.entry_pose = torch.cat(
            (self.ref["root_pos"][self.segment_start], self.ref["root_quat"][self.segment_start])
        )
        # Give arm placement and recovery time to settle even if they lag the example.
        approach_budget = self.course.approach_step + round(5.0 / self.step_dt)
        knee_budget = self.segment_end - self.segment_start + cfg.extra_budget_steps
        self.mount_steps = (
            self.course.last_step - self.course.recovery_step + round(3 / self.step_dt)
        )
        self.mount_duration_s = self.mount_steps * self.step_dt
        self.stage_budgets = self.starts.new_tensor(
            [approach_budget, knee_budget, self.mount_steps]
        )
        cfg.episode_length_s = (approach_budget + knee_budget + self.mount_steps + 1) * self.step_dt
        self.ground_reward = GroundMountReward(self)

    @property
    def course_time_s(self):
        return self.course_step * self.step_dt

    @property
    def course_reference_root(self):
        return self.course.poses[self.course_step, :3]

    @property
    def course_targets(self):
        return self.course.controls[self.course_step]

    def _get_observations(self):
        observations = super()._get_observations()
        stage = torch.nn.functional.one_hot(self.ground_stage, 3)
        progress = (self.course_step / self.course.last_step).unsqueeze(1)
        yaw = yaw_quat(self.robot.data.root_link_quat_w)
        forward = torch.nn.functional.pad(self.advance_direction, (0, 1)).expand(self.num_envs, -1)
        route = quat_apply_inverse(yaw, forward)[:, :2]
        speed = quat_apply_inverse(yaw, self.robot.data.root_link_lin_vel_w)
        state = torch.cat((stage, progress, route, speed, self.ground_contact_obs), dim=1)
        return {key: torch.cat((obs, state), dim=1) for key, obs in observations.items()}

    def _pre_physics_step(self, actions):
        if self.entry_policy is None:
            raise RuntimeError("Attach the frozen knee-entry policy before stepping")
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite ground course residual")
        self.mount_residual.copy_(actions)
        with torch.no_grad():
            entry_actions = self.entry_policy(self.entry_obs)
        applied = torch.where((self.ground_stage == 1)[:, None], entry_actions, actions)
        TrackerEnv._pre_physics_step(self, applied)

    def _joint_command(self):
        # The prior supplies motor targets only. The simulation owns every floating-base DOF.
        residual = self.mount_residual.clamp(-2, 2) * 0.42
        residual[:, self.wrist_joint] = 0
        targets = self._balance_ground(self.course_targets.clone())
        t = self.course_time_s
        blend = ((t - 1.10) / 0.35).clamp(0, 1) * ((4.05 - t) / 0.38).clamp(0, 1)
        blend *= self.ground_stage == 0
        if blend.any():
            root = self.robot.data.root_link_pos_w - self.scene.env_origins
            yaw_q = self.robot.data.root_link_quat_w
            w, x, y, z = yaw_q.unbind(1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
            direction = self.advance_direction
            left = torch.stack((-direction[1], direction[0]))
            dock = (
                self.course.poses[self.walk_dock_step, :2] + self.brace_settings[:, :1] * direction
            )
            error = dock - root[:, :2]
            forward = ((error @ direction) * 1.5).clamp(-0.10, 0.35) * ((t - 1.45) / 0.35).clamp(
                0, 1
            )
            side = ((error @ left) * 2.8).clamp(-0.30, 0.30)
            h = yaw - torch.atan2(direction[1], direction[0])
            command = torch.stack(
                (
                    h.cos() * forward + h.sin() * side,
                    -h.sin() * forward + h.cos() * side,
                    (-1.8 * torch.atan2(h.sin(), h.cos())).clamp(-0.35, 0.35),
                ),
                1,
            )
            walk_targets = self.gait.targets(self, command, blend > 0)
            walk_targets[:, self.wrist_joint] = targets[:, self.wrist_joint]
            targets = torch.lerp(targets, walk_targets, blend[:, None])
        brace = ((t - 3.90) / 0.60).clamp(0, 1) * (self.ground_stage == 0)
        if brace.any():
            targets = self._brace_forearms(targets, brace)
        targets += residual
        knee = self.ground_stage == 1
        if knee.any():
            targets = torch.where(knee[:, None], TrackerEnv._joint_command(self), targets)
        return targets

    def _brace_forearms(self, targets, blend):
        data = self.robot.data
        jacobians = self.robot.root_physx_view.get_jacobians()
        sites = self.site_positions()
        for side, (body, cols, joints, dofs) in enumerate(self.guided_targets.arms):
            j = jacobians[:, body, :, dofs]
            com = data.body_com_pos_w[:, body]
            shell, elbow = sites[:, 2 + side], sites[:, 4 + side]
            shell_j = point_jacobian(j, shell - com)
            elbow_j = point_jacobian(j, elbow - com)
            goal = self.scene.env_origins + self.ref["site_pos"][self.segment_start, 2 + side]
            goal[:, 2] -= self.brace_settings[:, 1]
            elbow_z = (
                self.scene.env_origins[:, 2] + self.ref["site_pos"][self.segment_start, 4 + side, 2]
            )
            error = torch.cat(
                (goal - shell, (elbow_z - self.brace_settings[:, 2] - elbow[:, 2])[:, None]), 1
            )
            task_j = torch.cat((shell_j, elbow_j[:, 2:3]), 1)
            current = data.joint_pos[:, joints]
            dq = (targets[:, cols] - current).unsqueeze(-1)
            correction, _ = torch.linalg.solve_ex(
                task_j @ task_j.mT + self.guided_targets.regularization,
                error.unsqueeze(-1) - task_j @ dq,
            )
            preload = (
                -self.brace_settings[:, 3:4]
                * shell_j[:, 2]
                / data.joint_stiffness[:, joints].clamp_min(1)
            )
            arm = current + (dq.squeeze(-1) + (task_j.mT @ correction).squeeze(-1) + preload).clamp(
                -0.7, 0.7
            )
            targets[:, cols] = torch.lerp(targets[:, cols], arm, blend[:, None])
        return targets

    def _balance_ground(self, targets):
        data = self.robot.data
        # A forward-drifting body needs plantarflexion at the supporting ankle.
        # Apply the same inverted-pendulum correction in the lateral direction,
        # with opposite sign because positive roll tips the body to its right.
        error_xy = (data.root_link_pos_w - self.scene.env_origins - self.course_reference_root)[
            :, :2
        ]
        velocity_xy = (data.root_link_lin_vel_w - self.course.velocities[self.course_step, :3])[
            :, :2
        ]
        # Smooth out the large instantaneous lateral velocity excursions in the old walking prior.
        standing = self.course_time_s < 1.45
        velocity_xy = torch.where(standing[:, None], data.root_link_lin_vel_w[:, :2], velocity_xy)
        fwd = self.advance_direction
        left = torch.stack((-fwd[1], fwd[0]))
        p, d, angle_gain, rate_gain = self.balance_gains.unbind(dim=1)
        position_pd = p[:, None] * error_xy + d[:, None] * velocity_xy
        projection = data.projected_gravity_b
        pitch = torch.atan2(projection[:, 0], -projection[:, 2])
        roll = torch.atan2(-projection[:, 1], -projection[:, 2])
        factor = ((3.85 - self.course_time_s) / 0.35).clamp(0, 1) * (self.ground_stage == 0)
        angle_pd = (
            angle_gain[:, None] * torch.stack((roll, pitch), 1)
            + rate_gain[:, None] * data.root_ang_vel_b[:, :2]
        )
        pitch_cmd = ((position_pd @ fwd) + 0.35 * angle_pd[:, 1]).clamp(-0.65, 0.65) * factor
        roll_cmd = (-(position_pd @ left) + 0.35 * angle_pd[:, 0]).clamp(-0.5, 0.5) * factor
        for ankle_pitch, ankle_roll in ((7, 8), (13, 14)):
            targets[:, ankle_pitch] += pitch_cmd
            targets[:, ankle_roll] += roll_cmd
        targets[:, 0] += factor * angle_pd[:, 1].clamp(-0.35, 0.35)
        targets[:, 1] += factor * angle_pd[:, 0].clamp(-0.3, 0.3)
        return targets

    def _get_dones(self):
        before = self.ground_stage.clone()
        approach, knee, recovery = before == 0, before == 1, before == 2
        if self.stage_training:
            self.training_stage_steps += torch.stack((approach.sum(), knee.sum(), recovery.sum()))
        failed, _ = super()._get_dones()
        m = self.last_metrics
        weight = self.mount_task.body_weight[:, None]
        self.ground_contact_obs.copy_(
            torch.cat(
                (
                    (m["foot_ground_patch_N"] / weight).clamp(0, 2),
                    m["foot_ground_slip_m_s"].clamp(0, 2),
                    (m["foot_top_N"] / weight).clamp(0, 2),
                    m["foot_tilt_deg"] / 90,
                ),
                1,
            )
        )
        self.stage_age += 1
        streak, ready = forearm_gate(m, self.entry_pose, self.entry_streak)
        self.entry_streak.copy_(torch.where(approach, streak, 0))
        approach_fell = (m["root_pose"][:, 2] - self.knee_contacts.floor_z < 0.46) | (
            m["torso_up"] < 0.18
        )
        failed = torch.where(approach, approach_fell, failed)
        enter_knee = approach & ready & ~failed
        enter_recovery = knee & m["entry_accepted"] & ~failed
        self.ground_stage[enter_knee] = 1
        self.ground_stage[enter_recovery] = 2
        self.mount_active.copy_(self.ground_stage == 2)
        # The legacy knee controller must see the original phase counters on its very first command.
        self.ref_steps[approach] = 0
        self.reference_wait_steps[approach] = 0
        enter_ids = enter_knee.nonzero().flatten()
        if len(enter_ids):
            self.knee_contacts.reset(enter_ids)
            self.guided_targets.reset(enter_ids)
            # The entrant has no previous actions from its own actor. Rebuild its
            # history from the measured handoff posture, as in its original reset.
            self.actions[enter_ids] = self.previous_actions[enter_ids] = 0
            self.history_steps[enter_ids] = -1
            self.end_settle[enter_ids] = 0
            self.gate_settle[enter_ids] = 0
            self.gate_latched[enter_ids] = True
            self.gate_allowed[enter_ids] = True
        if enter_recovery.any():
            self.mount_task.reset(enter_recovery.nonzero().flatten())
        self.mount_obs[~self.mount_active] = 0
        self.mount_age[~self.mount_active] = 0
        # A successful stand requires the full balance and heading criteria on the same steps.
        m["heading_error_deg"] = course_heading_error(
            m["root_pose"][:, 3:7], self.advance_direction
        )
        candidate = recovery & m["standing"] & (m["heading_error_deg"] <= 30)
        other_support = self.ground_support.force(candidate)
        m["nonfoot_platform_force_N"] = other_support
        m["nonfoot_support_checked"] = candidate
        aligned = candidate & (other_support < 5)
        self.aligned_stand.copy_(torch.where(aligned, self.aligned_stand + 1, 0))
        stand_steps = self.mount_task.stand_steps
        success = self.aligned_stand >= stand_steps
        m["standing"] = aligned
        m["stage"] = torch.where((m["stage"] == 5) & ~aligned, 4, m["stage"])
        m["stand_hold_s"] = self.aligned_stand * self.step_dt
        self.mount_obs[:, 2] = m["stand_hold_s"]
        changed = enter_knee | enter_recovery
        expired = (self.stage_age >= self.stage_budgets[before]) & ~changed & ~success
        m.update(
            approach_accepted=enter_knee,
            entry_accepted=enter_recovery,
            mount_active=self.mount_active.clone(),
            terminated=failed,
            expired=expired,
            success=success,
            course_stage=before,
            course_time_s=self.course_time_s,
        )
        m.update(self.ground_reward.measure(m, before))
        following = self.course.advance(self.course_step, before, enter_knee, enter_recovery)
        dock = (
            self.course.poses[self.walk_dock_step, :2]
            + self.brace_settings[:, :1] * self.advance_direction
        )
        to_dock = dock - m["root_pose"][:, :2]
        wait = approach & (self.course_step == self.walk_dock_step) & (to_dock.norm(dim=1) > 0.085)
        self.course_step = torch.where(wait & ~enter_knee, self.course_step, following)
        self.stage_age[changed] = 0
        return failed, expired | success

    def _get_rewards(self):
        return self.last_metrics["course_reward"]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "ground_stage"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        stage, step = self.course.seeds(env_ids, self.reset_count[env_ids], self.stage_training)
        if self.stage_training:
            self.reset_count[env_ids] += 1
        self.ground_stage[env_ids] = stage
        self.gait_actions[env_ids] = 0
        self.ground_contact_obs[env_ids] = 0
        self.course_step[env_ids] = step
        self.stage_age[env_ids] = self.aligned_stand[env_ids] = self.entry_streak[env_ids] = 0
        self.mount_active[env_ids] = stage == 2
        self.mount_obs[env_ids, 0] = (stage == 2).float()
        # The parent resets the trained knee entry exactly as before. Overwrite only non-knee
        # reset states in this same reset call; physical phase changes above never write pose.
        source = stage != 1
        ids, rows = env_ids[source], step[source]
        if len(ids):
            pose = self.course.poses[rows]
            velocity = self.course.velocities[rows]
            root = pose[:, :7].clone()
            root[:, :3] += self.scene.env_origins[ids]
            offset = quat_apply(root[:, 3:], self.robot.data.body_com_pos_b[ids, 0])
            angular = velocity[:, 3:6]
            vcom = velocity[:, :3] + torch.cross(angular, offset, dim=1)
            q = self.robot.data.default_joint_pos[ids].clone()
            qd = torch.zeros_like(q)
            q[:, self.joint_ids] = pose[:, 7:]
            qd[:, self.joint_ids] = velocity[:, 6:]
            self.robot.write_root_pose_to_sim(root, env_ids=ids)
            self.robot.write_root_velocity_to_sim(torch.cat((vcom, angular), dim=1), env_ids=ids)
            self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
            self.targets[ids] = pose[:, 7:]
            self.robot.set_joint_position_target(
                self.targets[ids], joint_ids=self.joint_ids, env_ids=ids
            )
        self.ground_reward.reset(env_ids)
