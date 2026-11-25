"""Commanded walking on the existing level starting surface after the accepted flip."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .environment import TrackerEnv, OBS_SIZE, CRITIC_SIZE
from .protocol import advance_gate, flip_retention_failures, premature_step
from .walk import FEATURES, Footsteps, HALF, STOP_STEPS, WALK_STEPS, reward_rates, walk_command


class WalkEnv(TrackerEnv):
    def __init__(self, cfg):
        cfg.observation_space = OBS_SIZE + FEATURES
        cfg.state_space = CRITIC_SIZE + FEATURES + 4  # planar speeds and measured foot support
        cfg.extra_budget_steps = WALK_STEPS + STOP_STEPS + 150
        super().__init__(cfg)
        self.prefix_test = False
        self.fixed_commands = False
        self.tick = torch.full_like(self.starts, -1)
        self.speeds = torch.zeros((self.num_envs, 2), device=self.device)
        self.first_side = self.speeds[:, 0].clone()
        self.footsteps = Footsteps(self.num_envs, self.device)
        self.last_feet = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.was_loaded = torch.zeros_like(self.speeds, dtype=torch.bool)
        self.walk_origin = self.speeds.clone()
        self.errors = self.speeds.clone()
        self.samples = torch.zeros_like(self.speeds, dtype=torch.long)
        self.stance_slip = self.first_side.clone()
        self.max_wrist_error = self.first_side.clone()
        quat0 = yaw_quat(self.ref["root_quat"][0:1])
        self.forward = quat_apply(quat0, self.ref["root_pos"].new_tensor([[1.0, 0.0, 0.0]]))[0, :2]
        self.left = torch.stack((-self.forward[1], self.forward[0]))
        self.floor = self.task_report["ground_top_m"]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        self.tick[ids] = -1
        self.footsteps.reset(ids)
        self.was_loaded[ids] = False
        self.last_feet[ids] = self.ref["site_pos"][self.starts[ids], :2]
        self.walk_origin[ids] = self.ref["root_pos"][self.starts[ids], :2]
        self.errors[ids] = 0
        self.samples[ids] = 0
        self.stance_slip[ids] = 0
        self.max_wrist_error[ids] = 0
        if self.fixed_commands:
            slow = self.speeds.new_full((len(ids),), 0.12)
            fast = torch.full_like(slow, 0.32)
            reverse = ids % 2 == 1
            self.first_side[ids] = ((ids // 2) % 2).float()
        else:
            speeds = torch.rand(len(ids), 2, device=self.device)
            slow, fast = 0.12 + 0.06 * speeds[:, 0], 0.26 + 0.06 * speeds[:, 1]
            reverse = torch.rand_like(slow) < 0.5
            self.first_side[ids] = torch.randint(2, (len(ids),), device=self.device).float()
        self.speeds[ids] = torch.stack(
            (torch.where(reverse, fast, slow), torch.where(reverse, slow, fast)), dim=1
        )

    def _get_observations(self):
        obs = super()._get_observations()
        _, features = walk_command(self.tick, self.speeds, self.first_side, self.step_dt)
        d = self.robot.data
        # Recenter the frozen prior's horizontal foot and torso goals while travelling.
        # Its original height, wrist and proprioceptive feedback remain live.
        a = obs["policy"]
        # Follow the feet so torso sway relative to its supports still produces
        # the corrective feedback the standing policy learned.
        shift = self.last_feet.mean(1) - self.ref["site_pos"][self.flip_end_frame, :2].mean(0)
        shift[:, 2] = 0
        local = quat_apply_inverse(yaw_quat(d.root_link_quat_w), shift) * features[:, -1:]
        goals = a[:, 591:615].reshape(-1, 8, 3)
        goals += local[:, None, :] * a[:, 615:623, None]
        a[:, 623:626] += local
        obs["critic"][:, :OBS_SIZE] = a
        velocity = d.root_link_lin_vel_w[:, :2]
        extra = torch.stack((velocity @ self.forward, velocity @ self.left), dim=1)
        obs["policy"] = torch.cat((a, features), dim=1)
        obs["critic"] = torch.cat((obs["critic"], features, extra, self.was_loaded.float()), dim=1)
        return obs

    def _get_dones(self):
        if self.prefix_test:
            return super()._get_dones()
        d = self.robot.data
        dt = self.step_dt
        speed, features = walk_command(self.tick, self.speeds, self.first_side, dt)
        active = self.tick >= 0
        moving = speed > 0
        braking = active & ~moving
        wrist, wrist_vel, clearance, _ = self.gripper_state()
        settle, latch, allowed, _ = advance_gate(
            wrist, wrist_vel, clearance, self.gate_settle, self.gate_latched, self.rules
        )
        self.gate_settle, self.gate_latched, self.gate_allowed = settle, latch, allowed
        self.ref_steps += 1
        start = ~active & allowed & (self.frames() >= self.flip_end_frame)
        root = d.root_link_pos_w - self.scene.env_origins
        self.walk_origin = torch.where(start[:, None], root[:, :2], self.walk_origin)
        displacement = root[:, :2] - self.walk_origin
        distance, lateral = displacement @ self.forward, displacement @ self.left
        vel = d.root_link_lin_vel_w[:, :2]
        forward, sideways = vel @ self.forward, vel @ self.left
        heading = quat_apply(
            yaw_quat(d.root_link_quat_w),
            d.root_link_pos_w.new_tensor([1.0, 0.0, 0.0]).expand(self.num_envs, -1),
        )[:, :2]
        yaw_error = torch.atan2(heading @ self.left, heading @ self.forward)
        tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1))
        feet = self.site_positions()[:, :2] - self.scene.env_origins[:, None, :]
        support_forces = self.support_forces()
        load = support_forces[:, :2, 2]
        loaded = load > 20
        height = feet[:, :, 2] - self.floor
        switched_foot = self.footsteps.update(moving, load, height, feet[:, :, :2] @ self.forward)
        contact = loaded & self.was_loaded
        speed_at_sole = (feet[:, :, :2] - self.last_feet[:, :, :2]).norm(dim=-1) / dt
        slip = (speed_at_sole * contact).sum(1) / contact.sum(1).clamp_min(1)
        self.last_feet.copy_(feet)
        self.was_loaded.copy_(loaded)
        self.stance_slip += slip * moving * dt
        self.max_wrist_error = torch.maximum(self.max_wrist_error, wrist.abs() * active)
        error = (forward - speed).abs()
        spans = torch.stack((moving & (self.tick < HALF), moving & (self.tick >= HALF)), dim=1)
        self.errors += spans * error[:, None]
        self.samples += spans.long()
        mean_errors = self.errors / self.samples.clamp_min(1)
        premature, _ = premature_step(
            feet,
            self.ref["site_pos"][self.starts, :2],
            self.forward,
            self.gate_latched,
            ~active,
            self.rules,
        )
        lost, collision = flip_retention_failures(wrist, clearance, latch, self.rules)
        contact_force = self.contacts.data.net_forces_w[:, self.forbidden_sensor_ids].norm(dim=-1)
        fall = ((root[:, 2] - self.ref["root_pos"][0, 2]).abs() > 0.32) | (tilt > 0.8)
        forbidden = (contact_force > 100).any(1) | (support_forces[:, 2:].norm(dim=-1) > 10).any(1)
        failure = (
            fall
            | forbidden
            | (lateral.abs() > 0.45)
            | (yaw_error.abs() > 0.9)
            | lost
            | collision
            | premature
        )
        stable = (
            braking
            & loaded.all(1)
            & (vel.norm(dim=1) < 0.08)
            & (d.root_ang_vel_w.norm(dim=1) < 0.5)
            & (tilt < 0.22)
            & (slip < 0.08)
        )
        self.end_settle = torch.where(stable, self.end_settle + 1, 0)
        expected = self.speeds.sum(1) * HALF * dt
        finish = self.tick >= WALK_STEPS + STOP_STEPS - 1
        success = (
            finish
            & ~failure
            & (self.samples == HALF).all(1)
            & (mean_errors < 0.09).all(1)
            & (distance >= 0.75 * expected)
            & (self.footsteps.counts >= 3).all(1)
            & (self.end_settle >= 50)
            & (lateral.abs() < 0.20)
            & (self.stance_slip < 0.45)
        )
        expired = finish | (
            ~active & ~start & (self.episode_length_buf >= self.flip_end_frame + 150)
        )
        self.last_metrics = {
            "success": success,
            "terminated": failure,
            "root_error_m": lateral.abs(),
            "gripper_error_rad": wrist.abs(),
            "gripper_collision": collision,
            "premature_advance": premature,
            "lost_flip": lost,
            "fall": fall,
            "forbidden_contact": forbidden,
            "walking_started": active | start,
            "distance_m": distance,
            "expected_distance_m": expected,
            "left_steps": self.footsteps.counts[:, 0].clone(),
            "right_steps": self.footsteps.counts[:, 1].clone(),
            "first_speed": self.speeds[:, 0].clone(),
            "second_speed": self.speeds[:, 1].clone(),
            "first_samples": self.samples[:, 0].clone(),
            "second_samples": self.samples[:, 1].clone(),
            "first_speed_mae": mean_errors[:, 0],
            "second_speed_mae": mean_errors[:, 1],
            "halt_hold_s": self.end_settle * dt,
            "stance_slip_m": self.stance_slip.clone(),
            "peak_walk_wrist_error_rad": self.max_wrist_error.clone(),
        }
        self.rates = reward_rates(
            speed,
            forward,
            sideways,
            tilt,
            d.root_ang_vel_b[:, :2].norm(dim=1),
            yaw_error,
            wrist,
            height,
            load,
            slip,
            features[:, 1],
            (self.actions - self.previous_actions).square().mean(1),
            (
                d.applied_torque[:, self.joint_ids]
                / d.joint_effort_limits[:, self.joint_ids].clamp_min(1)
            )
            .square()
            .mean(1),
        )
        # No walking reward during the frozen prefix. Issued command and scored transition agree.
        self.rates = {k: v * active for k, v in self.rates.items()}
        self.walk_reward = (
            sum(self.rates.values()) * dt + 0.12 * switched_foot - 2.0 * failure + 3.0 * success
        )
        self.tick = torch.where(active | start, self.tick + 1, self.tick)
        return failure, expired

    def _get_rewards(self):
        if self.prefix_test:
            return super()._get_rewards()
        self.extras["log"] = {"WalkReward/" + k: v.mean() for k, v in self.rates.items()}
        for k in ("walking_started", "left_steps", "right_steps", "distance_m", "halt_hold_s"):
            self.extras["log"]["Walk/" + k] = self.last_metrics[k].float().mean()
        return self.walk_reward
