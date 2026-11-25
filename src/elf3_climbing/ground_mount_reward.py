"""Physical shaping for the ground approach and the recovery after knee entry."""

import torch

from .walk import Footsteps


def _quiet(error, scale):
    return 1 / (1 + (error / scale).square())


class GroundMountReward:
    def __init__(self, env):
        self.env = env
        self.weight = env.knee_contacts.total_mass * abs(env.cfg.sim.gravity[2])
        self.forward = env.advance_direction
        self.left = torch.stack((-self.forward[1], self.forward[0]))
        self.foothold = self.weight.new_zeros((env.num_envs, 2))
        self.foothold_seen = torch.zeros_like(self.foothold, dtype=torch.bool)
        self.steps = Footsteps(env.num_envs, self.weight.device)
        self.best_rise = torch.zeros_like(self.weight)
        self.best_transfer = torch.zeros_like(self.weight)
        self.rise_seen = torch.zeros_like(self.weight, dtype=torch.bool)
        self.had_rise_support = torch.zeros_like(self.rise_seen)
        self.success_paid = torch.zeros_like(self.rise_seen)

    def reset(self, ids):
        self.foothold_seen[ids] = False
        self.steps.reset(ids)
        self.foothold[ids] = 0
        self.best_rise[ids] = 0
        self.best_transfer[ids] = 0
        self.rise_seen[ids] = False
        self.had_rise_support[ids] = False
        self.success_paid[ids] = False

    def measure(self, m, stage_before):
        env = self.env
        ground, rise = stage_before == 0, stage_before == 2
        t = env.course_time_s
        # These boundaries belong to the 0.70 m ground-to-rise control prior. The
        # gripper turns before the first step. Intentional forward lean starts at 3.7 s.
        stand = ground & (t < 1.5)
        walk = ground & (t >= 1.5) & (t < 3.7)
        brace = ground & (t >= 3.7)
        feet = env.site_positions()[:, :2] - env.scene.env_origins[:, None]
        feet_xy = feet[:, :, :2]
        velocity = env.robot.data.root_link_lin_vel_w
        forward_speed = velocity[:, :2] @ self.forward
        side_speed = velocity[:, :2] @ self.left
        heading = m["heading_error_deg"].abs()
        angular_speed = m["root_angular_speed_rad_s"]

        # KneeContacts computes tangential *material-point* velocity at measured
        # lower-surface contact points. An unloaded swing foot contributes no weight.
        load = m["foot_ground_patch_N"]
        bearing = load > (0.05 * self.weight[:, None]).clamp_min(12.0)
        loaded = bearing.any(1)
        support = load * bearing
        total = support.sum(1)
        share = support / total.clamp_min(1)[:, None]
        tilt = m["foot_tilt_deg"]
        slip = m["foot_ground_slip_m_s"]
        mean_slip, mean_tilt = (slip * share).sum(1), (tilt * share).sum(1)
        # Allow normal rolling contact and load exchange. A sharply inclined,
        # heavily loaded sole is different from an inclined toe that just lifted.
        grip = _quiet(slip, 0.16)
        level = _quiet((tilt - 15).clamp_min(0), 17)
        plant = (grip * level * share).sum(1)
        support_ratio = (total / (0.75 * self.weight)).clamp(0, 1)
        tilt_body = m["torso_up"].clamp(-1, 1).acos().rad2deg()
        steady = (
            _quiet((tilt_body - 7).clamp_min(0), 12)
            * _quiet((angular_speed - 0.65).clamp_min(0), 1.3)
            * _quiet(heading, 20)
            * _quiet(side_speed, 0.22)
        )
        controlled = plant * steady * support_ratio
        travel = (forward_speed / 0.20).clamp(0, 1) * _quiet(
            (forward_speed - 0.32).clamp_min(0), 0.22
        )
        walk_quality = controlled * travel * _quiet(velocity[:, 2], 0.25)
        stationary = controlled * _quiet(velocity.norm(dim=1), 0.10) * bearing.all(1)

        # Pay once per advance of a newly supported sole. Rocking the torso with
        # fixed feet or swinging the foot in midair cannot accumulate distance.
        foot_x = feet_xy @ self.forward
        switch = self.steps.update(walk, load, feet[:, :, 2] - env.knee_contacts.floor_z, foot_x)
        planted = bearing & (slip < 0.18) & ((walk | stand)[:, None])
        advance = (foot_x - self.foothold).clamp(0, 0.35) * planted * self.foothold_seen
        self.foothold.copy_(
            torch.where(
                planted,
                torch.where(self.foothold_seen, torch.maximum(foot_x, self.foothold), foot_x),
                self.foothold,
            )
        )
        self.foothold_seen |= planted
        # The entry checkpoint defines which upper-limb contacts can carry load.
        # In tabletop mode a wrist on the horizontal face is a valid support.
        arm_load = (m["arm_support_top_N"] / 12).clamp(0, 1)
        shell = (m["forearm_top_load_N"] / 12).clamp(0, 1)
        near = _quiet(m["forearm_target_error_m"], 0.15).mean(1)
        both = arm_load.amin(1) * plant * bearing.all(1) * _quiet(angular_speed, 1.2)
        allowed = ~m["wrist_forbidden"]
        brace_quality = (
            0.15 * near
            + allowed * (both + 0.25 * arm_load.mean(1) + 0.07 * shell.mean(1))
            - 0.55 * m["wrist_forbidden"]
        )
        # Independent rates allow a poor initial gait to improve one physical
        # feature at a time. Perfect posture while standing still has negative
        # walking reward. Forward velocity without planted feet also loses credit.
        slip_score = (grip * share).sum(1)
        tilt_score = (level * share).sum(1)
        walk_rate = (
            1.1 * travel * (0.3 + 0.7 * support_ratio)
            + 0.7 * travel * _quiet(side_speed, 0.22)
            + walk_quality
            - 0.45 * (1 - travel)
            + 0.40 * (support_ratio - 1)
            + 0.40 * (slip_score - 1)
            + 0.40 * (tilt_score - 1)
            + 0.35 * (_quiet((tilt_body - 7).clamp_min(0), 12) - 1)
            + 0.30 * (_quiet(heading, 20) - 1)
            + 0.30 * (_quiet((angular_speed - 0.65).clamp_min(0), 1.3) - 1)
        )
        foothold_reward = walk * (1 + 3 * controlled) * advance.sum(1)
        ground_reward = (
            env.step_dt * (stationary * stand + walk_rate * walk + brace_quality * brace)
            + foothold_reward
            + 0.5 * switch.float()
        )

        rise_reward, rise_quality = self._recovery(m, rise, feet_xy, velocity, side_speed)
        reward = ground_reward + rise_reward
        reward += 3 * m["approach_accepted"].float() + 5 * m["entry_accepted"].float()
        reward -= 2 * m["terminated"].float()
        return {
            "course_reward": reward,
            "course_stage": stage_before,
            "walk_active": walk,
            "rise_active": rise,
            "walk_loaded": loaded & walk,
            "walk_step_switch": switch,
            "walk_slip_m_s": mean_slip * walk,
            "walk_tilt_deg": mean_tilt * walk,
            "walk_quality": walk_quality * walk,
            "rise_quality": rise_quality * rise,
        }

    def _recovery(self, m, rise, feet_xy, velocity, side_speed):
        env = self.env
        tilt = m["foot_tilt_deg"]
        heading = m["heading_error_deg"].abs()
        # Upper-foot load must persist and lie inside the platform. Use the two
        # eigenvalues of the force-weighted patch: a rim or a line has zero breadth.
        top_load = m["foot_top_N"] * m["foot_bearing"]
        top_share = (top_load / self.weight[:, None]).clamp(0, 1)
        major = (m["foot_patch_major_std_m"] / 0.025).clamp(0, 1)
        minor = (m["foot_patch_minor_std_m"] / 0.008).clamp(0, 1)
        sole = (major * minor).sqrt() * _quiet((tilt - 8).clamp_min(0), 12)
        flat_load = (top_share * sole).sum(1).clamp(max=1)
        bilateral = (top_share.amin(1) / 0.20).clamp(0, 1)
        width = (feet_xy[:, 0] - feet_xy[:, 1]).norm(dim=1)
        # Constrain stance only as the second foot takes load. A free leg can
        # move outside standing width to clear the front of the platform.
        stance = torch.lerp(
            torch.ones_like(width), _quiet((width - 0.44).clamp_min(0), 0.12), bilateral
        )
        hips = ((m["hip_above_top_m"] - 0.14) / 0.47).clamp(0, 1)
        posture = hips * m["torso_up"].clamp(0, 1)
        alignment = _quiet(heading, 30)
        quiet_rise = _quiet(m["root_speed_m_s"], 0.40) * _quiet(m["root_angular_speed_rad_s"], 1.1)
        # A first flat foot earns partial progress while knees or arms still help.
        # The remaining credit requires load transfer, hip lift and stable heading.
        foot_quality = (
            flat_load
            * (0.30 + 0.70 * bilateral)
            * (0.25 + 0.75 * posture)
            * (0.35 + 0.65 * alignment)
            * (0.40 + 0.60 * quiet_rise)
            * stance
        )
        # Catch the body on the knee/forearm and transfer load inward before either
        # sole is broad. Losing these supports or landing on a hip cannot erase an
        # ongoing fall after the confirmed knee handoff.
        knee = m["knee_top_N"].clamp_min(0) / self.weight[:, None]
        arms = m["forearm_top_load_N"].clamp_min(0).sum(1) / self.weight
        support = knee.sum(1) + arms + m["foot_top_N"].sum(1) / self.weight
        supported = rise & (support > 0.12)
        self.had_rise_support |= supported | m["entry_accepted"]
        drop = (-torch.minimum(velocity[:, 2], m["com_vz_m_s"]) - 0.20).clamp_min(0)
        roll_rate = env.robot.data.root_link_ang_vel_w[:, :2].norm(dim=1)
        roll = (roll_rate - 1.2).clamp_min(0)
        drift = (side_speed.abs() - 0.32).clamp_min(0)
        fall_rate = (
            3.2 * (drop / 0.35).clamp(max=2)
            + 2.4 * (roll / 1.5).clamp(max=2)
            + 1.2 * (drift / 0.45).clamp(max=2)
        )
        inward = ((m["com_inside_edge_m"] + 0.25) / 0.50).clamp(0, 1)
        leg_share = (knee.sum(1) + top_share.sum(1)).clamp(0, 1)
        transfer = (
            (
                0.30 * leg_share
                + 0.30 * (knee.amin(1) / 0.15).clamp(0, 1)
                + 0.40 * inward * leg_share
            )
            * supported
            * quiet_rise
        )

        # Establish a baseline at the first recovery state, including curriculum
        # resets already up on a knee. A stall or repeated rock earns no fresh gain.
        valid = rise & self.rise_seen
        gain = (foot_quality - self.best_rise).clamp_min(0) * valid
        transfer_gain = (transfer - self.best_transfer).clamp_min(0) * valid
        self.best_rise.copy_(torch.maximum(self.best_rise, foot_quality * rise))
        self.best_transfer.copy_(torch.maximum(self.best_transfer, transfer))
        self.rise_seen |= rise
        success = rise & m["success"] & ~self.success_paid
        self.success_paid |= success
        reward = (
            7 * gain
            + 1.5 * transfer_gain
            + 12 * success.float()
            + env.step_dt
            * rise
            * (2 * m["standing"] * foot_quality - fall_rate * self.had_rise_support)
        )
        # Preserve a bounded diagnostic signal even before the first sole is planted.
        quality = foot_quality + 0.14 * (1 - foot_quality) * transfer
        return reward, quality
