"""Training-only interventions on states visited by a contact student."""

import torch

from .contact_tracker_task import direct_joint_targets


def mixed_command(previous, teacher, student, nominal, limits, rates, dt, fraction):
    """Compare both actions under the same actuator constraints and previous applied command."""
    candidate = direct_joint_targets(student, nominal, previous, limits, rates, dt)
    return torch.lerp(teacher, candidate, fraction[:, None])


def rollin_schedule(env, step, ground):
    # Stagger interventions so the data includes perturbations at initial standing,
    # walking and reaching, or at the preparation, push and flight of the old knee actor.
    ids = torch.arange(env.num_envs, device=env.device)
    starts = env.targets.new_tensor([0.0, 1.50, 3.50] if ground else [0.0, 0.30, 0.70])
    fraction = env.targets.new_tensor([0.70, 0.85, 1.0])[ids % 3]
    return fraction * (step * env.step_dt >= starts[(ids // 3) % 3])


def wait_for_walking_position(env):
    """Delay the ground teacher's reaching command if the mixed policy lags the dock."""
    t = env.course_time_s
    root = env.robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    forward = env.advance_direction
    left = torch.stack((-forward[1], forward[0]))
    dock = env.course.poses[env.walk_dock_step, :2] + env.brace_settings[:, :1] * forward
    delta = dock - root
    early = (delta @ forward > 0.14) | ((delta @ left).abs() > 0.18)
    wait = (env.ground_stage == 0) & (t >= 3.55) & (t < 3.7) & early
    env.course_step[wait] = round(3.55 / env.step_dt)
    return wait


def apply_intervention(env, previous, ground):
    """Change only the motor target; the simulator still advances every floating DOF."""
    fraction = env.student_fraction
    expert = env.applied_drive_target
    actual = mixed_command(
        previous,
        expert,
        env.student_suggestion,
        env.nominal,
        env.limits,
        env.target_rates,
        env.step_dt,
        fraction,
    )
    env.targets.copy_(actual)
    # A teacher trained with normalized residual history has no unique inverse after
    # IK/saturation. Feed back the executed correction in its own residual units,
    # retaining the teacher's exact representation where it actually acted alone.
    scale = torch.where(env.ground_stage[:, None] == 1, 0.50, 0.42) if ground else 0.50
    effective = (env.actions + (actual - expert) / scale).clamp(-3, 3)
    effective[:, env.wrist_joint] = (
        actual[:, env.wrist_joint] - env.nominal[:, env.wrist_joint]
    ) / env.cfg.action_scale
    env.actions.copy_(torch.where(fraction[:, None] > 0, effective, env.actions))
    if ground:
        d = env.robot.data
        q, qd = d.joint_pos[:, env.joint_ids], d.joint_vel[:, env.joint_ids]
        kp = d.joint_stiffness[:, env.joint_ids]
        kd = d.joint_damping[:, env.joint_ids]
        # Equivalent vendor-space drive reproducing the actually executed PD torque.
        vendor_q = q + (kp * (actual - q) + (env.gait.kd - kd) * qd) / env.gait.kp
        normalized = (vendor_q - env.gait.default) / env.gait.scale
        # The wrist flip is imposed separately, and is subtracted from gait proprioception.
        normalized[:, env.wrist_joint] = 0
        active = (
            (fraction > 0)
            & (env.ground_stage == 0)
            & (env.course_time_s > 1.10)
            & (env.course_time_s < 4.05)
        )
        env.gait_actions.copy_(torch.where(active[:, None], normalized, env.gait_actions))
    env.executed_drive_target = actual.clone()


class BridgeIntervention:
    """Brief student roll-ins at walking, reaching, established bracing or knee transfer."""

    def __init__(self, env, actor):
        from .contact_point_actor import PointHistory

        self.actor = actor
        self.history = PointHistory(env.num_envs, env.device)
        ids = torch.arange(env.num_envs, device=env.device)
        self.group = ids % 5
        self.peak = env.targets.new_tensor([0.15, 0.40, 0.70, 1.0])[(ids // 5) % 4]
        self.start = ids.new_full((env.num_envs,), -1)
        self.width = round(0.30 / env.step_dt)
        self.recovery = round(0.62 / env.step_dt)

    def prepare(self, env, raw, step, alive):
        from .contact_learning import encode_contact_observation

        ground = env.ground_stage == 0
        t = env.course_time_s
        draw = (
            ((self.group == 0) & ground & (t >= 1.5))
            | ((self.group == 1) & ground & (t >= 3.1))
            | ((self.group == 2) & env.bridge_active & (env.bridge_age >= 2))
            | ((self.group == 3) & (env.ground_stage == 1))
            | (
                (self.group == 4)
                & (env.ground_stage == 1)
                & (env.stage_age >= round(0.64 / env.step_dt))
            )
        )
        self.start = torch.where(alive & draw & (self.start < 0), step, self.start)
        age = step - self.start
        # Ease in/out over 3 control steps; the teacher then recovers the visited physical state.
        pulse = ((age + 1) / 3).clamp(0, 1) * ((self.width - age) / 3).clamp(0, 1)
        env.student_fraction = self.peak * pulse * (self.start >= 0) * alive
        base = encode_contact_observation(raw, env.nominal)
        geometry = (
            env.ref[k]
            for k in ("terrain_box_centers_w", "terrain_box_half_sizes", "terrain_box_quat_wxyz")
        )
        obs = self.history.encode(base, raw, *geometry, step)
        env.student_suggestion = self.actor(obs)
