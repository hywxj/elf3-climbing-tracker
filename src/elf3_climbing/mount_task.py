"""Outcome-driven measurements for the ELF3's 0.70 m platform ascent."""

import math

import torch

from .isaac_knee_entry import top_face_points


def sole_contact_patch(detail, box, top_heights, foot_centers):
    """Return actual vertical load and force-weighted in-plane spread on this foot."""
    rows, xyz, nz, keep, force = top_face_points(detail, box, top_heights)
    weight = torch.where(keep, force * nz, 0.0)
    load = torch.zeros_like(top_heights).index_add_(0, rows, weight)
    # Coordinates relative to the foot avoid cancellation for distant clones.
    xy = xyz[:, :2] - foot_centers[rows, :2]
    columns = top_heights.new_zeros((len(top_heights), 2))
    center = columns.index_add_(0, rows, weight[:, None] * xy) / load.clamp_min(1e-8)[:, None]
    dx, dy = (xy - center[rows]).unbind(1)
    products = torch.stack((dx.square(), dx * dy, dy.square()), dim=1)
    covariance = top_heights.new_zeros((len(top_heights), 3))
    covariance.index_add_(0, rows, weight[:, None] * products)
    xx, cross, yy = (covariance / load.clamp_min(1e-8)[:, None]).unbind(1)
    mean = (xx + yy) * 0.5
    half_range = torch.hypot((xx - yy) * 0.5, cross)
    return load, (mean + half_range).clamp_min(0).sqrt(), (mean - half_range).clamp_min(0).sqrt()


class MountTask:
    """Observe physics once per control step, before the environment resets."""

    def __init__(self, env):
        self.env = env
        self.stance = "flat-foot"
        self.contacts = env.knee_contacts
        self.contacts.enable_pose_trace()
        idx = self.contacts.upper
        self.center = env.ref["terrain_box_centers_w"][idx]
        self.half = env.ref["terrain_box_half_sizes"][idx, :2]
        # Terrain boxes have yaw only (checked at scene creation).
        w, _, _, z = env.ref["terrain_box_quat_wxyz"][idx]
        c, s = 1 - 2 * z.square(), 2 * w * z
        self.axes = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
        self.body_weight = self.contacts.total_mass * abs(env.cfg.sim.gravity[2])
        self.contact_steps = max(1, math.ceil(0.06 / env.step_dt - 1e-8))
        self.stand_steps = max(1, math.ceil(1.0 / env.step_dt - 1e-8))
        self.foot_streak = env.starts.new_zeros((env.num_envs, 2))
        self.knee_streak = torch.zeros_like(self.foot_streak)
        self.hold = torch.zeros_like(env.starts)
        self.best = self.body_weight.new_zeros((env.num_envs, 10))
        self.first = torch.ones_like(env.starts, dtype=torch.bool)

    def reset(self, ids):
        self.foot_streak[ids] = self.knee_streak[ids] = 0
        self.hold[ids] = 0
        self.best[ids] = 0
        self.first[ids] = True

    def footprint_margin(self, points):
        """Signed XY distance to the nearest of the four edges (positive inside)."""
        xy = (points[..., :2] - self.center[:2]) @ self.axes
        return (self.half - xy.abs()).amin(dim=-1)

    def measure(self):
        env, contact = self.env, self.contacts
        data = env.robot.data
        sites = env.site_positions()
        local = sites - env.scene.env_origins[:, None]
        feet, knees = local[:, :2], local[:, 6:8]
        com = (data.body_com_pos_w * contact.mass_fraction[..., None]).sum(
            1
        ) - env.scene.env_origins
        com_in = (com - contact.trace_edge) @ contact.trace_inward
        foot_in = (feet - contact.trace_edge) @ contact.trace_inward
        knee_in = (knees - contact.trace_edge) @ contact.trace_inward
        foot_margin = self.footprint_margin(feet)
        foot_height = sites[:, :2, 2] - contact.top_heights[:, None]
        knee_load = torch.stack(
            [
                contact.surface_loads(name, sites[:, index + 6])[0]
                for index, name in enumerate(contact.knee_names)
            ],
            1,
        )
        stance_metrics = {}
        if self.stance == "flat-foot":
            sole_quat = data.body_link_quat_w[:, env.site_ids[:2]]
            sole_up = 1 - 2 * sole_quat[:, :, 1:3].square().sum(dim=2)
            patches = []
            for side, name in enumerate(contact.foot_names):
                sensor = env.support_sensors[name].contact_physx_view
                detail = sensor.get_contact_data(dt=env.cfg.sim.dt)
                patches.append(
                    sole_contact_patch(detail, contact.upper, contact.top_heights, sites[:, side])
                )
            foot_load, major, minor = (torch.stack(v, dim=1) for v in zip(*patches, strict=True))
            spread = (major >= 0.025) & (minor >= 0.008)
            flat = sole_up >= math.cos(math.radians(15.0))
            stance_metrics = dict(
                foot_tilt_deg=sole_up.clamp(-1, 1).acos().rad2deg(),
                foot_patch_major_std_m=major,
                foot_patch_minor_std_m=minor,
                foot_patch_supported=spread,
            )
        elif self.stance == "legacy":
            foot_load = torch.stack(
                [contact.surface_loads(name)[0] for name in contact.foot_names], 1
            )
        else:
            raise ValueError(f"Unknown standing criterion: {self.stance}")
        floor_load = contact.loads(contact.foot_names, contact.lower).clamp_min(0)
        # The complete body must clear the lower level. The longest ELF3 collision
        # shape extends less than 0.50 m from its link origin (including the gripper).
        floor_clearance = data.body_link_pos_w[:, :, 2].amin(1) - contact.floor_heights
        ground_clear = (floor_load.amax(1) < 3) & (floor_clearance > 0.50)
        quaternion = data.root_link_quat_w
        up = 1 - 2 * quaternion[:, 1:3].square().sum(1)  # torso is the root link
        height = data.root_link_pos_w[:, 2] - contact.top_heights
        hip_height = data.body_link_pos_w[:, contact.hip_ids, 2].mean(1) - contact.top_heights
        speed = data.root_link_lin_vel_w.norm(dim=1)
        angular_speed = data.root_link_ang_vel_w.norm(dim=1)
        loaded = (foot_load > (0.05 * self.body_weight[:, None]).clamp_min(20.0)) & (
            foot_margin >= 0
        )
        self.foot_streak.copy_(torch.where(loaded, self.foot_streak + 1, 0))
        self.knee_streak.copy_(torch.where(knee_load > 8.0, self.knee_streak + 1, 0))
        foot_bearing = self.foot_streak >= self.contact_steps
        knee_bearing = self.knee_streak >= self.contact_steps
        both = foot_bearing.all(1)
        standing = (
            loaded.all(1)
            & (foot_load.sum(1) > 0.8 * self.body_weight)
            & ground_clear
            & (self.footprint_margin(com) >= 0)
            & (up > 0.9)
            & (height > 0.75)
            & (hip_height > 0.55)
            & (speed < 0.25)
            & (angular_speed < 0.5)
        )
        if self.stance == "flat-foot":
            standing &= (flat & spread).all(dim=1)
        self.hold.copy_(torch.where(standing, self.hold + 1, 0))
        success = self.hold >= self.stand_steps

        # Per-feature high-water marks give each physical gain at most once. Feet
        # earn approach credit only as they near the upper surface and its footprint.
        reach_error = torch.hypot((-foot_margin).clamp_min(0), (-foot_height).clamp_min(0))
        foot_reach = (1 - reach_error / 0.7).clamp(0, 1)
        knee = (self.knee_streak.float() / self.contact_steps).clamp(max=1)
        foot = (self.foot_streak.float() / self.contact_steps).clamp(max=1)
        foot_weight = (foot_load.sum(1) / self.body_weight).clamp(0, 1) * foot_bearing.any(1)
        lift = ((hip_height - 0.15) / 0.55).clamp(0, 1) * foot_bearing.any(1)
        progress = torch.cat(
            (
                ((com_in + 0.4) / 0.55).clamp(0, 1)[:, None],
                knee,
                2 * foot_reach,
                3 * foot,
                2 * foot_weight[:, None],
                2 * lift[:, None],
                5 * (self.hold.float() / self.stand_steps).clamp(max=1)[:, None],
            ),
            1,
        )
        gain = (progress - self.best).clamp_min(0).sum(1)
        reward = torch.where(self.first, 0.0, gain) + 10 * (self.hold == self.stand_steps).float()
        self.best.copy_(torch.maximum(progress, self.best))
        self.first.zero_()
        stage = torch.where(foot_bearing.any(1), 2 + foot_bearing.sum(1), knee_bearing.sum(1))
        stage = torch.where(standing, 5, stage)
        return {
            **stance_metrics,
            "success": success,
            "standing": standing,
            "reward": reward,
            "stand_hold_s": self.hold.float() * env.step_dt,
            "stage": stage,
            "foot_bearing": foot_bearing,
            "knee_bearing": knee_bearing,
            "both_feet_loaded": both,
            "foot_top_N": foot_load,
            "foot_floor_N": floor_load,
            "knee_top_N": knee_load,
            "foot_inside_edge_m": foot_in,
            "knee_inside_edge_m": knee_in,
            "com_inside_edge_m": com_in,
            "foot_above_top_m": foot_height,
            "torso_up": up,
            "root_speed_m_s": speed,
            "hip_above_top_m": hip_height,
            "ground_clear": ground_clear,
        }
