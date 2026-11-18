"""Independent Isaac Lab contact-conditioned tracker prototype (simulation only)."""

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from .robot_cfg import ELF3LITE_CFG
from .reference import read_pack
from .paths import DEFAULT_PACK, URDF
from .gripper import specification, apply_usd_mass
from .protocol import (
    GripperRules,
    advance_gate,
    obb_separation,
    BackwardCurriculum,
    premature_step,
    completion_bonus,
)

PROPRIO_SIZE = 93  # angular velocity + gravity + q + qdot + previous action
OBS_SIZE = 5 * PROPRIO_SIZE + 63 * 2 + 8 * 4 + 3 + 2 + 4
CRITIC_SIZE = OBS_SIZE + 29 * 2 + 3 + 4


def conservative_robot_cfg():
    cfg = ELF3LITE_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Robot"
    cfg.soft_joint_pos_limit_factor = 1.0
    xml = ET.parse(URDF).getroot()
    limits = {
        n.get("name"): n.find("limit") for n in xml.findall("joint") if n.find("limit") is not None
    }
    # Keep original gains, clamp only this copied config's effort/velocity limits.
    for actuator in cfg.actuators.values():
        for field, key in [("effort_limit_sim", "effort"), ("velocity_limit_sim", "velocity")]:
            original = getattr(actuator, field)
            values = {}
            for name, lim in limits.items():
                if any(re.fullmatch(pattern, name) for pattern in actuator.joint_names_expr):
                    value = (
                        next(v for pattern, v in original.items() if re.fullmatch(pattern, name))
                        if isinstance(original, dict)
                        else original
                    )
                    values[name] = min(float(lim.get(key)), float(value))
            setattr(actuator, field, values)
    return cfg


@configclass
class TrackerCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    episode_length_s = 2.0
    action_space = 29
    observation_space = OBS_SIZE
    state_space = CRITIC_SIZE
    sim = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.6, restitution=0.0, friction_combine_mode="min"
        ),
    )
    scene = InteractiveSceneCfg(num_envs=4, env_spacing=8.0, replicate_physics=True)
    robot_cfg = conservative_robot_cfg()
    reference_path: str = str(DEFAULT_PACK)
    segment: str = "flip"
    random_start: bool = False
    curriculum_enabled: bool = False
    reward_stage: str = "prior"
    end_hold_steps: int = 10
    extra_budget_steps: int = 50
    action_scale: float = 1.0
    # Noise is disabled in the first interface/physics checks, not claimed sim2real-ready.
    height_noise_std: float = 0.0
    is_finite_horizon = True


class TrackerEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        self.reference_np, self.reference_manifest = read_pack(cfg.reference_path)
        segments = json.loads(str(self.reference_np["segments_json"]))
        prepare = np.flatnonzero(
            np.isin(
                self.reference_np["phase"],
                [
                    "READY",
                    "GRIPPER_FLIP",
                    "PREPARED",
                    "WALK_LEFT_CLEAR",
                    "WALK_LEFT_LAND",
                    "WALK_RIGHT_CLEAR",
                    "APPROACH_READY",
                    "ARM_HOVER",
                    "FOREARMS",
                    "FOREARM_HOLD",
                ],
            )
        )
        segments["prepare"] = [int(prepare[0]), int(prepare[-1])]
        if cfg.segment not in segments:
            raise ValueError("Only flip/stand/prepare tracking experiments are enabled")
        self.segment_start, self.segment_end = segments[cfg.segment]
        cfg.episode_length_s = (
            self.segment_end - self.segment_start + 1 + cfg.extra_budget_steps
        ) * 0.02
        super().__init__(cfg, **kwargs)
        self.joint_ids, names = self.robot.find_joints(
            self.reference_np["joint_names"].tolist(), preserve_order=True
        )
        if names != self.reference_np["joint_names"].tolist() or self.robot.num_joints != 29:
            raise ValueError("ELF3 named joint mapping mismatch")
        if self.robot.body_names[0] != "torso_link":
            raise ValueError("Reference root is torso_link")
        self.site_ids = [
            self.robot.find_bodies(n, preserve_order=True)[0][0]
            for n in self.reference_np["site_bodies"].tolist()
        ]
        self.sensor_ids = [
            self.contacts.find_bodies(n, preserve_order=True)[0][0]
            for n in self.reference_np["site_bodies"].tolist()
        ]
        self.forbidden_sensor_ids = self.contacts.find_bodies(
            ["torso_link", "waist_z_link", "l_wrist_z_link", "r_wrist_z_link"]
        )[0]
        self.ref = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in self.reference_np.items()
            if v.dtype.kind in "fiu b".replace(" ", "")
        }
        self.starts = torch.full(
            (self.num_envs,), self.segment_start, device=self.device, dtype=torch.long
        )
        self.actions = torch.zeros((self.num_envs, 29), device=self.device)
        self.previous_actions = self.actions.clone()
        self.targets = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.history = torch.zeros((self.num_envs, 5, PROPRIO_SIZE), device=self.device)
        self.history_steps = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.nominal = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        self.grid = torch.cartesian_prod(
            torch.linspace(-0.8, 1.2, 9, device=self.device),
            torch.linspace(-0.6, 0.6, 7, device=self.device),
        )
        self.grid = torch.cat([self.grid, torch.zeros(63, 1, device=self.device)], dim=-1)
        self.last_metrics = {}
        self.completed = 0
        self.failed = 0
        self.successful = 0
        self.rules = GripperRules()
        self.curriculum = BackwardCurriculum()
        self.training_mode = False
        self.use_curriculum = False
        self.curriculum_events = []
        self.evaluation_start_phase = None
        self.ref_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_settle = self.ref_steps.clone()
        self.end_settle = self.ref_steps.clone()
        self.gate_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gate_allowed = self.gate_latched.clone()
        self.assumed_prerequisite = self.gate_latched.clone()
        self.reset_phase = torch.zeros(self.num_envs, device=self.device)
        self.wrist_joint = self.reference_np["joint_names"].tolist().index("l_wrist_y_joint")
        self.wrist_body = self.robot.find_bodies("l_wrist_z_link", preserve_order=True)[0][0]
        self.approach_barrier = int(
            np.flatnonzero(self.reference_np["phase"] == "WALK_LEFT_CLEAR")[0]
        )
        self.hold_mask = torch.as_tensor(
            np.isin(self.reference_np["phase"], ["FOREARM_HOLD"]), device=self.device
        )
        delta = (
            self.ref["root_pos"][
                int(np.flatnonzero(self.reference_np["phase"] == "ARM_HOVER")[0]), :2
            ]
            - self.ref["root_pos"][0, :2]
        )
        self.advance_direction = delta / torch.linalg.vector_norm(delta)
        gc = specification()
        self.gripper_center = torch.tensor(gc["center_in_wrist_m"], device=self.device)
        self.gripper_half = torch.tensor(gc["half_size_m"], device=self.device)
        self.gripper_mount_quat = torch.tensor(gc["mount_quaternion_wxyz"], device=self.device)
        self.target_rates = torch.full((29,), 6.0, device=self.device)
        self.target_rates[self.wrist_joint] = 1.5

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # Attach collision geometry and combine explicit inertials on the spawned
        # wrist only. Source USD files remain unchanged.
        from pxr import UsdPhysics

        stage = self.sim.stage
        matches = [
            p
            for p in stage.Traverse()
            if p.GetPath().pathString.startswith("/World/envs/env_0/Robot/")
            and p.GetName() == "l_wrist_z_link"
            and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot locate unique left wrist rigid body: {len(matches)}")
        gripper = specification()
        gc = sim_utils.CuboidCfg(
            size=tuple(np.array(gripper["half_size_m"]) * 2),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.05)),
        )
        gc.func(
            str(matches[0].GetPath()) + "/ClimbingGripperEnvelope",
            gc,
            translation=tuple(gripper["center_in_wrist_m"]),
            orientation=tuple(gripper["mount_quaternion_wxyz"]),
        )
        self.gripper_mass_report = apply_usd_mass(matches[0])
        ground = sim_utils.GroundPlaneCfg(physics_material=self.cfg.sim.physics_material)
        ground.func("/World/ground", ground)
        for i, (pos, quat, size) in enumerate(
            zip(
                self.reference_np["terrain_box_centers_w"],
                self.reference_np["terrain_box_quat_wxyz"],
                self.reference_np["terrain_box_half_sizes"],
            )
        ):
            # The height-query interface is deliberately restricted to yaw-only boxes.
            from scipy.spatial.transform import Rotation

            if not np.allclose(
                Rotation.from_quat(quat, scalar_first=True).as_matrix()[:, 2], [0, 0, 1], atol=1e-6
            ):
                raise ValueError("Tilted boxes need a different height-query implementation")
            bc = sim_utils.CuboidCfg(
                size=tuple(2 * size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0
                ),
                physics_material=self.cfg.sim.physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 0.7)),
            )
            bc.func(
                f"/World/envs/env_0/Platform_{i}",
                bc,
                translation=tuple(pos),
                orientation=tuple(quat),
            )
        self.contacts = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=4
            )
        )
        # Single-body filtered sensors distinguish terrain support from self-contact.
        self.support_sensors = {}
        for body in dict.fromkeys(self.reference_np["site_bodies"].tolist()):
            self.support_sensors[body] = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/" + body,
                    update_period=0.0,
                    history_length=1,
                    filter_prim_paths_expr=[
                        f"/World/envs/env_.*/Platform_{i}"
                        for i in range(len(self.reference_np["terrain_box_centers_w"]))
                    ],
                )
            )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contacts"] = self.contacts
        for body, sensor in self.support_sensors.items():
            self.scene.sensors["support_" + body] = sensor
        light = sim_utils.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)

    def frames(self):
        return (self.starts + self.ref_steps).clamp(max=self.segment_end)

    def site_positions(self):
        d = self.robot.data
        quat = d.body_link_quat_w[:, self.site_ids]
        off = self.ref["site_offsets"].expand(self.num_envs, -1, -1)
        return d.body_link_pos_w[:, self.site_ids] + quat_apply(
            quat.reshape(-1, 4), off.reshape(-1, 3)
        ).reshape(self.num_envs, 8, 3)

    def support_forces(self):
        forces = []
        for body in self.reference_np["site_bodies"].tolist():
            matrix = self.support_sensors[body].data.force_matrix_w
            if matrix is None:
                raise RuntimeError("Filtered terrain contact reporting unavailable")
            forces.append(matrix[:, 0].sum(dim=1))
        return torch.stack(forces, dim=1)

    def gripper_state(self):
        from isaaclab.utils.math import quat_mul

        d = self.robot.data
        wrist_q = d.body_link_quat_w[:, self.wrist_body]
        q = quat_mul(wrist_q, self.gripper_mount_quat.expand(self.num_envs, -1))
        center = (
            d.body_link_pos_w[:, self.wrist_body]
            - self.scene.env_origins
            + quat_apply(wrist_q, self.gripper_center.expand(self.num_envs, -1))
        )
        basis = torch.eye(3, device=self.device).expand(self.num_envs, -1, -1)
        axes = (
            quat_apply(q[:, None, :].expand(-1, 3, -1).reshape(-1, 4), basis.reshape(-1, 3))
            .reshape(self.num_envs, 3, 3)
            .transpose(1, 2)
        )
        margins = []
        for c, bq, h in zip(
            self.ref["terrain_box_centers_w"],
            self.ref["terrain_box_quat_wxyz"],
            self.ref["terrain_box_half_sizes"],
        ):
            ba = quat_apply(bq.expand(3, -1), torch.eye(3, device=self.device)).T
            margins.append(obb_separation(center, axes, self.gripper_half, c, ba, h))
        clearance = torch.stack(margins, dim=1).amin(dim=1)
        angle_error = d.joint_pos[:, self.joint_ids[self.wrist_joint]] - self.rules.target_angle
        velocity = d.joint_vel[:, self.joint_ids[self.wrist_joint]]
        up_angle = torch.acos(axes[:, 2, 0].clamp(-1, 1))
        return angle_error, velocity, clearance, up_angle

    def height_observation(self):
        """Ground-truth 2.5-D height query; NOT raycasting, LiDAR or neural mapping."""
        root = self.robot.data.root_link_pos_w
        yaw = yaw_quat(self.robot.data.root_link_quat_w)
        xyz = quat_apply(
            yaw[:, None, :].expand(-1, 63, -1).reshape(-1, 4),
            self.grid.expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, 63, 3)
        xyz += root[:, None, :] - self.scene.env_origins[:, None, :]
        heights = torch.zeros((self.num_envs, 63), device=self.device)
        for c, q, s in zip(
            self.ref["terrain_box_centers_w"],
            self.ref["terrain_box_quat_wxyz"],
            self.ref["terrain_box_half_sizes"],
        ):
            local = quat_apply_inverse(
                q.expand(self.num_envs * 63, -1), (xyz - c).reshape(-1, 3)
            ).reshape(self.num_envs, 63, 3)
            inside = (local[..., :2].abs() <= s[:2]).all(-1)
            heights = torch.where(inside, torch.maximum(heights, c[2] + s[2]), heights)
        heights -= (root[:, 2] - self.scene.env_origins[:, 2])[:, None]
        if self.cfg.height_noise_std:
            heights += torch.randn_like(heights) * self.cfg.height_noise_std
        return torch.cat([heights.clamp(-3, 3), torch.ones_like(heights)], dim=-1)

    def _get_observations(self):
        d = self.robot.data
        i = self.frames()
        yaw = yaw_quat(d.root_link_quat_w)
        prop = torch.cat(
            [
                d.root_ang_vel_b * 0.25,
                d.projected_gravity_b,
                d.joint_pos[:, self.joint_ids] - self.nominal,
                d.joint_vel[:, self.joint_ids] * 0.05,
                self.actions,
            ],
            dim=-1,
        )
        first = self.history_steps < 0
        advance = (self.history_steps != self.episode_length_buf) & ~first
        self.history[advance, :-1] = self.history[advance, 1:].clone()
        self.history[advance, -1] = prop[advance]
        self.history[first] = prop[first, None, :].expand(-1, 5, -1)
        self.history_steps[:] = self.episode_length_buf
        commands = self.ref["contact_targets"][i].clone()
        valid = self.ref["contact_valid"][i].clone()
        blocked = ~self.gate_allowed & (self.starts < self.approach_barrier)
        commands[blocked] = 0
        commands[blocked, :2] = self.ref["site_pos"][self.segment_start, :2]
        valid[blocked] = 0
        valid[blocked, :2] = 1
        target = commands + self.scene.env_origins[:, None, :] - d.root_link_pos_w[:, None, :]
        target = quat_apply_inverse(
            yaw[:, None, :].expand(-1, 8, -1).reshape(-1, 4), target.reshape(-1, 3)
        ).reshape(self.num_envs, 8, 3)
        target = target * valid[..., None]
        goal = self.ref["root_pos"][self.segment_end] + self.scene.env_origins - d.root_link_pos_w
        gate_goal = (
            self.ref["root_pos"][min(self.approach_barrier - 1, self.segment_end)]
            + self.scene.env_origins
            - d.root_link_pos_w
        )
        goal = torch.where(blocked[:, None], gate_goal, goal)
        goal = quat_apply_inverse(yaw, goal)
        _, _, clearance, _ = self.gripper_state()
        protocol_obs = torch.stack(
            [
                self.gate_allowed.float(),
                (self.gate_settle.float() / self.rules.settle_steps).clamp(max=1),
                clearance.clamp(-1, 1),
                (i >= self.approach_barrier).float(),
            ],
            dim=-1,
        )
        # Gripper clearance is a task command: contacts alone do not specify wrist roll.
        # Fixed per task, later supplied by the FSM; not a reference-pose observation.
        grip = torch.tensor([-1.309, 1.0], device=self.device).expand(self.num_envs, -1)
        actor = torch.cat(
            [
                self.history.flatten(1),
                self.height_observation(),
                target.flatten(1),
                valid,
                goal,
                grip,
                protocol_obs,
            ],
            dim=-1,
        )
        # Privileged reference information only in critic/rewards, not actor/action baseline.
        critic = torch.cat(
            [
                actor,
                self.ref["joint_pos"][i],
                self.ref["joint_vel"][i] * 0.05,
                self.ref["root_pos"][i] + self.scene.env_origins - d.root_link_pos_w,
                self.ref["root_quat"][i],
            ],
            dim=-1,
        )
        return {"policy": actor, "critic": critic}

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite action")
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-3.0, 3.0))
        desired = torch.clamp(
            self.nominal + self.cfg.action_scale * self.actions,
            self.limits[..., 0],
            self.limits[..., 1],
        )
        delta = (desired - self.targets).clamp(
            -self.target_rates * self.step_dt, self.target_rates * self.step_dt
        )
        self.targets.copy_((self.targets + delta).clamp(self.limits[..., 0], self.limits[..., 1]))

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def _measure(self):
        i = self.frames()
        d = self.robot.data
        rooterr = torch.linalg.vector_norm(
            d.root_link_pos_w - self.scene.env_origins - self.ref["root_pos"][i], dim=-1
        )
        dot = (d.root_link_quat_w * self.ref["root_quat"][i]).sum(-1).abs().clamp(0, 1)
        angle = 2 * torch.acos(dot)
        qerr = (d.joint_pos[:, self.joint_ids] - self.ref["joint_pos"][i]).square().mean(-1)
        return i, rooterr, angle, qerr

    def _get_dones(self):
        d = self.robot.data
        ge, gv, clearance, up = self.gripper_state()
        settle, latched, allowed, ready = advance_gate(
            ge, gv, clearance, self.gate_settle, self.gate_latched, self.rules
        )
        self.gate_settle.copy_(settle)
        self.gate_latched.copy_(latched)
        self.gate_allowed.copy_(allowed)
        next_frame = self.starts + self.ref_steps + 1
        blocked = (
            (next_frame >= self.approach_barrier)
            & ~self.gate_allowed
            & (self.starts < self.approach_barrier)
        )
        self.ref_steps += (~blocked).long()
        i, pos, angle, qerr = self._measure()
        forbidden = (
            torch.linalg.vector_norm(
                self.contacts.data.net_forces_w[:, self.forbidden_sensor_ids], dim=-1
            ).amax(-1)
            > 100.0
        )
        local_root = d.root_link_pos_w - self.scene.env_origins
        feet = self.site_positions()[:, :2] - self.scene.env_origins[:, None, :]
        premature, foot_advance = premature_step(
            feet,
            self.ref["site_pos"][self.starts, :2],
            self.advance_direction,
            self.gate_latched,
            self.starts < self.approach_barrier,
            self.rules,
        )
        collision = clearance < -self.rules.penetration_tolerance
        lost_flip = self.gate_latched & ~self.gate_allowed
        failure = (
            (pos > 0.45)
            | (angle > 1.1)
            | (qerr > 1.0)
            | forbidden
            | premature
            | collision
            | lost_flip
        )
        forces = torch.linalg.vector_norm(self.support_forces(), dim=-1)
        early_arm = (forces[:, 2:4] > 3).any(-1) & ~self.gate_latched
        failure = failure | early_arm
        support = (forces[:, :2] > 5).all(-1)
        if self.cfg.segment == "prepare":
            point_error = torch.linalg.vector_norm(
                self.site_positions()[:, 2:6]
                - self.scene.env_origins[:, None, :]
                - self.ref["site_pos"][i, 2:6],
                dim=-1,
            )
            support = (
                support & (forces[:, 2:4] > 3).all(-1) & (point_error < 0.05).all(-1) & (up < 0.524)
            )
        near_end = (
            (i >= self.segment_end)
            & ~failure
            & (pos < 0.15)
            & (angle < 0.4)
            & (qerr < 0.12)
            & support
            & ready
        )
        self.end_settle.copy_(
            torch.where(near_end, self.end_settle + 1, torch.zeros_like(self.end_settle))
        )
        success = self.end_settle >= self.cfg.end_hold_steps
        expired = self.episode_length_buf >= (
            self.segment_end - self.starts + self.cfg.extra_budget_steps
        )
        timeout = success | expired
        done = failure | timeout
        if self.training_mode and self.cfg.curriculum_enabled and done.any():
            previous = self.curriculum.phase
            self.curriculum.observe(success[done].tolist(), self.reset_phase[done].tolist())
            if previous != self.curriculum.phase:
                self.curriculum_events.append(self.curriculum.state())
        self.completed += int(done.sum())
        self.failed += int((done & ~success).sum())
        self.successful += int(success.sum())
        self.last_metrics = {
            "root_error_m": pos.detach().clone(),
            "orientation_error_rad": angle.detach().clone(),
            "joint_rmse_rad": qerr.sqrt().detach().clone(),
            "success": success.detach().clone(),
            "foot_force_N": forces[:, :2].detach().clone(),
            "failure_orientation": (angle > 1.1).detach().clone(),
            "failure_root": (pos > 0.45).detach().clone(),
            "failure_forbidden_contact": forbidden.detach().clone(),
            "premature_advance": premature.detach().clone(),
            "premature_arm_contact": early_arm.detach().clone(),
            "foot_advance_m": foot_advance.detach().clone(),
            "gripper_collision": collision.detach().clone(),
            "lost_flip": lost_flip.detach().clone(),
            "gripper_error_rad": ge.abs().detach().clone(),
            "gripper_speed_rad_s": gv.abs().detach().clone(),
            "gripper_clearance_m": clearance.detach().clone(),
            "gate_allowed": self.gate_allowed.detach().clone(),
            "assumed_prerequisite": self.assumed_prerequisite.detach().clone(),
            "reset_frame": self.starts.detach().clone(),
            "reference_frame": i.detach().clone(),
            "root_pose": torch.cat([local_root, d.root_link_quat_w], dim=-1).detach().clone(),
            "joint_pos": d.joint_pos[:, self.joint_ids].detach().clone(),
            "joint_vel": d.joint_vel[:, self.joint_ids].detach().clone(),
            "torque_ratio": (
                d.applied_torque[:, self.joint_ids].abs()
                / d.joint_effort_limits[:, self.joint_ids].clamp(min=1e-6)
            )
            .amax(-1)
            .detach()
            .clone(),
        }
        return failure, timeout

    def _get_rewards(self):
        i, pos, angle, qerr = self._measure()
        d = self.robot.data
        qvel = (d.joint_vel[:, self.joint_ids] - self.ref["joint_vel"][i]).square().mean(-1)
        mask = self.ref["contact_mask"][i]
        distance = (
            (self.site_positions() - self.scene.env_origins[:, None, :] - self.ref["site_pos"][i])
            .square()
            .sum(-1)
        )
        # Body-level force detection cannot distinguish two contact sites on one forearm.
        force = torch.linalg.vector_norm(self.support_forces(), dim=-1)
        contact = (torch.exp(-distance / 0.0025) * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        actual = ((force > 3).float() * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        smooth = (self.actions - self.previous_actions).square().mean(-1)
        torque = d.applied_torque[:, self.joint_ids].square().mean(-1)
        ge, gv, clearance, up = self.gripper_state()
        imitation_weight = 1.0 if self.cfg.reward_stage == "prior" else 0.35
        gripper_score = (1 - ge.abs() / 1.309).clamp(0, 1)
        forearm_orientation = torch.exp(-up.square() / 0.12) * self.hold_mask[i].float()
        reward = (
            imitation_weight * torch.exp(-qerr / 0.15)
            + torch.exp(-pos.square() / 0.04)
            + 0.5 * torch.exp(-angle.square() / 0.25)
            + imitation_weight * 0.1 * torch.exp(-qvel / 10)
            + 1.5 * gripper_score
            + 0.3 * torch.exp(-ge.square() / 0.01 - gv.square() / 0.0625)
            + 0.3 * forearm_orientation
            + contact
            + 0.3 * actual
            - 0.02 * smooth
            - 0.00005 * torque
        ) * self.step_dt
        remaining = (
            self.segment_end - self.starts + self.cfg.extra_budget_steps - self.episode_length_buf
        )
        reward += (
            completion_bonus(remaining, self.step_dt) * self.last_metrics["success"].float()
            - self.reset_terminated.float()
        )
        reward -= 0.5 * (self.reset_time_outs & ~self.last_metrics["success"]).float()
        self.extras["log"] = {
            "Tracker/root_error_m": pos.mean(),
            "Tracker/joint_rmse_rad": qerr.sqrt().mean(),
            "Tracker/reference_contact_score": contact.mean(),
            "Tracker/gripper_error_rad": ge.abs().mean(),
            "Tracker/gate_ready_fraction": self.gate_allowed.float().mean(),
            "Tracker/gripper_clearance_min_m": clearance.min(),
            "Curriculum/initial_phase": torch.tensor(self.curriculum.phase, device=self.device),
        }
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        if self.evaluation_start_phase is not None:
            phase = self.evaluation_start_phase
            self.starts[env_ids] = self.segment_start + round(
                phase * (self.segment_end - self.segment_start)
            )
            self.reset_phase[env_ids] = phase
        elif self.use_curriculum and self.cfg.curriculum_enabled:
            phase = self.curriculum.phase
            self.starts[env_ids] = self.segment_start + round(
                phase * (self.segment_end - self.segment_start)
            )
            self.reset_phase[env_ids] = phase
        elif self.cfg.random_start:
            self.starts[env_ids] = torch.randint(
                self.segment_start,
                max(self.segment_start + 1, self.segment_end - 10),
                (n,),
                device=self.device,
            )
            self.reset_phase[env_ids] = (self.starts[env_ids] - self.segment_start) / max(
                1, self.segment_end - self.segment_start
            )
        else:
            self.starts[env_ids] = self.segment_start
            self.reset_phase[env_ids] = 0
        self.ref_steps[env_ids] = 0
        self.gate_settle[env_ids] = 0
        self.end_settle[env_ids] = 0
        self.assumed_prerequisite[env_ids] = self.starts[env_ids] >= self.approach_barrier
        self.gate_latched[env_ids] = self.assumed_prerequisite[env_ids]
        self.gate_allowed[env_ids] = self.assumed_prerequisite[env_ids]
        i = self.starts[env_ids]
        pose = torch.cat(
            [self.ref["root_pos"][i] + self.scene.env_origins[env_ids], self.ref["root_quat"][i]],
            dim=-1,
        )
        # Isaac velocity setter expects root COM linear velocity; reference is link origin.
        offset = quat_apply(pose[:, 3:7], self.robot.data.body_com_pos_b[env_ids, 0])
        angular = self.ref["root_ang_vel"][i]
        velocity = torch.cat(
            [self.ref["root_lin_vel"][i] + torch.cross(angular, offset, dim=-1), angular], dim=-1
        )
        q = self.robot.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        q[:, self.joint_ids] = self.ref["joint_pos"][i]
        qd[:, self.joint_ids] = self.ref["joint_vel"][i]
        self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.targets[env_ids] = self.ref["joint_pos"][i]
        self.robot.set_joint_position_target(
            self.targets[env_ids], joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.history_steps[env_ids] = -1

    def reference_pd_action(self):
        """Diagnostic only. Policy actions do not use this reference baseline."""
        i = (self.frames() + 1).clamp(max=self.segment_end)
        return (self.ref["joint_pos"][i] - self.nominal) / self.cfg.action_scale
