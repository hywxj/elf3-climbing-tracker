"""Flat, proprioceptive ELF3 walking extension of the local TienKung-Lab task.

Import only after AppLauncher. No modification to the shared project or Hiking.
This stage starts with the wrist already flipped; skill handover is separate.
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from legged_lab.envs.elf3.elf3_env import Elf3Env
from legged_lab.envs.elf3.walk_cfg import Elf3WalkFlatEnvCfg, Elf3WalkAgentCfg
from legged_lab.motion.elf3_contract import JOINT_NAMES
import legged_lab.mdp as mdp

from .gripper import specification, apply_usd_mass
from .robot_cfg import ELF3LITE_CFG
from .walk_data import WRIST_NAME, WRIST_TARGET, mask_wrist


def spawn_with_gripper(prim_path, cfg, **kwargs):
    prim = sim_utils.spawn_from_usd(prim_path, cfg, **kwargs)
    from pxr import UsdPhysics
    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    wrists = [
        p
        for p in stage.Traverse()
        if p.GetName() == "l_wrist_z_link"
        and p.HasAPI(UsdPhysics.RigidBodyAPI)
        and p.GetPath().pathString.startswith("/World/envs/")
    ]
    if not wrists:
        raise ValueError("No left wrist found for gripper geometry")
    g = specification()
    if g["mass_kg"] != 0:
        raise ValueError("This motion-learning pilot requires zero added gripper mass")
    cube = sim_utils.CuboidCfg(
        size=tuple(2 * x for x in g["half_size_m"]),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.05)),
    )
    for wrist in wrists:
        cube.func(
            str(wrist.GetPath()) + "/ClimbingGripperEnvelope",
            cube,
            translation=tuple(g["center_in_wrist_m"]),
            orientation=tuple(g["mount_quaternion_wxyz"]),
        )
        apply_usd_mass(wrist)
    return prim


def velocity_reward(env):
    error = (env.robot.data.root_lin_vel_b[:, :2] - env.command_generator.command[:, :2]).norm(
        dim=-1
    )
    # A small nonzero error is acceptable; no equality-to-zero success condition.
    return torch.exp(-((error - 0.05).clamp_min(0) / 0.20).square())


def wrist_reward(env):
    i = env.robot.joint_names.index(WRIST_NAME)
    return torch.exp(-((env.robot.data.joint_pos[:, i] - WRIST_TARGET) / 0.10).square())


def stop_motion(env):
    stop = env.command_generator.command.norm(dim=-1) < 0.05
    return stop * env.robot.data.joint_vel.square().mean(dim=-1)


@configclass
class WalkRewards:
    velocity = RewTerm(func=velocity_reward, weight=5.0)
    yaw_velocity = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.4})
    wrist = RewTerm(func=wrist_reward, weight=1.5)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    angular_motion = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    vertical_motion = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-4)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x.*"),
        },
    )
    air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
        },
    )
    stop_motion = RewTerm(func=stop_motion, weight=-0.02)
    failure = RewTerm(func=mdp.is_terminated, weight=-100.0)


def make_configs(repo, num_envs, device, seed, manifest):
    cfg, agent = Elf3WalkFlatEnvCfg(), Elf3WalkAgentCfg()
    cfg.device = agent.device = device
    cfg.scene.seed = agent.seed = seed
    cfg.scene.robot = ELF3LITE_CFG.copy()
    # The required wrist angle is at its physical lower limit. The parent's
    # 90% soft range would clip reset poses back by ~0.131 rad and penalize the
    # desired target. Use actual limits, as in the accepted flip environment.
    cfg.scene.robot.soft_joint_pos_limit_factor = 1.0
    cfg.scene.robot.spawn.func = spawn_with_gripper
    cfg.scene.robot.init_state.joint_pos[WRIST_NAME] = WRIST_TARGET
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 4.0
    cfg.scene.max_episode_length_s = 10.0
    cfg.scene.terrain_type, cfg.scene.terrain_generator = "plane", None
    cfg.scene.contact_history_length = 4
    cfg.scene.height_scanner.enable_height_scan = False
    cfg.scene.depth_camera.enable_depth_camera = False
    cfg.scene.depth_camera.geometry.enabled = False
    cfg.scene.depth_camera.geometry.step_control_enabled = False
    cfg.scene.depth_camera.geometry.bootstrap_stairs = False
    cfg.scene.lidar.enable_lidar = False
    cfg.robot.terminate_contacts_body_names = ["^(?![lr]_ankle_x_link$).*"]
    cfg.commands.ranges.lin_vel_x = (0.10, 0.30)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.heading_command = False
    cfg.commands.rel_standing_envs = 0.25
    cfg.commands.resampling_time_range = (2.0, 4.0)
    cfg.commands.debug_vis = False
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "yaw": (-0.03, 0.03),
    }
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
    cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (0.99, 1.01)
    cfg.domain_rand.action_delay.enable = False
    cfg.normalization.clip_actions = 3.0
    cfg.noise.noise_scales.joint_vel = 0.2
    cfg.reward = WalkRewards()
    cfg.amp_motion_files_display = [
        str(repo / "legged_lab/envs/elf3/datasets/motion_visualization/walk.txt")
    ]
    agent.amp_motion_manifest = str(manifest)
    agent.amp_reward_time_scaled = True
    agent.amp_motion_mask_enabled = True
    agent.amp_reward_coef = 2.0
    agent.amp_task_reward_lerp = 0.7
    agent.amp_discr_hidden_dims = [256, 128]
    agent.amp_num_preload_transitions = 0
    agent.policy.actor_hidden_dims = [256, 128]
    agent.policy.critic_hidden_dims = [256, 128]
    agent.policy.init_noise_std = 0.3
    agent.algorithm.learning_rate = 3.0e-4
    agent.algorithm.entropy_coef = 0.001
    agent.algorithm.num_learning_epochs = 3
    agent.save_interval = 100
    agent.experiment_name = "elf3_blind_walk_flip"
    return cfg, agent


class BlindWalkEnv(Elf3Env):
    def init_buffers(self):
        # Reorder every scale by name, never by incidental PhysX articulation order.
        scales = dict(zip(JOINT_NAMES, self.cfg.robot.action_scale, strict=True))
        scales[WRIST_NAME] = 0.10
        if set(self.robot.joint_names) != set(scales):
            raise ValueError("ELF3 joint contract mismatch")
        self.cfg.robot.action_scale = [scales[n] for n in self.robot.joint_names]
        super().init_buffers()
        self.walk_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.walk_speed_error = torch.zeros(self.num_envs, device=self.device)
        self.walk_wrist_error = torch.zeros_like(self.walk_speed_error)
        self.walk_steps = torch.zeros_like(self.walk_speed_error)

    def get_amp_obs_for_expert_trans(self):
        return mask_wrist(super().get_amp_obs_for_expert_trans())

    def check_reset(self):
        if hasattr(self, "eval_command"):
            self.command_generator.command[:] = 0
            self.command_generator.command[:, 0] = self.eval_command
        data = self.robot.data
        force = (
            self.contact_sensor.data.net_forces_w_history[
                :, :, self.termination_contact_cfg.body_ids
            ]
            .norm(dim=-1)
            .amax(dim=(1, 2))
        )
        low = data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] < 0.65
        tilted = -data.projected_gravity_b[:, 2] < 0.7
        finite = torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.root_state_w).all(-1)
        self.walk_failure = low | tilted | (force > 40.0) | ~finite
        timeout = self.episode_length_buf >= self.max_episode_length
        self.walk_speed_error += (
            data.root_lin_vel_b[:, :2] - self.command_generator.command[:, :2]
        ).norm(dim=-1)
        self.walk_wrist_error += (
            data.joint_pos[:, self.robot.joint_names.index(WRIST_NAME)] - WRIST_TARGET
        ).abs()
        self.walk_steps += 1
        return self.walk_failure | timeout, timeout

    def reset(self, env_ids):
        done = env_ids[self.episode_length_buf[env_ids] > 0]
        stats = {}
        if len(done):
            stats = {
                "Walk/failure_fraction": self.walk_failure[done].float().mean(),
                "Walk/episode_seconds": self.episode_length_buf[done].float().mean() * self.step_dt,
                "Walk/velocity_error_m_s": (
                    self.walk_speed_error[done] / self.walk_steps[done].clamp_min(1)
                ).mean(),
                "Walk/wrist_error_rad": (
                    self.walk_wrist_error[done] / self.walk_steps[done].clamp_min(1)
                ).mean(),
            }
        super().reset(env_ids)
        self.extras["log"].update(stats)
        for name in ("walk_speed_error", "walk_wrist_error", "walk_steps"):
            getattr(self, name)[env_ids] = 0
