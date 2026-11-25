"""Continue a frozen knee-entry policy in the same PhysX scene through a full ascent."""

import torch

from .environment import TrackerEnv
from .mount_control import MountControl
from .mount_task import MountTask


class MountEnv(TrackerEnv):
    """Run the original entry policy and hand control over at its first accepted knee hold."""

    mount_duration_s = 9.0
    extra_observations = 12

    def __init__(self, cfg, **kwargs):
        if cfg.segment != "knee_entry" or not cfg.knee_transfer_guide:
            raise ValueError("Mount requires a guided knee-entry environment")
        cfg.observation_space += self.extra_observations
        cfg.state_space += self.extra_observations
        super().__init__(cfg, **kwargs)
        self.mount_steps = round(self.mount_duration_s / self.step_dt)
        self.cfg.episode_length_s += self.mount_duration_s
        self.entry_policy = None
        self.mount_active = torch.zeros_like(self.gate_latched)
        self.mount_age = torch.zeros_like(self.starts)
        self.mount_residual = torch.zeros_like(self.actions)
        self.mount_obs = torch.zeros(self.num_envs, self.extra_observations, device=self.device)
        self.knee_contacts.enable_pose_trace()
        self.mount_task = MountTask(self)
        self.mount_control = MountControl(self)

    def _get_observations(self):
        obs = super()._get_observations()
        self.entry_obs = obs["policy"]
        return {key: torch.cat((value, self.mount_obs), dim=-1) for key, value in obs.items()}

    def _pre_physics_step(self, actions):
        if self.entry_policy is None:
            raise RuntimeError("Attach the frozen knee-entry policy before stepping MountEnv")
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite mount residual")
        self.mount_residual.copy_(actions)
        with torch.no_grad():
            entry_actions = self.entry_policy(self.entry_obs)
        super()._pre_physics_step(entry_actions)

    def _joint_command(self):
        entry = super()._joint_command()
        if not self.mount_active.any():
            return entry
        return self.mount_control.desired(entry, self.mount_residual, self.mount_active, self.mount_age)

    def _get_dones(self):
        entry_failed, _ = super()._get_dones()
        metrics = self.last_metrics
        was_active = self.mount_active.clone()
        handoff = ~was_active & metrics["success"] & ~entry_failed
        self.mount_age += was_active.long()
        self.mount_active |= handoff

        outcome = self.mount_task.measure()
        metrics.update(outcome)
        metrics["entry_accepted"] = handoff
        metrics["mount_active"] = self.mount_active.clone()
        metrics["mount_age"] = self.mount_age.clone()
        metrics["success"] = was_active & outcome["success"]
        # Pose and knee-contact order gates apply to entry only. In recovery the
        # robot may roll, unload its upper limbs or bring either leg up first.
        root_height = metrics["root_pose"][:, 2] - self.knee_contacts.floor_z
        fallen = root_height < 0.26
        failed = torch.where(was_active, fallen, entry_failed)
        entry_budget = self.segment_end - self.starts + self.cfg.extra_budget_steps
        expired = torch.where(self.mount_active, self.mount_age >= self.mount_steps,
                              self.episode_length_buf >= entry_budget)
        metrics.update(terminated=failed, expired=expired)
        self.mount_obs.copy_(torch.cat((
            self.mount_active[:, None].float(),
            (self.mount_age / self.mount_steps)[:, None],
            outcome["stand_hold_s"][:, None],
            outcome["foot_inside_edge_m"], outcome["knee_inside_edge_m"],
            (outcome["foot_top_N"] / 200).clamp(0, 3),
            (outcome["knee_top_N"] / 200).clamp(0, 3),
            outcome["com_inside_edge_m"][:, None],
        ), dim=1) * self.mount_active[:, None])
        return failed, expired | metrics["success"]

    def _get_rewards(self):
        m = self.last_metrics
        return torch.where(
            m["mount_active"], m["reward"] - 2.0 * m["terminated"],
            torch.zeros_like(m["reward"]),
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "mount_task"):
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            self.mount_active[env_ids] = False
            self.mount_age[env_ids] = 0
            self.mount_obs[env_ids] = 0
            self.mount_residual[env_ids] = 0
            self.mount_task.reset(env_ids)
