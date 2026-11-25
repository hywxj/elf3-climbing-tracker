"""One 29-joint student controls the continuous ground-to-right-knee task in PhysX."""

from pathlib import Path

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .contact_examples import student_contact_state
from .contact_tracker_task import (
    ContactMilestones,
    FrontApproach,
    accepted_knee,
    direct_joint_targets,
    ground_reset_offsets,
)
from .environment import TrackerEnv
from .walk import Footsteps


def training_start_episodes(paths, cloning_split_paths=None):
    """Exclude whole trajectories held out by PPO replay or the actor's original cloning split."""
    from .contact_learning import _episode_batches

    catalogs = {}
    excluded = set()
    splits = [tuple(Path(p).resolve() for p in paths)]
    if cloning_split_paths:
        splits.append(tuple(Path(p).resolve() for p in cloning_split_paths))
    for manifest in dict.fromkeys(splits):
        grouped = {}
        for path in manifest:
            if path not in catalogs:
                catalogs[path] = [
                    (source, fingerprint, meta["episode"])
                    for source, fingerprint, _, meta in _episode_batches(path)[2]
                ]
            for source, fingerprint, _ in catalogs[path]:
                grouped.setdefault(source, {})[fingerprint] = None
        # Match load_examples()' default replay split, including its deterministic
        # source order and grouping of identical replicas.
        rng = np.random.default_rng(42)
        for source, members in grouped.items():
            keys = list(members)
            rng.shuffle(keys)
            count = min(len(keys) - 1, max(1, round(len(keys) * 0.2)))
            excluded.update((source, key) for key in keys[:count])
    return {
        path: {int(episode) for source, key, episode in catalogs[path] if (source, key) not in excluded}
        for path in splits[0]
    }


def grounded_prelaunch_rows(pack, allowed):
    """Measured PRE-step indices from the first verified four-support brace to the actual lift."""
    ids, steps = pack["episode_id"], pack["step"]
    force, centers = pack["student_contact_force"], pack["student_contact_point"]
    floor = (pack["terrain_box_centers_w"][:, 2] + pack["terrain_box_half_sizes"][:, 2]).min()
    grounded = (
        pack["contact_observed"]
        & (force[:, :2] > 12).all(1)
        & (np.abs(centers[:, :2, 2] - floor) < 0.012).all(1)
        & (force[:, 7] < 3)
    )
    brace, lift = pack["post_brace"], pack["post_takeoff"]
    pushed, bearing = pack["post_bilateral_push"], pack["post_knee_hold_steps"]
    walked, advance = pack["post_walk_switches"], pack["post_foot_advance_m"]
    dt = float(pack["step_dt"])
    rows = {}
    for episode in sorted(allowed):
        idx = np.flatnonzero(ids == episode)
        idx = idx[np.argsort(steps[idx])]
        if not len(idx) or not (np.diff(steps[idx]) == 1).all():
            continue
        dock = np.flatnonzero(brace[idx] & (walked[idx] >= 1) & (advance[idx] >= 0.25))
        airborne = np.flatnonzero(lift[idx])
        if not len(dock) or not len(airborne):
            continue
        start, end = int(dock[0]), int(airborne[0])
        if (end <= start or not pushed[idx[start:end + 1]].any()
                or not (bearing[idx[end:]] * dt >= 0.20 - 1e-6).any()):
            continue
        # Event flags describe the result of a motor command. The first row that
        # observes the established brace is the following pre-command state.
        span = idx[start + 1:end + 1]
        eligible = span[grounded[span]]
        if len(eligible):
            rows[episode] = eligible
    return rows


def student_physical_state(env, site_pos, contact_force):
    """State measured before the next command. Shared with the demonstration recorder."""
    d = env.robot.data
    return {
        "joint_pos": d.joint_pos[:, env.joint_ids],
        "joint_vel": d.joint_vel[:, env.joint_ids],
        "prev_target": env.targets,
        "root_pos": d.root_link_pos_w - env.scene.env_origins,
        "root_quat": d.root_link_quat_w,
        "root_lin_vel_b": quat_apply_inverse(d.root_link_quat_w, d.root_link_lin_vel_w),
        "root_ang_vel_b": d.root_ang_vel_b,
        "projected_gravity_b": d.projected_gravity_b,
        "terrain": env.height_observation(),
        "site_pos": site_pos,
        "contact_force": contact_force,
    }


class ContactTrackerEnv(TrackerEnv):
    """The sole action entry point accepts absolute joint targets relative to the fixed nominal."""

    def __init__(self, cfg, *, edge_observation=False, point_observation=False, **kwargs):
        from .contact_learning import CONTACT_OBS_DIM

        self.edge_observation = edge_observation
        self.point_observation = point_observation
        observation_dim = CONTACT_OBS_DIM
        if point_observation:
            from .contact_point_actor import POINT_OBS_DIM

            observation_dim = POINT_OBS_DIM
        elif edge_observation:
            from .contact_edge import EDGE_OBS_DIM

            observation_dim = EDGE_OBS_DIM

        if cfg.segment != "knee_entry" or cfg.preload_example or cfg.knee_transfer_guide:
            raise ValueError("Use knee_entry geometry without a motion controller or preload guide")
        if cfg.knee_wrist_contact != "tabletop" or cfg.knee_support_rule != "assisted-push":
            raise ValueError(
                "Use measured tabletop contacts and the assisted-push contact criteria"
            )
        cfg.observation_space = cfg.state_space = observation_dim
        cfg.random_start = cfg.curriculum_enabled = False
        super().__init__(cfg, **kwargs)
        if self.point_observation:
            from .contact_point_actor import PointHistory

            self.point_history = PointHistory(self.num_envs, self.device)
        self.knee_contacts.enable_pose_trace()
        self.body_weight_N = self.knee_contacts.total_mass * abs(cfg.sim.gravity[2])
        from .contact_plan_reward import ContactPlanReward

        self.plan_score = ContactPlanReward(self.num_envs, self.device, self.step_dt)
        self.has_plan = False
        self.cfg.episode_length_s = 12.0
        self.stage_training = False
        self.demonstration_starts = []
        self.prelaunch_starts = []
        self.diagnostic_start = None
        self.target_rates.fill_(12.0)
        self.target_rates.clamp_(max=self.robot.data.joint_vel_limits[0, self.joint_ids])
        self.target_rates[self.wrist_joint] = 1.5
        self.milestones = ContactMilestones(self.num_envs, self.device, self.step_dt)
        self.footsteps = Footsteps(self.num_envs, self.device)
        upper = self.knee_contacts.upper
        self.front_approach = FrontApproach(
            self.num_envs,
            self.ref["terrain_box_centers_w"][upper],
            self.ref["terrain_box_half_sizes"][upper],
            self.ref["terrain_box_quat_wxyz"][upper],
            self.ref["root_pos"][0, :2],
        )
        self.initial_foot_forward = torch.zeros_like(self.actions[:, 0])
        self.reset_count = torch.zeros_like(self.starts)
        self.ground_start = torch.zeros_like(self.starts, dtype=torch.bool)
        self.prelaunch_start = torch.zeros_like(self.ground_start)
        self.ground_reset_offset_xy = self.actions.new_zeros((self.num_envs, 2))
        self.progress_best = self.actions.new_zeros((self.num_envs, 6))
        self.first_sample = torch.ones_like(self.ground_start)
        self._sensor_frame = torch.full_like(self.starts, self.segment_end)
        self.nominal_absolute = self.nominal[0].detach().clone()
        self.walk_start_frame = int(
            (self.reference_np["phase"] == "WALK_LEFT_CLEAR").nonzero()[0][0]
        )
        self.arm_approach_frame = int((self.reference_np["phase"] == "ARM_HOVER").nonzero()[0][0])

    @property
    def contact_frame(self):
        # Diagnostic reference index; never consulted to act or to accept contact.
        return (self.starts + self.episode_length_buf).clamp(max=self.segment_end)

    def raw_state(self):
        sites, force = student_contact_state(self)
        force = torch.where(self.episode_length_buf[:, None] == 0, 0.0, force)
        return student_physical_state(self, sites, force)

    def _get_observations(self):
        from .contact_learning import encode_contact_observation

        raw = self.raw_state()
        obs = encode_contact_observation(raw, self.nominal_absolute)
        if self.point_observation:
            obs = self.point_history.encode(
                obs,
                raw,
                self.ref["terrain_box_centers_w"],
                self.ref["terrain_box_half_sizes"],
                self.ref["terrain_box_quat_wxyz"],
                self.common_step_counter,
            )
        elif self.edge_observation:
            from .contact_edge import simulated_edge_features

            edge = simulated_edge_features(
                raw["root_pos"],
                raw["root_quat"],
                self.ref["terrain_box_centers_w"],
                self.ref["terrain_box_half_sizes"],
                self.ref["terrain_box_quat_wxyz"],
                version=int(self.edge_observation),
            )
            obs = torch.cat((obs, edge), dim=1)
        return {"policy": obs, "critic": obs}

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite student action")
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        # Exactly one learned action for each of the 29 joints, including the gripper.
        # Limits and slew bounds are fixed actuator constraints shared across all poses.
        self.targets.copy_(
            direct_joint_targets(
                actions, self.nominal, self.targets, self.limits, self.target_rates, self.step_dt
            )
        )

    def commit_plan(self, logits, points, *, yaw_frame):
        root = self.robot.data.root_link_pos_w - self.scene.env_origins
        quat = self.robot.data.root_link_quat_w
        if yaw_frame:
            quat = yaw_quat(quat)
        world = (
            quat_apply(
                quat[:, None, :].expand(-1, 8, -1).reshape(-1, 4), points.reshape(-1, 3)
            ).reshape_as(points)
            + root[:, None, :]
        )
        self.plan_score.propose(logits > 0, world)
        self.has_plan = True

    def _get_dones(self):
        d, k = self.robot.data, self.knee_contacts
        self.ref_steps += 1
        sites = self.site_positions() - self.scene.env_origins[:, None]
        feet = sites[:, :2]
        # Use the finished reference index solely for the existing contact calculator's
        # geometry. A physical push at any time is eligible; it need not match a demo clock.
        k.measure(
            self._sensor_frame, feet, require_reference_goal=False, recover_supported_lift=True
        )
        m = dict(k.metrics)
        foot_load = m["foot_ground_patch_N"]
        feet_forward = feet[..., :2] @ self.advance_direction
        self.initial_foot_forward.copy_(
            torch.where(self.first_sample, feet_forward.mean(1), self.initial_foot_forward)
        )
        advance = feet_forward.mean(1) - self.initial_foot_forward
        switched = self.footsteps.update(
            ~self.milestones.braced, foot_load, feet[:, :, 2] - k.floor_z, feet_forward
        )
        front_gain, rewarded_switch, front_progress, near_frontier = self.front_approach.update(
            feet[:, :, :2], self.first_sample, switched
        )
        four_point = (
            (m["arm_support_top_N"] > 3).all(1) & (foot_load > 8).all(1) & ~m["wrist_forbidden"]
        )
        pivot = sites[:, 7] - self.ref["terrain_box_centers_w"][k.upper]
        platform_quat = self.ref["terrain_box_quat_wxyz"][k.upper].expand(self.num_envs, -1)
        knee_local = quat_apply_inverse(platform_quat, pivot)
        footprint = (
            self.ref["terrain_box_half_sizes"][k.upper, :2] - knee_local[:, :2].abs()
        ).amin(1)
        contact = accepted_knee(m, m["foot_ground_load_N"], footprint, self.body_weight_N)
        was_braced = self.milestones.braced.clone()
        was_airborne = self.milestones.airborne.clone()
        knee, success = self.milestones.update(
            switched,
            advance,
            four_point,
            m["bilateral_push"],
            m["takeoff_observed"],
            contact,
            self.ground_start,
        )
        root = d.root_link_pos_w - self.scene.env_origins
        fallen = root[:, 2] - k.floor_z < 0.43
        expired = self.episode_length_buf >= self.max_episode_length
        m.update(
            success=success,
            knee_reached=knee,
            ground_start=self.ground_start.clone(),
            prelaunch_start=self.prelaunch_start.clone(),
            ground_reset_offset_xy=self.ground_reset_offset_xy.clone(),
            walk_switches=self.milestones.walk_switches.clone(),
            front_progress_m=front_progress,
            rewarded_walk_switch=rewarded_switch,
            advance_m=advance,
            walked_to_edge=self.milestones.walked_to_edge.clone(),
            bilateral_braced=self.milestones.braced.clone(),
            push_observed=self.milestones.push_observed.clone(),
            takeoff_observed=self.milestones.airborne.clone(),
            right_knee_top_N=m["right_knee_top_load_N"],
            right_knee_patch_touched=m["right_knee_top_load_N"] > 8,
            right_knee_load_bodyweight=m["right_knee_top_load_N"] / self.body_weight_N,
            knee_footprint_margin_m=footprint,
            knee_hold_s=self.milestones.hold * self.step_dt,
            knee_contact=contact,
            traj_initial_frame=self.starts.clone(),
            terminated=fallen,
            expired=expired,
            root_pose=torch.cat((root, d.root_link_quat_w), 1),
            joint_pos=d.joint_pos[:, self.joint_ids],
            joint_vel=d.joint_vel[:, self.joint_ids],
        )
        m["reward"] = self._physical_reward(
            m, sites, front_gain, rewarded_switch, near_frontier, was_braced, was_airborne
        )
        if self.has_plan:
            _, forces, centers = student_contact_state(self, with_points=True)
            score = self.plan_score.measure(forces, centers)
            score = self.front_approach.foot_plan_reward(
                score, self.plan_score.match[:, :2] / self.plan_score.dwell,
                self.plan_score.loaded[:, :2] == self.plan_score.dwell, self.milestones.braced,
                front_progress, near_frontier,
            )
            m["reward"] += 0.75 * score
            self.has_plan = False
        self.first_sample.zero_()
        self.last_metrics = m
        # A seeded success or a real knee without the walk is visible as a partial
        # result. Speed, posture, or a progress timeout alone can never produce success.
        return fallen, expired | knee

    def _physical_reward(self, m, sites, front_gain, rewarded_switch, near_frontier,
                         was_braced, was_airborne):
        d = self.robot.data
        up = -d.projected_gravity_b[:, 2]
        floor_load = m["foot_ground_patch_N"]
        bearing = floor_load > 15
        share = floor_load * bearing
        slip = (m["foot_ground_slip_m_s"] * share).sum(1) / share.sum(1).clamp_min(1)
        upright = ((up - 0.70) / 0.26).clamp(0, 1)
        walk = ~self.milestones.braced & bearing.any(1) & (slip < 0.25)
        gait_quality = walk * upright / (1 + (slip / 0.14).square())
        arms = (
            (m["arm_support_top_N"] / 18).clamp(0, 1).mean(1)
            * ~m["wrist_forbidden"]
            * bearing.all(1)
        )
        upper = self.knee_contacts.upper
        half = self.ref["terrain_box_half_sizes"][upper]
        rel = sites[:, 2:4] - self.ref["terrain_box_centers_w"][upper]
        orient = self.ref["terrain_box_quat_wxyz"][upper].expand(self.num_envs * 2, -1)
        arm_local = quat_apply_inverse(orient, rel.reshape(-1, 3)).reshape_as(rel)
        outer = (arm_local[:, :, :2].abs() - half[:2]).clamp_min(0).norm(dim=2)
        arm_distance = torch.hypot(outer, arm_local[:, :, 2] - half[2]).mean(1)
        vertical_gap = (
            sites[:, 7, 2] - (self.knee_contacts.top_heights - self.scene.env_origins[:, 2])
        ).abs()
        surface_gap = torch.hypot((-m["knee_footprint_margin_m"]).clamp_min(0), vertical_gap)
        knee_close = (1 - surface_gap / 0.65).clamp(0, 1)
        approach_complete = self.milestones.walked_to_edge | ~self.ground_start
        walked = ~self.ground_start | (
            (self.milestones.walk_switches >= 1) & (m["advance_m"] > 0.20)
            & (near_frontier | self.milestones.braced)
        )
        foot_balance = (2 * floor_load.amin(1) / floor_load.sum(1).clamp_min(1)).clamp(0, 1)
        dock = (
            arms
            * foot_balance
            / (1 + (d.root_link_lin_vel_w.norm(dim=1) / 0.25).square())
            * self.milestones.braced
            * approach_complete
        )
        transfer = (m["right_knee_load_bodyweight"] / 0.20).clamp(0, 1)
        transfer *= (
            self.milestones.airborne
            & (m["knee_footprint_margin_m"] >= 0)
            & ~m["wrist_forbidden"]
            & (m["foot_ground_load_N"] < 3).all(1)
        )
        # After a measured push the arms carry weight as the foot loads diminish.
        # This term is zero for resting, slumping onto the ledge, and falling onto a knee.
        recent_push = (self.ref_steps - self.knee_contacts.push_step) <= round(0.16 / self.step_dt)
        unloaded = (1 - floor_load.sum(1) / self.body_weight_N).clamp(0, 1)
        arm_carry = (m["arm_support_top_N"].amin(1) / 18).clamp(0, 1)
        supported_rise = (recent_push * approach_complete * arm_carry * unloaded
                          * (m["com_vz_m_s"] / .25).clamp(0, 1) * ~m["wrist_forbidden"])
        features = torch.stack(
            (
                (1 - arm_distance / 0.55).clamp(0, 1) * bearing.any(1) * walked,
                arms * walked,
                knee_close * self.milestones.braced * approach_complete,
                dock,
                transfer,
                supported_rise,
            ),
            1,
        )
        gain = (features - self.progress_best).clamp_min(0)
        gain[self.first_sample] = 0
        self.progress_best.copy_(torch.maximum(self.progress_best, features))
        reward = (gain * gain.new_tensor([1.0, 2.0, 4.0, 3.0, 4.0, 3.0])).sum(1)
        reward += 6 * front_gain * walk * upright
        reward += gait_quality * (front_gain / 0.28).clamp(max=self.step_dt) + rewarded_switch.float()
        reward += (self.milestones.braced & ~was_braced).float() * (0.25 + 1.75 * approach_complete)
        reward += (self.milestones.airborne & ~was_airborne).float() * 3
        reward += self.step_dt * 5 * m["knee_contact"] * self.milestones.airborne
        reward += 5 * m["knee_reached"] + 10 * m["success"] - 2 * m["terminated"]
        return reward

    def _get_rewards(self):
        return self.last_metrics["reward"]

    def load_demonstration_starts(self, paths, *, initial_knee=False, cloning_split_paths=None):
        """Seed practice from recorded, still-grounded states. All event counters restart at zero."""
        paths = [Path(p).resolve() for p in paths]
        has_complete = False
        if not initial_knee:
            for path in paths:
                with np.load(path, allow_pickle=False) as pack:
                    has_complete |= str(pack["source_kind"].item()) == "isaac_ground_bridge_knee"
        allowed = training_start_episodes(paths, cloning_split_paths) if has_complete else {}
        banks = []
        prelaunch = []
        counts = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as pack:
                if pack["joint_names"].tolist() != self.reference_np["joint_names"].tolist():
                    raise ValueError(f"Demonstration joint order differs: {path}")
                if not np.allclose(
                    pack["terrain_box_centers_w"], self.reference_np["terrain_box_centers_w"]
                ):
                    raise ValueError(f"Demonstration uses another terrain: {path}")
                # Early examples from the knee teacher and late examples from approaching
                # the ledge meet at double foot support. An airborne start cannot earn credit.
                feet = pack["student_contact_force"][:, :2]
                mask = (
                    (feet > 12).all(1)
                    & pack["contact_observed"]
                    & (pack["student_contact_force"][:, 7] < 3)
                )
                if path in allowed:
                    mask &= np.isin(pack["episode_id"], list(allowed[path]))
                if initial_knee:
                    if pack["source_kind"].item() != "isaac_knee2300":
                        raise ValueError(
                            "The local diagnostic needs the original knee teacher sample"
                        )
                    mask = pack["step"] == 0
                if not mask.any():
                    continue

                def bank_from(rows):
                    def selected(key):
                        return torch.as_tensor(pack[key][rows], dtype=torch.float32, device=self.device)

                    bank = {key: selected(key) for key in ("root_pose", "joint_pos", "joint_vel", "prev_target")}
                    for velocity in ("lin", "ang"):
                        world_key = f"root_{velocity}_vel_w"
                        bank[world_key] = (selected(world_key) if world_key in pack
                            else quat_apply(bank["root_pose"][:, 3:], selected(f"root_{velocity}_vel_b")))
                    bank["frame"] = (selected("reference_frame").long() if "reference_frame" in pack
                        else torch.full((len(bank["root_pose"]),), -1, device=self.device, dtype=torch.long))
                    return bank

                banks.append(bank_from(mask))
                if path in allowed and str(pack["source_kind"].item()) == "isaac_ground_bridge_knee":
                    for episode, indices in grounded_prelaunch_rows(pack, allowed[path]).items():
                        prelaunch.append(bank_from(indices))
                        counts[f"{path}#{episode}"] = len(indices)
        if initial_knee:
            self.diagnostic_start = banks[0]
        else:
            self.demonstration_starts = banks
            self.prelaunch_starts = prelaunch
            self.prelaunch_start_counts = counts
            if has_complete:
                print(f"Grounded prelaunch reset bank: {sum(counts.values())} measured states from {len(counts)} training episodes", flush=True)

    def _reset_idx(self, env_ids):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        DirectRLEnv._reset_idx(self, ids)
        if hasattr(self, "point_history"):
            self.point_history.reset(ids)
        n = len(ids)
        self.prelaunch_start[ids] = False
        frames = torch.zeros(n, dtype=torch.long, device=self.device)
        if self.stage_training:
            bucket = (ids + self.reset_count[ids]) % 8
            self.reset_count[ids] += 1
            # Every partial start precedes takeoff and still has to perform a genuine push.
            # Half of the training resets and every formal evaluation reset use frame zero.
            for region, lo, hi in (
                (bucket == 4, self.walk_start_frame, self.arm_approach_frame),
                (bucket == 5, self.arm_approach_frame, self.segment_start),
                (bucket >= 6, self.segment_start, self.knee_contacts.push_start + 1),
            ):
                candidate = torch.randint(lo, max(lo + 1, hi), (n,), device=self.device)
                frames = torch.where(region, candidate, frames)
        self.starts[ids] = frames
        self.ground_start[ids] = frames == 0
        if self.diagnostic_start is not None:
            self.ground_start[ids] = False
        self.ground_reset_offset_xy[ids] = ground_reset_offsets(
            self.ground_start[ids], self.advance_direction
        )
        self.knee_contacts.reset(ids)
        self.milestones.reset(ids)
        self.footsteps.reset(ids)
        self.plan_score.reset(ids)
        self.ref_steps[ids] = 0
        self.progress_best[ids] = 0
        self.first_sample[ids] = True
        ref, d = self.ref, self.robot.data
        pose = torch.cat(
            (ref["root_pos"][frames] + self.scene.env_origins[ids], ref["root_quat"][frames]), 1
        )
        angular = ref["root_ang_vel"][frames]
        linear = ref["root_lin_vel"][frames]
        pose[:, :2] += self.ground_reset_offset_xy[ids]
        q = d.default_joint_pos[ids].clone()
        qd = torch.zeros_like(q)
        js = ref["joint_pos"][frames]
        vs = ref["joint_vel"][frames]
        targets = js.clone()
        banks = (
            [self.diagnostic_start]
            if self.diagnostic_start is not None
            else self.demonstration_starts
            if self.stage_training
            else []
        )
        if banks:
            bank_ids = torch.randint(len(banks), (n,), device=self.device)
            if self.stage_training and self.diagnostic_start is None and self.prelaunch_starts:
                chosen = bucket >= 6
                self.prelaunch_start[ids] = chosen
                bank_ids[chosen] = len(banks) + torch.randint(
                    len(self.prelaunch_starts), (int(chosen.sum()),), device=self.device)
                banks = [*banks, *self.prelaunch_starts]
            for idx, bank in enumerate(banks):
                mask = ~self.ground_start[ids] & (bank_ids == idx)
                src = torch.randint(len(bank["joint_pos"]), (int(mask.sum()),), device=self.device)
                self.starts[ids[mask]] = bank["frame"][src]
                pose[mask] = bank["root_pose"][src]
                pose[mask, :3] += self.scene.env_origins[ids[mask]]
                linear[mask], angular[mask] = (
                    bank["root_lin_vel_w"][src],
                    bank["root_ang_vel_w"][src],
                )
                js[mask], vs[mask], targets[mask] = (
                    bank["joint_pos"][src],
                    bank["joint_vel"][src],
                    bank["prev_target"][src],
                )
        offset = quat_apply(pose[:, 3:], d.body_com_pos_b[ids, 0])
        vcom = linear + torch.cross(angular, offset, dim=1)
        q[:, self.joint_ids], qd[:, self.joint_ids] = js, vs
        self.robot.write_root_pose_to_sim(pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.cat((vcom, angular), 1), env_ids=ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.targets[ids] = targets
        self.robot.set_joint_position_target(
            self.targets[ids], joint_ids=self.joint_ids, env_ids=ids
        )
        self.actions[ids] = self.targets[ids] - self.nominal[ids]
        self.previous_actions[ids] = self.actions[ids]
