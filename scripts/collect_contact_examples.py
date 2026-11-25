"""Record physical pre-step inputs and post-limiter motor labels from frozen teachers."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "third_party/rsl_rl")]
from elf3_climbing.paths import OUTPUT
from elf3_climbing.reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["knee2300", "walk", "brace"])
    p.add_argument(
        "--ground-checkpoint", type=Path, help="Frozen approach policy for the brace segment"
    )
    p.add_argument(
        "--student", type=Path, help="Collect DAgger corrective labels on mixed/student roll-ins"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--vary-knee-starts",
        action="store_true",
        help="Sample graded horizontal and joint perturbations around the original knee start",
    )
    p.add_argument(
        "--entry",
        type=Path,
        default=OUTPUT / "knee_070_contact_transfer_20261122_204705/checkpoint_02300.pt",
    )
    p.add_argument(
        "--trajectory",
        type=Path,
        default=OUTPUT / "mount_070_ground_to_steady_rise_20261123/trajectory.npz",
    )
    args = p.parse_args()
    if args.num_envs < 1 or args.output.suffix != ".npz" or args.output.exists():
        p.error("Choose a new .npz output and a positive environment count")
    if (args.source == "brace") != (args.ground_checkpoint is not None):
        p.error("Provide --ground-checkpoint for the brace teacher only")
    if args.student and args.source == "walk":
        p.error("Choose brace (full ground start) or knee2300 for DAgger")
    if args.vary_knee_starts and args.source != "knee2300":
        p.error("--vary-knee-starts applies to the original knee teacher only")
    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    env = None
    try:
        from elf3_climbing.environment import TrackerEnv
        from elf3_climbing.ground_mount import GroundMountEnv
        from elf3_climbing.mount_teacher import (
            frozen_entry,
            ground_teacher_actor,
            knee_entry_config,
            source_inputs,
        )

        request, weights, files, hashes = source_inputs(args.entry)
        cfg = knee_entry_config(
            request, files, seed=args.seed, num_envs=args.num_envs, device="cuda:0"
        )
        ground_start = args.source != "knee2300"
        base = GroundMountEnv if ground_start else TrackerEnv

        class Teacher(base):
            def _reset_idx(self, env_ids):
                super()._reset_idx(env_ids)
                if args.vary_knee_starts:
                    vary_knee_starts(self, self.robot._ALL_INDICES if env_ids is None else env_ids)

            def _pre_physics_step(self, actions):
                if args.student:
                    previous = self.targets.clone()
                super()._pre_physics_step(actions)
                # Capture the command before an automatic terminal reset can overwrite it.
                self.applied_drive_target = self.targets.clone()
                if args.student:
                    from elf3_climbing.contact_dagger import apply_intervention

                    apply_intervention(self, previous, ground_start)

            def _get_dones(self):
                done = super()._get_dones()
                if args.vary_knee_starts:
                    # Measure the loaded knee BEFORE DirectRLEnv auto-resets a finished lane.
                    from isaaclab.utils.math import quat_apply_inverse
                    from elf3_climbing.contact_tracker_task import accepted_knee

                    k, m = self.knee_contacts, self.last_metrics
                    pivot = self.site_positions()[:, 7] - self.scene.env_origins
                    pivot -= self.ref["terrain_box_centers_w"][k.upper]
                    rot = self.ref["terrain_box_quat_wxyz"][k.upper].expand(self.num_envs, -1)
                    deck_xy = quat_apply_inverse(rot, pivot)[:, :2]
                    margin = (self.ref["terrain_box_half_sizes"][k.upper, :2] - deck_xy.abs()).amin(
                        1
                    )
                    m["student_knee_bearing"] = (
                        accepted_knee(
                            m,
                            m["foot_ground_load_N"],
                            margin,
                            k.total_mass * abs(self.cfg.sim.gravity[2]),
                        )
                        & m["takeoff_observed"]
                    )
                return done

        env = Teacher(cfg, args.trajectory) if ground_start else Teacher(cfg)
        if args.vary_knee_starts:
            env.knee_contacts.enable_pose_trace()
        env.evaluation_start_phase = 0.0
        actor = frozen_entry(weights, args.entry.parent / "ppo_config.json", env.device)
        if ground_start:
            from elf3_climbing.ground_locomotion import GAIT_MODEL

            hashes.update(
                gait_weights_sha256=digest(GAIT_MODEL), trajectory_sha256=digest(args.trajectory)
            )
            env.entry_policy = actor
        if args.source == "brace":
            checkpoint = torch.load(args.ground_checkpoint, map_location="cpu", weights_only=True)
            contract = checkpoint["infos"]["contract"]
            if any(contract[key] != value for key, value in hashes.items() if key in contract):
                raise ValueError(
                    "Ground teacher weights use different robot, reference or source assets"
                )
            actor = ground_teacher_actor(checkpoint, cfg, env.device)
            hashes["ground_checkpoint_sha256"] = digest(args.ground_checkpoint)
        with torch.inference_mode():
            collect(env, actor, args, hashes)
    finally:
        if env is not None:
            env.close()
        app.close()


def vary_knee_starts(env, ids):
    """Vary the real start only. Keep the original teacher's reference and success test."""
    if not len(ids):
        return
    frames = env.starts[ids]
    group = ids % 3
    # Fixed reference height/orientation and physical velocity avoid initial drop impacts.
    # Each band broadens both translation in the ground plane and angular variation.
    shift_max = env.targets.new_tensor([0.008, 0.017, 0.030])[group]
    angle_max = env.targets.new_tensor([0.010, 0.022, 0.040])[group]
    xy = (2 * torch.rand((len(ids), 2), device=env.device) - 1) * shift_max[:, None]
    pose = torch.cat(
        [env.ref["root_pos"][frames] + env.scene.env_origins[ids], env.ref["root_quat"][frames]], 1
    )
    pose[:, :2] += xy
    dq = (2 * torch.rand_like(env.targets[ids]) - 1) * angle_max[:, None]
    # Preserve the pre-rotated left wrist used by this teacher.
    dq[:, env.wrist_joint] = 0
    q = env.robot.data.joint_pos[ids].clone()
    qd = env.robot.data.joint_vel[ids].clone()
    angles = (env.ref["joint_pos"][frames] + dq).clamp(env.limits[ids, :, 0], env.limits[ids, :, 1])
    q[:, env.joint_ids] = angles
    env.robot.write_root_pose_to_sim(pose, env_ids=ids)
    env.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
    env.targets[ids] = angles
    env.robot.set_joint_position_target(angles, joint_ids=env.joint_ids, env_ids=ids)


def collect(env, actor, args, hashes):
    from elf3_climbing.contact_examples import CONTACT_NAMES, student_contact_state
    from elf3_climbing.contact_tracker_env import student_physical_state

    walking = args.source == "walk"
    bracing = args.source == "brace"
    student = None
    if args.student:
        from elf3_climbing.contact_dagger import rollin_schedule, wait_for_walking_position
        from elf3_climbing.contact_learning import ContactActor, encode_contact_observation

        state = torch.load(args.student, map_location=env.device, weights_only=True)
        if "backbone.0.weight" in state:
            from elf3_climbing.contact_gait_actor import GaitContactActor

            student = GaitContactActor()
        else:
            student = ContactActor()
        student = student.to(env.device).eval()
        student.load_state_dict(state)
        hashes["student_sha256"] = digest(args.student)
    obs, _ = env.reset()
    n = env.num_envs
    alive = torch.ones(n, device=env.device, dtype=torch.bool)
    accepted = torch.zeros_like(alive)
    lengths = torch.zeros(n, device=env.device, dtype=torch.int64)
    bearing_run = torch.zeros_like(lengths)
    bearing_peak = torch.zeros_like(lengths)
    switches = torch.zeros_like(lengths)
    minimum_up = torch.ones_like(lengths, dtype=torch.float32)
    max_drift = torch.zeros_like(minimum_up)
    origin = env.robot.data.root_link_pos_w[:, :2].clone()
    walk_origin = origin.clone()
    distance = torch.zeros_like(minimum_up)
    buffers = {}
    corrective_rows = []
    # Walking labels end on the lower surface, before the reaching/brace controller.
    horizon = round(3.7 / env.step_dt) if walking else env.max_episode_length + 1
    sensor_names = list(env.support_sensors)
    for step in range(horizon):
        waiting = torch.zeros_like(alive)
        if student is not None and bracing:
            waiting = wait_for_walking_position(env)
            # Re-read the teacher's task observation after retiming; physical state
            # and its five-frame history have not advanced.
            if waiting.any():
                obs = env._get_observations()
        pos, force, points = student_contact_state(env, with_points=True)
        if step == 0:
            force.zero_()
            points.fill_(torch.nan)
        d = env.robot.data
        raw = student_physical_state(env, pos, force)
        if student is not None:
            env.student_fraction = rollin_schedule(env, step, bracing)
            env.student_suggestion = student(encode_contact_observation(raw, env.nominal))
            up = -raw["projected_gravity_b"][:, 2]
            floor_h = raw["root_pos"][:, 2] - env.knee_contacts.floor_z
            if bracing:
                before_reaching = env.course_time_s < 3.70
                too_low = torch.where(
                    before_reaching, (up < 0.78) | (floor_h < 0.68), (up < 0.20) | (floor_h < 0.46)
                )
            else:
                # The original knee controller deliberately leans deeply on the
                # arms during its jump. Its original failure conditions still apply.
                too_low = floor_h < 0.46
            alive &= ~too_low
            keep = alive & (env.student_fraction > 0)
            corrective_rows.append(keep.cpu().numpy().copy())
            raw.update(student_fraction=env.student_fraction, waited_for_position=waiting)
        # Keep the original forearm/elbow markers independently of the wrist channels.
        raw["student_site_pos"] = raw.pop("site_pos")
        raw["student_contact_force"] = raw.pop("contact_force")
        raw.update(
            student_contact_point=points,
            site_pos=env.site_positions() - env.scene.env_origins[:, None],
            root_pose=torch.cat((raw["root_pos"], raw["root_quat"]), dim=1),
            root_lin_vel_w=d.root_link_lin_vel_w,
            root_ang_vel_w=d.root_link_ang_vel_w,
            body_pos=d.body_link_pos_w - env.scene.env_origins[:, None],
            body_quat=d.body_link_quat_w,
            support_force_matrix_w=torch.stack(
                [env.support_sensors[name].data.force_matrix_w[:, 0] for name in sensor_names],
                dim=1,
            ),
            reference_frame=env.frames(),
            contact_observed=torch.full_like(alive, step != 0),
        )
        for key, value in raw.items():
            buffers.setdefault(key, []).append(value.cpu().numpy().copy())
        action = torch.zeros_like(env.actions) if walking else actor(obs["policy"])
        obs, _, failed, truncated, _ = env.step(action)
        done = failed | truncated
        buffers.setdefault("drive_target", []).append(env.applied_drive_target.cpu().numpy().copy())
        if student is not None:
            buffers.setdefault("executed_target", []).append(
                env.executed_drive_target.cpu().numpy().copy()
            )
        m = env.last_metrics
        lengths += alive.long()
        if args.vary_knee_starts:
            bearing_run = torch.where(alive & m["student_knee_bearing"], bearing_run + 1, 0)
            bearing_peak = torch.maximum(bearing_peak, bearing_run)
            for key, value in (
                ("post_knee_hold_steps", bearing_run),
                ("post_right_knee_load_N", m["right_knee_top_load_N"]),
                ("post_right_knee_off_patch_N", m["right_knee_off_patch_load_N"]),
                ("post_floor_load_N", m["foot_ground_load_N"]),
            ):
                buffers.setdefault(key, []).append(value.cpu().numpy().copy())
        if walking:
            xy = m["root_pose"][:, :2] + env.scene.env_origins[:, :2]
            if step < round(1.5 / env.step_dt):
                max_drift = torch.maximum(max_drift, (xy - origin).norm(dim=1))
                walk_origin.copy_(xy)
            else:
                switches += m["walk_step_switch"].long() * alive.long()
                minimum_up = torch.minimum(minimum_up, m["torso_up"])
            distance = (xy - walk_origin) @ env.advance_direction
        else:
            reached = m["approach_accepted"] if bracing else m["success"]
            if args.vary_knee_starts:
                reached = reached & (bearing_peak >= round(0.20 / env.step_dt))
            accepted |= alive & reached & ~failed
            if bracing:
                # Last stored command is the last ground-stage action. No frozen
                # knee actions or post-gate falls belong to this bridge dataset.
                done |= reached
        alive &= ~done
        if not alive.any():
            break
    if walking:
        # Accept only an independently measured walking segment, never represent
        # these lower-level trials as successful forearm or knee handoffs.
        accepted = (
            alive & (switches >= 1) & (distance >= 0.18) & (max_drift < 0.16) & (minimum_up > 0.85)
        )
    keep_rows = np.stack(corrective_rows) if student is not None else None
    if keep_rows is not None:
        # A short but continuous correction is useful before a failure. Never
        # describe a correction sample as having met the teacher's terminal goal.
        ids = np.flatnonzero(keep_rows.sum(axis=0) >= max(8, round(0.2 / env.step_dt)))
    else:
        ids = accepted.nonzero().flatten().cpu().numpy()
    if not len(ids):
        raise RuntimeError(f"No verified {args.source} examples out of {n} environment resets")
    count = lengths.cpu().numpy()
    select = [
        np.flatnonzero(keep_rows[:, i]) if keep_rows is not None else np.arange(count[i])
        for i in ids
    ]
    arrays = {}
    for name, values in buffers.items():
        by_step = np.stack(values)
        arrays[name] = np.concatenate(
            [by_step[index, i] for i, index in zip(ids, select, strict=True)]
        )
    episodes = np.repeat(ids, [len(index) for index in select])
    arrays.update(
        episode_id=episodes,
        step=np.concatenate(select).astype(np.int32),
        schema=np.array("elf3_contact_examples_v1"),
        source_kind=np.array(
            {
                "walk": "isaac_official_ground_walk",
                "brace": "isaac_ground_brace",
                "knee2300": "isaac_knee2300",
            }[args.source]
        ),
        student_site_names=np.array(CONTACT_NAMES),
        site_names=env.reference_np["site_names"],
        joint_names=env.reference_np["joint_names"],
        nominal=env.nominal[0].cpu().numpy(),
        body_names=np.array(env.robot.body_names),
        support_sensor_names=np.array(sensor_names),
        step_dt=np.array(env.step_dt),
    )
    for key in ("terrain_box_centers_w", "terrain_box_half_sizes", "terrain_box_quat_wxyz"):
        arrays[key] = env.reference_np[key]
    provenance = dict(
        source_kind=str(arrays["source_kind"]),
        attempts=n,
        accepted=len(ids),
        steps=len(episodes),
        seed=args.seed,
        entry_checkpoint=str(args.entry.resolve()),
        wrist_mode=env.cfg.knee_wrist_contact,
        hashes=hashes,
        qualification=(
            "first 3.7 s survived, alternating loaded landing, forward >=0.18 m, flip drift <0.16 m, walking torso up >0.85"
            if walking
            else "accepted original 2300 deterministic knee policy's first episode, required continuous right knee hold"
        ),
    )
    if bracing:
        provenance.update(
            qualification="Unassisted ground start to verified 3-step bilateral arm and sole support handoff; no knee-phase actions",
            ground_checkpoint=str(args.ground_checkpoint.resolve()),
        )
    if walking:
        provenance["walking_switches"] = switches.cpu().numpy()[ids].tolist()
        provenance["walking_advance_m"] = distance.cpu().numpy()[ids].tolist()
    if args.vary_knee_starts:
        first = np.r_[True, episodes[1:] != episodes[:-1]]
        frame = env.segment_start
        xy_ref = env.ref["root_pos"][frame, :2].cpu().numpy()
        q_ref = env.ref["joint_pos"][frame].cpu().numpy()
        provenance.update(
            start_variation="Independent uniform joint and XY offsets about original phase-zero preload",
            start_bounds_xy_m=[0.008, 0.017, 0.030],
            start_bounds_joint_rad=[0.010, 0.022, 0.040],
            attempts_by_band=np.bincount(np.arange(n) % 3, minlength=3).tolist(),
            accepted_by_band=np.bincount(ids % 3, minlength=3).tolist(),
            # For DAgger the first saved row may be after reset, so only describe
            # starting displacements when the exact first physical row is kept.
        )
        if student is None:
            provenance["qualification"] += (
                "; observed takeoff and >=0.20 s of uninterrupted true knee bearing >=0.15 bodyweight, both soles clear and unloaded, knee inside deck and off-patch load <=8 N"
            )
            provenance["start_actual_xy_offset_m"] = (
                arrays["root_pose"][first, :2] - xy_ref
            ).tolist()
            provenance["start_actual_joint_rms_rad"] = np.sqrt(
                np.mean((arrays["joint_pos"][first] - q_ref) ** 2, axis=1)
            ).tolist()
    if student is not None:
        arrays["source_kind"] = np.array("dagger_" + str(arrays["source_kind"]))
        provenance.update(
            source_kind=str(arrays["source_kind"]),
            student=str(args.student.resolve()),
            qualification="Continuous physically viable student/mixed roll-in before the first failure or handoff; corrective action supervision only",
            corrective_episodes=len(ids),
            reached_teacher_goal=int(accepted.sum().item()),
            action_label="Teacher's current slew-limited absolute motor target from the actual previous target",
            executed_target="Motor target sent to physics; convex mixture with per-episode student fraction",
            student_fractions=np.unique(arrays["student_fraction"]).tolist(),
        )
    arrays["provenance"] = np.array(json.dumps(provenance, allow_nan=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publish: never replace an existing collection, even on a race.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        path = Path(temp) / "examples.npz"
        np.savez_compressed(path, **arrays)
        os.link(path, args.output)
    print("CONTACT_EXAMPLES", json.dumps(provenance), args.output, flush=True)


if __name__ == "__main__":
    main()
