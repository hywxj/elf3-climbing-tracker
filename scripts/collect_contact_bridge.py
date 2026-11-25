"""Collect training-only motor examples during a physically integrated ground-to-knee bridge."""

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/rsl_rl")]
from elf3_climbing.paths import OUTPUT
from elf3_climbing.reference import digest


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-checkpoint", type=Path, required=True)
    p.add_argument("--knee-examples", type=Path, required=True)
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
    p.add_argument("--num-envs", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dagger-student",
        type=Path,
        help="Use brief roll-ins of a frozen terrain-point actor and save the live teacher corrections",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for metrics and qualifying physical episodes",
    )
    p.add_argument(
        "--settings",
        type=Path,
        help="JSON rows [blend duration, joint feedback, ankle feedback, bilateral load feedback, optional early-swing right leg shortening (rad), optional prelaunch support precision (0–1), optional left swing shortening (rad)]",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="Save observed bridge transitions that meet the reported criteria",
    )
    args = p.parse_args()
    if args.dagger_student and not args.record:
        p.error("--dagger-student requires --record")
    return args


def rollout(env, actor, settings, args, hashes):
    from elf3_climbing.contact_examples import CONTACT_NAMES, student_contact_state
    from elf3_climbing.contact_tracker_env import student_physical_state
    from elf3_climbing.contact_tracker_task import accepted_knee
    from elf3_climbing.walk import Footsteps

    obs, _ = env.reset(seed=args.seed)
    n = env.num_envs
    device = env.device
    ids = torch.arange(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    started = torch.zeros_like(alive)
    qualified = started.clone()
    knee = started.clone()
    loaded_knee = started.clone()
    walked = started.clone()
    hold = torch.zeros_like(ids)
    partial_streak = hold.clone()
    peak_knee_hold = hold.clone()
    best_top_N = torch.zeros(n, device=device)
    push_seen = started.clone()
    takeoff_seen = started.clone()
    first = hold.new_full((n,), -1)
    best_step = first.clone()
    end_step = first.clone()
    partial_step = first.clone()
    initial = env.actions.new_full((n,), float("inf"))
    best = initial.clone()
    partial_cost = best.clone()
    steps = Footsteps(n, device)
    switches = hold.clone()
    brace_distance = initial.clone()
    brace_steps = hold[:, None].repeat(1, 2)
    origin = env.site_positions()[:, :2, :2].mean(1) - env.scene.env_origins[:, :2]
    buffers = [[] for _ in range(n)]
    report = {}
    intervention = None
    if args.dagger_student:
        from elf3_climbing.contact_dagger import BridgeIntervention
        from elf3_climbing.contact_point_actor import PointContactActor

        policy = PointContactActor().to(device).eval()
        policy.load_state_dict(
            torch.load(args.dagger_student, weights_only=True, map_location=device)
        )
        intervention = BridgeIntervention(env, policy)
        hashes["dagger_student"] = digest(args.dagger_student)
    attempt_steps = int(env.max_episode_length) + 2
    for i in range(attempt_steps):
        before = env.ground_stage.clone()
        taking = alive.clone()
        if args.record and taking.any():
            ix = taking.nonzero().flatten()
            sites, force, points = student_contact_state(env, with_points=True)
            if i == 0:
                force.zero_()
                points.fill_(torch.nan)
            raw = student_physical_state(env, sites, force)
            if intervention is not None:
                up = -raw["projected_gravity_b"][:, 2]
                floor_h = raw["root_pos"][:, 2] - env.knee_contacts.floor_z
                approaching = (before == 0) & (env.course_time_s < 3.72) & ~env.bridge_started
                unstable = torch.where(approaching, (floor_h < 0.68) | (up < 0.78), floor_h < 0.46)
                alive &= ~unstable
                end_step = torch.where(taking & ~alive, i - 1, end_step)
                taking = alive.clone()
                if not taking.any():
                    break
                ix = taking.nonzero().flatten()
                intervention.prepare(env, raw, i, alive)
                raw.update(
                    student_fraction=env.student_fraction,
                    intervention_group=intervention.group,
                    course_stage=before,
                    bridge_active=env.bridge_active,
                )
            raw["student_site_pos"] = raw.pop("site_pos")
            raw["student_contact_force"] = raw.pop("contact_force")
            raw["student_contact_point"] = points
            raw["root_pose"] = torch.cat((raw["root_pos"], raw["root_quat"]), 1)
            payload = {k: v[ix].cpu().numpy() for k, v in raw.items()}
        obs, _, fail, done, _ = env.step(actor(obs["policy"]))
        m = env.last_metrics
        sites = env.site_positions() - env.scene.env_origins[:, None]
        feet = sites[:, :2]
        switches += steps.update(
            alive & ~started,
            m["foot_ground_patch_N"],
            feet[:, :, 2] - env.knee_contacts.floor_z,
            feet[:, :, :2] @ env.advance_direction,
        ).long()
        advance = (feet[:, :, :2].mean(1) - origin) @ env.advance_direction
        enter = alive & m["bridge_start"]
        first = torch.where(enter, i, first)
        brace_distance = torch.where(enter, advance, brace_distance)
        brace_steps = torch.where(enter[:, None], steps.counts, brace_steps)
        walked |= enter & (switches >= 1) & (advance >= 0.25)
        started |= enter
        initial = torch.where(enter, m["bridge_initial_cost"], initial)
        candidate = m["bridge_active"] & m["bridge_support"] & alive
        better = candidate & (m["bridge_cost"] < best)
        best = torch.where(better, m["bridge_cost"], best)
        best_step = torch.where(better, i, best_step)
        partial_streak = torch.where(m["bridge_partial"] & alive, partial_streak + 1, 0)
        keep = (partial_streak >= 4) & (m["bridge_cost"] < partial_cost)
        partial_step = torch.where(keep, i, partial_step)
        partial_cost = torch.where(keep, m["bridge_cost"], partial_cost)
        qualified |= alive & m["bridge_qualified"]
        knee |= alive & m["entry_accepted"]
        margin = env.mount_task.footprint_margin(sites[:, 7])
        bearing = (
            (before == 1)
            & ~fail
            & ~done
            & accepted_knee(m, m["foot_ground_load_N"], margin, env.mount_task.body_weight)
        )
        hold = torch.where(bearing & alive, hold + 1, 0)
        peak_knee_hold = torch.maximum(peak_knee_hold, hold)
        best_top_N = torch.maximum(best_top_N, torch.where(alive, m["right_knee_top_load_N"], 0.0))
        push_seen |= alive & (before == 1) & m["bilateral_push"]
        takeoff_seen |= alive & (before == 1) & m["takeoff_observed"]
        loaded_knee |= hold >= 10
        if args.record and taking.any():
            after = {
                "drive_target": env.applied_drive_target,
                "post_knee_hold_steps": hold,
                "post_brace": enter,
                "post_prelaunch_matched": m["bridge_qualified"],
                "post_bilateral_push": m["bilateral_push"],
                "post_takeoff": m["takeoff_observed"],
                "post_right_knee_top_N": m["right_knee_top_load_N"],
                "post_knee_inside_m": margin,
                "post_walk_switches": switches,
                "post_foot_advance_m": advance,
            }
            if intervention is not None:
                after["executed_target"] = env.executed_drive_target
                after["student_target"] = env.nominal + env.student_suggestion
            payload.update({k: v[ix].cpu().numpy() for k, v in after.items()})
            for row, env_id in enumerate(ix.tolist()):
                buffers[env_id].append((i, row, payload))
        stop = alive & (fail | done | m["entry_accepted"] | loaded_knee)
        end_step = torch.where(stop, i, end_step)
        alive &= ~stop
        if i % 150 == 149:
            print(
                "BRIDGE_STATUS",
                i,
                "reached",
                int(started.sum()),
                "substantially improved",
                int((partial_step >= 0).sum()),
                "matched",
                int(qualified.sum()),
                "knee",
                int(knee.sum()),
                flush=True,
            )
        if not alive.any():
            break

    def finite(v):
        return round(float(v), 5) if torch.isfinite(v) else None

    outcomes = []
    for s, params in enumerate(settings):
        mask = ids % len(settings) == s
        s0 = mask & started
        good = s0 & best.isfinite()
        rat = best[good] / initial[good]
        outcomes.append(
            dict(
                params=params,
                n=int(mask.sum()),
                braced=int(s0.sum()),
                walked_and_braced=int((mask & walked).sum()),
                substantial=int((s0 & (partial_step >= 0)).sum()),
                qualified=int((s0 & qualified).sum()),
                knee=int((s0 & knee).sum()),
                knee_loaded=int((s0 & loaded_knee).sum()),
                full_ground_knee=int((s0 & loaded_knee & walked).sum()),
                median_best_cost=finite(best[good].median()) if good.any() else None,
                median_cost_ratio=finite(rat.median()) if good.any() else None,
            )
        )
    report.update(
        attempts=n,
        seed=args.seed,
        braced=int(started.sum()),
        substantially_improved=int((partial_step >= 0).sum()),
        matched=int(qualified.sum()),
        knee=int(knee.sum()),
        knee_loaded=int(loaded_knee.sum()),
        walk_then_loaded_knee=int((walked & loaded_knee).sum()),
        qualified_with_walk=int((qualified & walked).sum()),
        push_seen=int(push_seen.sum()),
        takeoff_seen=int(takeoff_seen.sum()),
        physical_knee_touch=int((best_top_N > 8).sum()),
        positive_loaded_hold=int((peak_knee_hold > 0).sum()),
        peak_hold_steps=int(peak_knee_hold.max()),
        params=outcomes,
        loaded_episodes=[
            dict(
                env=int(j),
                switches=int(switches[j]),
                left_right_steps=brace_steps[j].tolist(),
                at_brace_advance_m=finite(brace_distance[j]),
                walked=bool(walked[j]),
            )
            for j in loaded_knee.nonzero().flatten().tolist()
        ],
        hashes=hashes,
    )

    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    def save(
        name,
        selection,
        last,
        *,
        ground_history,
        qualification,
        first_step=None,
        clone_executed=False,
    ):
        chosen = selection.nonzero().flatten().tolist()
        series = []
        for j in chosen:
            earliest = (
                int(first_step[j])
                if first_step is not None
                else (0 if ground_history else int(first[j]) + 1)
            )
            series.append(
                [sample for sample in buffers[j] if earliest <= sample[0] <= int(last[j])]
            )
        valid = [(j, rows) for j, rows in zip(chosen, series, strict=True) if len(rows) >= 4]
        if not valid:
            return 0
        first_payload = valid[0][1][0][2]
        arrays = {
            k: np.stack([p[k][row] for _, rows in valid for _, row, p in rows])
            for k in first_payload
        }
        if clone_executed:
            arrays["teacher_target"] = arrays["drive_target"]
            arrays["drive_target"] = arrays["executed_target"]
        arrays.update(
            episode_id=np.concatenate([np.full(len(rows), j, dtype=np.int32) for j, rows in valid]),
            step=np.array([t for _, rows in valid for t, _, _ in rows], dtype=np.int32),
            contact_observed=np.array([t > 0 for _, rows in valid for t, _, _ in rows]),
            body_weight_N=np.concatenate(
                [
                    np.full(len(rows), float(env.mount_task.body_weight[j]), dtype=np.float32)
                    for j, rows in valid
                ]
            ),
            joint_names=env.reference_np["joint_names"],
            nominal=env.nominal[0].cpu().numpy(),
            student_site_names=np.array(CONTACT_NAMES),
            step_dt=np.array(env.step_dt),
            schema=np.array("elf3_contact_examples_v1"),
            source_kind=np.array(name),
        )
        metadata = dict(
            source_kind=name,
            seed=args.seed,
            attempts=n,
            accepted=len(valid),
            steps=len(arrays["step"]),
            hashes=hashes,
            ground_continuous=True,
            recorded_segment="from physical ground reset"
            if ground_history
            else "after physical four-support approach",
            qualification=qualification,
            settings=[settings[j % len(settings)] for j, _ in valid],
        )
        if intervention is not None:
            metadata.update(
                student=str(args.dagger_student.resolve()),
                rollin="One physical 0.30s student pulse with 3-step ramps; teacher then corrects the visited state",
                executed_target="Actual slew-limited absolute joint target supplied to physics",
                student_target="Unconstrained absolute target proposed by the point-policy student",
                drive_target="Actual executed target"
                if clone_executed
                else "Live teacher target after slew limit from the actual previous motor target",
                start_groups=[
                    "walking",
                    "approach",
                    "established_four_support",
                    "knee_entry",
                    "late_knee_entry",
                ],
                actual_full_success_episodes=[
                    j
                    for j, _ in valid
                    if bool(walked[j] & push_seen[j] & takeoff_seen[j] & loaded_knee[j])
                ],
            )
            if first_step is not None:
                metadata["recorded_segment"] = (
                    "4 real history frames before takeover, then student pulse and physically valid teacher recovery"
                )
        arrays["provenance"] = np.array(json.dumps(metadata, allow_nan=False))
        for k in ["terrain_box_centers_w", "terrain_box_half_sizes", "terrain_box_quat_wxyz"]:
            arrays[k] = env.reference_np[k]
        with tempfile.TemporaryDirectory(dir=args.output) as tmp:
            f = Path(tmp) / "episodes.npz"
            np.savez_compressed(f, **arrays)
            os.link(f, args.output / (name + ".npz"))
        return len(valid)

    if intervention is not None:
        # Keep the four history frames consumed by this policy. Clip at the first physical
        # failure or qualifying knee event, including when it occurs inside the pulse.
        started_at = intervention.start
        report["intervened"] = int((started_at >= 0).sum())
        last = torch.minimum(
            torch.where(end_step >= 0, end_step, i),
            started_at + intervention.width + intervention.recovery,
        )
        touched = (started_at >= 0) & (last >= started_at + 3)
        report["corrective_episodes_saved"] = save(
            "dagger_full_bridge",
            touched,
            last,
            ground_history=True,
            first_step=(started_at - 4).clamp_min(0),
            qualification="Measured student/teacher-mixed roll-in and teacher correction; no requirement of terminal success",
        )
        report["full_episodes_saved"] = save(
            "isaac_mixed_bridge_success",
            touched & walked & push_seen & takeoff_seen & loaded_knee,
            end_step,
            ground_history=True,
            clone_executed=True,
            qualification="Successful physical mixed execution from upright ground through alternating walking, bilateral bracing and 0.20s accepted loaded knee; motor label is actual executed target",
        )
    elif args.record:
        report["full_episodes_saved"] = save(
            "isaac_ground_bridge_knee",
            loaded_knee & walked,
            end_step,
            ground_history=True,
            qualification="Physical alternating steps and >=0.25m advance at brace, then 0.20s right-knee top load >=15% body weight with both feet clear after takeoff",
        )
        report["loaded_only_episodes_saved"] = save(
            "isaac_ground_bridge_loaded_knee",
            loaded_knee & ~walked,
            end_step,
            ground_history=True,
            qualification="Continuous ground to 0.20s loaded right knee after takeoff; measured walk to brace incomplete",
        )
        report["partial_episodes_saved"] = save(
            "isaac_ground_bridge_settle",
            (partial_step >= 0) & ~loaded_knee,
            partial_step,
            ground_history=False,
            qualification="Four consecutive four-support samples after walking with >=32% reduction in prelaunch cost, speed <0.26m/s and right knee flexion >0.24rad",
        )
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        "BRIDGE_RESULT",
        json.dumps(
            {k: v for k, v in report.items() if k not in ("params", "hashes", "loaded_episodes")}
        ),
        flush=True,
    )


def main():
    args = options()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = (
        json.loads(args.settings.read_text())
        if args.settings
        else list(
            itertools.product([0.18, 0.38, 0.65], [0.0, 0.6, 1.3], [0.0, 0.65, 1.4], [0.0, 0.40])
        )
    )
    if not settings or any(
        len(x) not in (4, 5, 6, 7) or x[0] <= 0 or not np.isfinite(x).all() for x in settings
    ):
        raise ValueError("Invalid bridge settings")
    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    env = None
    try:
        from elf3_climbing.mount_teacher import (
            frozen_entry,
            ground_teacher_actor,
            knee_entry_config,
            source_inputs,
        )
        from elf3_climbing.contact_bridge_teacher import BridgeTeacher

        request, weights, files, hashes = source_inputs(args.entry)
        cfg = knee_entry_config(
            request, files, seed=args.seed, num_envs=args.num_envs, device="cuda:0"
        )
        if args.dagger_student:
            from elf3_climbing.contact_dagger import apply_intervention

            class MixedBridge(BridgeTeacher):
                def _pre_physics_step(self, actions):
                    previous = self.targets.clone()
                    super()._pre_physics_step(actions)
                    apply_intervention(self, previous, ground=True)

            env = MixedBridge(cfg, args.trajectory)
        else:
            env = BridgeTeacher(cfg, args.trajectory)
        env.mount_task.stance = "flat-foot"
        env.entry_policy = frozen_entry(weights, args.entry.parent / "ppo_config.json", env.device)
        checkpoint = torch.load(args.ground_checkpoint, map_location="cpu", weights_only=True)
        actor = ground_teacher_actor(checkpoint, cfg, env.device)
        hashes.update(
            ground_checkpoint=digest(args.ground_checkpoint),
            knee_examples=digest(args.knee_examples),
            trajectory=digest(args.trajectory),
        )
        env.attach_profile(
            args.knee_examples, [settings[i % len(settings)] for i in range(env.num_envs)]
        )
        with torch.inference_mode():
            rollout(env, actor, settings, args, hashes)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
