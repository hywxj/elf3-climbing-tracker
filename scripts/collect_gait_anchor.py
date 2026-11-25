"""Save physically measured, advancing lower-ground segments of a frozen gait actor."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from elf3_climbing.reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument(
        "--walk-starts", type=Path, help="Optionally include measured double-support walking starts"
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--alternating-only",
        action="store_true",
        help="Keep only physically alternating, low-slip, directionally stable clips",
    )
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--seed", type=int, default=7241)
    args = p.parse_args()
    if args.output.exists() or args.output.suffix != ".npz" or args.num_envs < 1:
        p.error("Choose a new .npz output and a positive environment count")
    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    env = None
    try:
        from elf3_climbing.contact_gait_actor import GaitContactActor
        from elf3_climbing.contact_tracker_env import ContactTrackerEnv
        from elf3_climbing.environment import TrackerCfg

        cfg = TrackerCfg()
        cfg.sim.device, cfg.scene.num_envs, cfg.seed = "cuda:0", args.num_envs, args.seed
        cfg.reference_path, cfg.segment = str(args.reference), "knee_entry"
        cfg.preload_example = cfg.knee_transfer_guide = None
        cfg.knee_support_rule, cfg.knee_wrist_contact = "assisted-push", "tabletop"
        cfg.random_start = cfg.curriculum_enabled = False

        class AnchorEnv(ContactTrackerEnv):
            def _pre_physics_step(self, actions):
                super()._pre_physics_step(actions)
                self.anchor_drive = self.targets.clone()

        env = AnchorEnv(cfg)
        if args.walk_starts:
            env.load_demonstration_starts([args.walk_starts])
            env.stage_training = True
        actor = GaitContactActor().to(env.device).eval()
        actor.load_state_dict(torch.load(args.weights, weights_only=True, map_location=env.device))
        with torch.inference_mode():
            collect(env, actor, args)
    finally:
        if env is not None:
            env.close()
        app.close()


def collect(env, actor, args):
    from elf3_climbing.contact_examples import CONTACT_NAMES, student_contact_state
    from elf3_climbing.contact_tracker_env import student_physical_state

    obs, _ = env.reset(seed=args.seed)
    upright_start = env.ground_start.cpu().numpy().copy()
    k = env.knee_contacts
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    length = torch.zeros_like(alive, dtype=torch.long)
    rows = {}
    for step in range(env.max_episode_length):
        sites, force, points = student_contact_state(env, with_points=True)
        if step == 0:
            force.zero_()
            points.fill_(torch.nan)
        raw = student_physical_state(env, sites, force)
        raw.update(
            root_pose=torch.cat((raw["root_pos"], raw["root_quat"]), dim=1),
            student_site_pos=raw.pop("site_pos"),
            student_contact_force=raw.pop("contact_force"),
            student_contact_point=points,
            contact_observed=torch.full_like(alive, step != 0),
        )
        # Stop BEFORE the leading foot enters the last 0.24 m to the deck lip.
        clearance = -((sites[:, :2] - k.trace_edge) @ k.trace_inward).amax(dim=1)
        alive &= (clearance >= 0.24) & (force[:, 2:].amax(dim=1) <= 3)
        if args.alternating_only:
            alive &= -raw["projected_gravity_b"][:, 2] >= 0.88
        if not alive.any():
            break
        length += alive.long()
        raw["leading_sole_to_edge_m"] = clearance
        action, logits, locations = actor.action_and_plan(obs["policy"])
        raw["predicted_contact_probability"] = logits.sigmoid()
        raw["predicted_contact_point_b"] = locations
        for key, value in raw.items():
            rows.setdefault(key, []).append(value.cpu().numpy().copy())
        obs, _, failed, expired, _ = env.step(action)
        m = env.last_metrics
        loaded = m["foot_ground_patch_N"] > 20
        share = m["foot_ground_patch_N"] * loaded
        slip = (share * m["foot_ground_slip_m_s"]).sum(1) / share.sum(1).clamp_min(1)
        for key, value in dict(
            drive_target=env.anchor_drive,
            loaded_slip=slip,
            loaded=loaded.any(1),
            walk_switches=m["walk_switches"],
            foot_advance_m=m["advance_m"],
        ).items():
            rows.setdefault(key, []).append(value.cpu().numpy().copy())
        alive &= ~failed & ~expired
    if not rows:
        raise RuntimeError("No lower-ground anchor states before the margin")
    data = {name: np.stack(column) for name, column in rows.items()}
    counts = length.cpu().numpy()
    accepted = []
    alternating_before_margin = 0
    forward = env.advance_direction.cpu().numpy()
    lateral = np.array([-forward[1], forward[0]])
    for i, original_end in enumerate(counts.copy()):
        end = original_end
        if args.alternating_only:
            # Keep the continuous start-to-landing clip, even if the student later
            # drifts or falls. Never retain a correction or hop as a walking success.
            cum_switch = data["walk_switches"][:end, i]
            switch_steps = np.flatnonzero(np.diff(cum_switch, prepend=0) > 0)
            if not len(switch_steps):
                continue
            alternating_before_margin += 1
            xy = data["root_pos"][:end, i, :2]
            bad = np.flatnonzero(np.abs((xy - xy[0]) @ lateral) > 0.16)
            if len(bad):
                end = int(bad[0])
            # Only count switches already observed in a recorded physical state.
            # The landing load was measured after the preceding motor target.
            switch_steps = switch_steps[switch_steps + 1 < end]
            if not len(switch_steps):
                continue
            end = min(end, int(switch_steps[-1]) + 1 + round(0.12 / env.step_dt))
            counts[i] = end
        if end * env.step_dt < 1.0:
            continue
        loaded = data["loaded"][:end, i]
        slip = data["loaded_slip"][:end, i][loaded]
        up = -data["projected_gravity_b"][:end, i, 2]
        advance = float(data["foot_advance_m"][end - 1, i])
        switches = int(data["walk_switches"][end - 1, i])
        xy = data["root_pos"][:end, i, :2]
        delta = xy - xy[0]
        root_forward = float(delta[-1] @ forward)
        lateral_excursion = float(np.abs(delta @ lateral).max())
        if args.alternating_only and (root_forward < 0.12 or lateral_excursion > 0.16):
            continue
        if (
            len(slip)
            and loaded.mean() >= 0.9
            and advance >= 0.12
            and up.min() >= 0.88
            and slip.mean() <= 0.10
            and np.quantile(slip, 0.95) <= 0.22
        ):
            accepted.append(
                dict(
                    id=i,
                    upright_start=bool(upright_start[i]),
                    switches=switches,
                    advance=advance,
                    root_advance=root_forward,
                    lateral_excursion=lateral_excursion,
                    slip=float(slip.mean()),
                    slip_p95=float(np.quantile(slip, 0.95)),
                )
            )
    # Keep every genuine alternating gait, plus at most eight quiet advancing
    # lower-ground rollouts as a stability anchor. Record the distinction.
    strong = [e for e in accepted if e["switches"] > 0]
    quiet = sorted((e for e in accepted if e["switches"] == 0), key=lambda e: e["slip"])[:8]
    if args.alternating_only:
        quiet = []
    kept = strong + quiet
    if not kept:
        raise RuntimeError(
            f"No verified clips: switched before the far-ground cutoff {alternating_before_margin}/{env.num_envs}, "
            f"met posture, advance and slip requirements {len(accepted)}"
        )
    ids = [e["id"] for e in kept]
    arrays = {
        name: np.concatenate([column[: counts[i], i] for i in ids]) for name, column in data.items()
    }
    report = dict(
        source_kind="isaac_student_gait_alternating"
        if args.alternating_only
        else "gait_anchor_far",
        attempts=env.num_envs,
        switched_before_cutoff=alternating_before_margin if args.alternating_only else None,
        low_slip_advancing=len(accepted),
        true_alternating_retained=len(strong),
        stability_only_retained=len(quiet),
        clips=kept,
        seed=args.seed,
        weights_sha256=digest(args.weights),
        reference_sha256=digest(args.reference),
        qualification="Ground-only stability prior, >=0.24m from lip; any unconfirmed alternations are recorded as such",
    )
    if args.walk_starts:
        report["walking_starts_sha256"] = digest(args.walk_starts)
    if args.alternating_only:
        report["qualification"] = (
            "Observed alternating loaded landings, advancing feet and root >=0.12 m, lateral displacement <=0.16 m, mean stance slip <=0.10 m/s and p95 <=0.22 m/s, >=0.24 m from ledge"
        )
    arrays.update(
        episode_id=np.repeat(ids, counts[ids]),
        step=np.concatenate([np.arange(counts[i], dtype=np.int32) for i in ids]),
        step_dt=np.asarray(env.step_dt),
        source_kind=np.array(report["source_kind"]),
        schema=np.array("elf3_contact_examples_v1"),
        student_site_names=np.array(CONTACT_NAMES),
        nominal=env.nominal[0].cpu().numpy(),
        joint_names=env.reference_np["joint_names"],
        provenance=np.array(json.dumps(report, allow_nan=False)),
    )
    for key in ("terrain_box_centers_w", "terrain_box_half_sizes", "terrain_box_quat_wxyz"):
        arrays[key] = env.reference_np[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        path = Path(tmp) / "anchor.npz"
        np.savez_compressed(path, **arrays)
        os.link(path, args.output)
    print("GAIT_ANCHOR", json.dumps(report), args.output, flush=True)


if __name__ == "__main__":
    main()
