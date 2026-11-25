"""Check exported data with fresh ELF3 FK and collision queries; no dynamics."""

from pathlib import Path
import argparse, contextlib, io, json, tempfile, sys, xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from view_motion import load, model_file, terrain, ASSET, SOURCE
from screen_support import screen


def limb(body):
    if body.startswith(("l_", "r_")):
        if body[2:].startswith(("shoulder", "elbow", "wrist")):
            return body[:2] + "arm"
        if body[2:].startswith(("hip", "knee", "ankle")):
            return body[:2] + "leg"
    return body


def longest_bout(mask, fps):
    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return float((ends - starts).max() / fps) if len(starts) else 0.0


def inspect(path, required_support="forearm", include_gripper=True):
    if required_support not in ("forearm", "wrist_baseline"):
        raise ValueError("Unknown support requirement")
    path, a = load(str(path))
    n = len(a["timestamps"])
    names = a["joint_names"].tolist()
    fps = float(a["fps"])
    limits = {
        j.get("name"): np.array(
            [float(j.find("limit").get(k)) for k in ("lower", "upper", "velocity")]
        )
        for j in ET.parse(ASSET / "urdf/elf3.urdf").getroot().findall("joint")
    }
    bounds = np.array([limits[s] for s in names])
    violations = int(
        np.count_nonzero(
            (a["joint_pos"] < bounds[:, 0] - 1e-5) | (a["joint_pos"] > bounds[:, 1] + 1e-5)
        )
    )
    speed_violations = int(np.count_nonzero(np.abs(a["joint_vel"]) > bounds[:, 2] + 1e-4))
    derivative_error = float(
        np.max(np.abs(np.diff(a["joint_pos"], axis=0) * fps - a["joint_vel"][:-1]))
    )
    env_depth = []
    self_depth = []
    feet = []
    actual_sites = []
    pelvis_positions = []
    pelvis_rotations = []
    limb_rotations = []
    forearm_elbows = []
    forearm_wrists = []
    gripper_env_depth = []
    gripper_self_depth = []
    from gripper_geometry import specification

    gripper_cfg = specification()
    contact_frames = a.get(
        "contact_frame_names",
        np.array(["l_ankle_x_link", "r_ankle_x_link", "l_wrist_z_link", "r_wrist_z_link"]),
    ).tolist()
    local_points = a.get("contact_local_positions")
    source_row = next(
        r
        for r in json.loads((SOURCE / "index.json").read_text())["motions"]
        if r["id"] == str(a["source_id"])
    )
    with contextlib.redirect_stdout(io.StringIO()):
        import retarget as rt
    source_names, source_pos, source_rot, rest, source_fps, human, aux = rt.source_motion(
        source_row, rt.G1Kinematics()
    )
    mapped = rt.pairs(human)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="elf3_check_") as folder:
        xml = model_file(folder, a, True, include_gripper)
        m = mujoco.MjModel.from_xml_path(str(xml))
        d = mujoco.MjData(m)
        ids = np.array([m.jnt_qposadr[m.joint(s).id] for s in names])
        gripper_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gripper_cfg["geom_name"])
        calibration, _, _ = rt.calibration(
            m, rt.UrdfKinematics(ASSET / "urdf/elf3.urdf", names), source_names, rest, human
        )
        # Earlier v4 exports did not store these local points. Recover them
        # once, then independently check that every saved point follows FK.
        inferred_local_points = local_points is None
        for frame in range(n):
            d.qpos[:7] = a["qpos"][frame, :7]
            d.qpos[ids] = a["joint_pos"][frame]
            mujoco.mj_forward(m, d)
            body_ids = [m.body(s).id for s in contact_frames]
            if local_points is None:
                local_points = np.array(
                    [
                        d.xmat[b].reshape(3, 3).T @ (p - d.xpos[b])
                        for b, p in zip(body_ids, a["contact_sites_w"][frame])
                    ]
                )
            actual_sites.append(
                [d.xpos[b] + d.xmat[b].reshape(3, 3) @ p for b, p in zip(body_ids, local_points)]
            )
            pelvis_positions.append(d.xpos[m.body("waist_z_link").id].copy())
            pelvis_rotations.append(d.xmat[m.body("waist_z_link").id].reshape(3, 3).copy())
            limb_rotations.append(
                [d.xmat[m.body(target).id].reshape(3, 3).copy() for target, _, _, _ in mapped]
            )
            forearm_elbows.append(
                [d.xpos[m.body(s + "_elbow_y_link").id].copy() for s in ("l", "r")]
            )
            forearm_wrists.append(
                [d.xpos[m.body(s + "_wrist_z_link").id].copy() for s in ("l", "r")]
            )
            env = 0.0
            self_collision = 0.0
            grip_env = 0.0
            grip_self = 0.0
            for c in d.contact:
                bodies = [int(m.geom_bodyid[int(g)]) for g in (c.geom1, c.geom2)]
                depth = max(0.0, -float(c.dist))
                if 0 in bodies:
                    env = max(env, depth)
                elif limb(m.body(bodies[0]).name) != limb(m.body(bodies[1]).name):
                    self_collision = max(self_collision, depth)
                if gripper_id >= 0 and gripper_id in (int(c.geom1), int(c.geom2)):
                    if 0 in bodies:
                        grip_env = max(grip_env, depth)
                    else:
                        grip_self = max(grip_self, depth)
            env_depth.append(env)
            self_depth.append(self_collision)
            gripper_env_depth.append(grip_env)
            gripper_self_depth.append(grip_self)
            feet.append(
                [
                    d.xpos[m.body(s + "_ankle_x_link").id]
                    + d.xmat[m.body(s + "_ankle_x_link").id].reshape(3, 3)
                    @ np.array([0.03, 0, -0.04])
                    for s in ("l", "r")
                ]
            )
    root_step = float(np.linalg.norm(np.diff(a["base_pos_w"], axis=0), axis=1).max())
    qstep = float(np.abs(np.diff(a["joint_pos"], axis=0)).max())
    rotations = Rotation.from_quat(a["base_quat_wxyz"], scalar_first=True)
    angular_step = float((rotations[:-1].inv() * rotations[1:]).magnitude().max())
    contacts = a.get("geometric_contact_mask", np.zeros((n, 4), bool))
    intended = a.get("intended_contact_mask", contacts)
    actual_sites = np.array(actual_sites)
    contact_error = np.linalg.norm(actual_sites - a["contact_targets_w"], axis=-1)
    contact_fk_error = float(np.linalg.norm(actual_sites - a["contact_sites_w"], axis=-1).max())
    realized_contacts = intended & (contact_error < 0.005)
    contact_mask_matches = bool(np.array_equal(contacts, realized_contacts))
    contacts = realized_contacts
    source_indices = a.get("source_frame_indices", np.arange(n))
    mapping_ok = bool(
        source_indices[0] == 0
        and np.all((np.diff(source_indices) >= 0) & (np.diff(source_indices) <= 1))
    )
    expected_frames = (
        source_row["frames"]
        if source_row["dataset"] == "OmniRetarget"
        else (source_row["frames"] + 3) // 4
    )
    mapping_ok = mapping_ok and int(source_indices[-1]) == expected_frames - 1
    source_indices = np.minimum(source_indices, len(source_pos) - 1)
    if "source_world_translation" in a:
        translation = a["source_world_translation"]
    else:
        # Same constant neutral sole-height convention, recomputed from the
        # source skeleton and target model, without trusting exported errors.
        _, seed, scale = rt.calibration(
            m, rt.UrdfKinematics(ASSET / "urdf/elf3.urdf", names), source_names, rest, human
        )
        d.qpos[:] = seed
        mujoco.mj_forward(m, d)
        target_height = np.mean(
            [
                (
                    d.xpos[m.body(s + "_ankle_x_link").id]
                    + d.xmat[m.body(s + "_ankle_x_link").id].reshape(3, 3)
                    @ np.array([0.03, 0, -0.04])
                )[2]
                - d.xpos[m.body("waist_z_link").id, 2]
                for s in ("l", "r")
            ]
        )
        translation = np.array(
            [0.0, 0.0, np.mean(aux["rest_feet"][:, 2] - rest[0][0, 2]) * scale - target_height]
        )
    pelvis_error = np.linalg.norm(
        np.array(pelvis_positions)
        - (source_pos[:, 0] * float(a["morphology_scale"]) + translation)[source_indices],
        axis=1,
    )
    limb_errors = {}
    for column, (target, src, _, _) in enumerate(mapped):
        offset = Rotation.from_quat(
            calibration["ik_match_table1"][target][4], scalar_first=True
        ).as_matrix()
        desired = source_rot[source_indices, source_names.index(src)] @ offset
        error = Rotation.from_matrix(
            np.transpose(desired, (0, 2, 1)) @ np.array(limb_rotations)[:, column]
        ).magnitude()
        limb_errors[target] = {
            "p95_deg": float(np.degrees(np.percentile(error, 95))),
            "max_deg": float(np.degrees(error.max())),
        }
        if target == "waist_z_link":
            pelvis_angle_error = error
    intended_errors = []
    for k in range(4):
        values = contact_error[intended[:, k], k]
        intended_errors.append(
            {
                "p95_m": float(np.percentile(values, 95)) if len(values) else None,
                "max_m": float(values.max()) if len(values) else None,
            }
        )
    # Proposed offline gates; a PASS here certifies only these stated checks.
    reasons = []
    if violations:
        reasons.append("joint limits")
    if speed_violations:
        reasons.append("URDF velocity limits")
    if derivative_error > 1e-5:
        reasons.append("velocity/position inconsistency")
    if max(env_depth) > 0.003:
        reasons.append("environment penetration >3mm")
    if max(self_depth) > 0.003:
        reasons.append("non-neighbor self penetration >3mm")
    if root_step > 0.061 or angular_step > 0.35 or qstep > 0.50:
        reasons.append("large per-frame change")
    if not mapping_ok:
        reasons.append("source frame mapping")
    if contact_fk_error > 1e-7 or not contact_mask_matches:
        reasons.append("saved contact data disagrees with fresh FK")
    if np.percentile(pelvis_angle_error, 95) > np.deg2rad(15):
        reasons.append("pelvis orientation p95 >15deg")
    if np.percentile(pelvis_error, 95) > 0.12:
        reasons.append("pelvis position p95 >12cm")
    if pelvis_error.max() > 0.25:
        reasons.append("pelvis position max >25cm")
    if pelvis_angle_error.max() > np.deg2rad(30):
        reasons.append("pelvis orientation max >30deg")
    if any(
        v["p95_deg"] > 30 or v["max_deg"] > 60
        for name, v in limb_errors.items()
        if name not in ("waist_z_link", "torso_link")
    ):
        reasons.append("mapped limb orientation p95 >30deg or max >60deg")
    if any(
        v["p95_m"] is not None and (v["p95_m"] > 0.03 or v["max_m"] > 0.10) for v in intended_errors
    ):
        reasons.append("intended contact error p95 >3cm or max >10cm")
    foot_top = np.zeros((n, 2), bool)
    same_box_feet = np.zeros(n, bool)
    feet = np.array(feet)
    if "terrain_box_centers_w" in a:
        for center, quat, size in zip(
            a["terrain_box_centers_w"], a["terrain_box_quat_wxyz"], a["terrain_box_half_sizes"]
        ):
            R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
            if center[2] + R[2, 2] * size[2] < 0.15:
                continue
            local = (feet - center) @ R
            near = np.clip(local, -size, size)
            near[:, :, 2] = size[2]
            gap = np.linalg.norm(local - near, axis=-1)
            foot_top |= gap < 0.05
            same_box_feet |= (gap < 0.05).all(1)
    if np.any(foot_top.sum(0) < max(3, round(0.10 * fps))):
        reasons.append("both feet have not reached the platform top")
    # Several originals climb up and then walk down. Verify a completed top
    # phase anywhere in the full clip, rather than requiring it at the end.
    dual_foot_bout = longest_bout(same_box_feet, fps)
    if dual_foot_bout < 0.10:
        reasons.append("no sustained dual-foot phase on one platform")
    dual_wrist_s = float(np.all(contacts[:, 2:], axis=1).sum() / fps)
    dual_wrist_bout = longest_bout(np.all(contacts[:, 2:], axis=1), fps)
    if dual_wrist_bout < 0.10:
        reasons.append("no sustained dual upper-limb geometry contact")
    tolerance = gripper_cfg["collision_tolerance_mm"] / 1000
    grip_env_bad = np.flatnonzero(np.array(gripper_env_depth) > tolerance)
    grip_self_bad = np.flatnonzero(np.array(gripper_self_depth) > tolerance)
    if len(grip_env_bad):
        reasons.append("left gripper envelope intersects terrain")
    if len(grip_self_bad):
        reasons.append("left gripper envelope intersects robot")
    legacy_status = "PASS" if not reasons else "NEEDS_REVIEW"
    forearm = screen(forearm_elbows, forearm_wrists, fps, terrain(a))
    task_status = "BLOCKED" if forearm["geometric_candidate"] else "FAIL"
    if required_support == "forearm":
        if not forearm["geometric_candidate"]:
            reasons.append("required dual elbow-to-wrist placement absent")
        else:
            reasons.append("forearm surface contact verification not implemented")
    status = (
        legacy_status
        if required_support == "wrist_baseline"
        else ("BLOCKED" if task_status == "BLOCKED" else "NEEDS_REVIEW")
    )
    gripper_checked = include_gripper and gripper_cfg["enabled"]
    return {
        "status": status,
        "validation_revision": 5,
        "required_support": required_support,
        "left_gripper_envelope_check": {
            "status": (
                "INTERSECTION"
                if len(grip_env_bad) or len(grip_self_bad)
                else "CLEAR_FOR_ASSUMED_ENVELOPE"
            )
            if gripper_checked
            else "NOT RUN",
            "config_sha256": gripper_cfg["config_sha256"],
            "dimensions_mm": gripper_cfg["dimensions_mm"],
            "center_in_wrist_m": gripper_cfg["center_in_wrist_m"],
            "mount_assumption": gripper_cfg["mount_assumption"],
            "max_environment_penetration_m": max(gripper_env_depth),
            "max_robot_penetration_m": max(gripper_self_depth),
            "environment_intersection_frames": int(len(grip_env_bad)),
            "robot_intersection_frames": int(len(grip_self_bad)),
            "first_environment_intersection_s": float(grip_env_bad[0] / fps)
            if len(grip_env_bad)
            else None,
            "collision_tolerance_m": tolerance,
            "scope": "box-envelope geometry only; mounting assumption remains provisional",
        },
        "legacy_wrist_kinematic_geometry_status": legacy_status,
        "forearm_task_status": task_status,
        "forearm_geometry_screen": forearm,
        "forearm_surface_contact_status": "NOT RUN",
        "scope": "export integrity, model limits, continuity, geometry, mapped body orientation and source-pelvis tracking only; not dynamics or complete motion acceptance",
        "thresholds": {
            "penetration_m": 0.003,
            "pelvis_position_p95_m": 0.12,
            "pelvis_position_max_m": 0.25,
            "pelvis_orientation_p95_deg": 15,
            "pelvis_orientation_max_deg": 30,
            "limb_orientation_p95_deg": 30,
            "limb_orientation_max_deg": 60,
            "intended_contact_error_p95_m": 0.03,
            "intended_contact_error_max_m": 0.10,
            "geometric_contact_gap_m": 0.005,
            "dual_wrist_bout_s": 0.10,
            "dual_foot_top_bout_s": 0.10,
            "foot_top_gap_m": 0.05,
        },
        "file": path.name,
        "frames": n,
        "fps": fps,
        "joint_limit_violations": violations,
        "velocity_violations": speed_violations,
        "velocity_derivative_error": derivative_error,
        "max_environment_penetration_m": max(env_depth),
        "max_non_neighbor_self_penetration_m": max(self_depth),
        "self_collision_scope": "different limbs and torso; mechanical neighbors within one arm/leg excluded",
        "max_root_step_m": root_step,
        "max_root_angular_step_rad": angular_step,
        "max_joint_step_rad": qstep,
        "source_frame_mapping_ok": mapping_ok,
        "geometric_contact_frames": contacts.sum(0).tolist(),
        "intended_contact_frames": intended.sum(0).tolist(),
        "contact_fk_max_error_m": contact_fk_error,
        "contact_local_points_inferred": inferred_local_points,
        "intended_contact_errors": intended_errors,
        "mapped_body_orientation_error": limb_errors,
        "max_realized_contact_point_error_m": float(contact_error[contacts].max())
        if contacts.any()
        else None,
        "dual_wrist_geometry_contact_s": dual_wrist_s,
        "foot_top_geometry_frames": foot_top.sum(0).tolist(),
        "longest_dual_wrist_geometry_contact_s": dual_wrist_bout,
        "longest_dual_foot_top_phase_s": dual_foot_bout,
        "pelvis_position_p95_m": float(np.percentile(pelvis_error, 95)),
        "pelvis_orientation_p95_deg": float(np.degrees(np.percentile(pelvis_angle_error, 95))),
        "reasons": reasons,
        "dynamics_status": "NOT RUN",
        "contact_force_status": "NOT RUN",
        "elbow_contact_status": "NOT RUN",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("motion")
    p.add_argument("--support", choices=["forearm", "wrist_baseline"], default="forearm")
    p.add_argument("--no-gripper", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            inspect(args.motion, args.support, not args.no_gripper), indent=2, ensure_ascii=False
        )
    )
