"""Independent review metrics for the ordered recovery candidate.
Geometry and necessary inverse dynamics remain separate acceptance categories.
"""

import argparse, json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from refine_recovery_demo import Recovery
from audit_foot_separation import audit as foot_audit
from audit_contact_wrench import audit as force_audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    r = Recovery(args.motion)
    a = r.a
    p = r.p
    N = len(r.refs)
    rows = []
    fk = []
    for i in range(N):
        if i + 1 < N:
            rot = Slerp([0, 1], Rotation.from_rotvec(r.refs[i : i + 2, 3:6]))
        for sub in range(1 if i == N - 1 else 4):
            u = sub / 4
            x = (1 - u) * r.refs[i] + u * r.refs[min(i + 1, N - 1)]
            if i + 1 < N:
                x[3:6] = rot(u).as_rotvec()
            f = r.features(x)
            feet = r.local(f["feet"])
            knees = r.local(f["knees"])
            rows.append(
                {
                    "frame": i,
                    "subframe": sub,
                    "time_s": (i + u) / float(a["fps"]),
                    "phase": str(a["phase"][i]),
                    "foot_lateral_span_m": float(feet[0, 1] - feet[1, 1]),
                    "knee_foot_lateral_offset_m": abs(knees[:, 1] - feet[:, 1]).tolist(),
                    "environment_penetration_m": float(max(0, -f["en"].min())),
                    "self_penetration_m": float(max(0, -f["ss"].min())),
                }
            )
            if sub == 0:
                fk.append({k: np.asarray(f[k]).copy() for k in ["knees", "feet", "armpts", "com"]})
    ff = foot_audit(str(r.path), args.work / "foot_separation.json")
    force_audit(
        str(r.path),
        args.work / "contact_wrench.json",
        0.6,
        list(dict.fromkeys(a["phase"].tolist())),
    )
    force = json.loads((args.work / "contact_wrench.json").read_text())
    phase_stats = []
    for phase in dict.fromkeys(a["phase"].tolist()):
        ix = np.flatnonzero(a["phase"] == phase)
        force_rows = [z for z in force["frames"] if z.get("phase") == phase]
        entry = {
            "phase": phase,
            "frames": len(ix),
            "start_s": float(ix[0] / 30),
            "end_s": float(ix[-1] / 30),
            "necessary_force_feasible": sum(
                z.get("force_and_urdf_effort_feasible", False) for z in force_rows
            ),
        }
        if phase in [
            "KNEEL_UP",
            "SHIFT_LEFT",
            "RIGHT_FOOT_LIFT",
            "RIGHT_FOOT_CLEAR",
            "RIGHT_FOOT",
            "SHIFT_RIGHT",
        ]:
            sides = [0, 1] if phase in ["KNEEL_UP", "SHIFT_LEFT"] else [0]
            entry["knee_center_xy_range_m"] = {
                str(k): np.ptp(
                    np.array([fk[i]["knees"][k] @ p.R for i in ix])[:, :2], axis=0
                ).tolist()
                for k in sides
            }
        if phase in [
            "SHIFT_RIGHT",
            "LEFT_FOOT_LIFT",
            "LEFT_FOOT_CLEAR",
            "BOTH_FEET_LOAD",
            "CENTER",
            "STAND",
            "FINISHED",
        ]:
            entry["right_sole_xy_range_m"] = np.ptp(
                np.array([fk[i]["feet"][1] @ p.R for i in ix])[:, :2], axis=0
            ).tolist()
        phase_stats.append(entry)
    rec = [z for z in rows if z["frame"] >= 202]
    transfer = [z for z in rows if z["phase"] == "RIGHT_FOOT_CLEAR"]
    ri = list(a["joint_names"]).index("r_shoulder_x_joint")
    rise = np.flatnonzero(np.isin(a["phase"], ["BOTH_FEET_LOAD", "CENTER", "STAND"]))
    src = Recovery("elf3_full_elbow_climb_demo_v6_candidate")
    preserved = {
        k: bool(np.allclose(a[k][:202], src.a[k][:202], rtol=0, atol=1e-12))
        for k in ["joint_pos", "base_pos_w", "base_quat_wxyz"]
    }
    summary = {
        "sample_rate_hz": 120,
        "foot_order_reversed_samples": sum(z["foot_lateral_span_m"] <= 0 for z in rows),
        "recovery_min_foot_span_m": min(z["foot_lateral_span_m"] for z in rec),
        "right_transfer_max_knee_foot_lateral_offset_m": max(
            z["knee_foot_lateral_offset_m"][1] for z in transfer
        ),
        "max_environment_penetration_m": max(z["environment_penetration_m"] for z in rows),
        "max_self_penetration_m": max(z["self_penetration_m"] for z in rows),
        "rise_right_shoulder_x_range_deg": np.rad2deg(
            [a["joint_pos"][rise, ri].min(), a["joint_pos"][rise, ri].max()]
        ).tolist(),
        "rise_max_right_shoulder_x_step_deg": float(
            np.rad2deg(abs(np.diff(a["joint_pos"][rise, ri])).max())
        ),
        "first_202_frames_unchanged_atol_1e_12": preserved,
        "foot_clearance": ff["summary"],
        "contact_force_screen": force["summary"],
        "closed_loop_dynamics": "NOT RUN",
        "training_ready": False,
    }
    summary["leg_path_status"] = (
        "PASS"
        if summary["foot_order_reversed_samples"] == 0
        and summary["right_transfer_max_knee_foot_lateral_offset_m"] < 0.15
        else "FAIL"
    )
    report = {
        "motion": str(r.path),
        "sha256": hashlib.sha256(r.path.read_bytes()).hexdigest(),
        "summary": summary,
        "phases": phase_stats,
        "samples": rows,
        "scope": "Knee lateral tolerance is a geometric design preference, not a hardware limit. Knee-center range is an indicator of support drift, not material contact-point slip.",
    }
    (args.work / "ordered_recovery_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summary, "phases": phase_stats}, indent=2))


if __name__ == "__main__":
    main()
