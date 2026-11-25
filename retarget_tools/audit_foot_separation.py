"""Independent positive foot-clearance check, including interpolated frames.
The 30 mm default is a candidate design margin, not a hardware-certified limit.
"""

import argparse, ast, itertools, json, hashlib
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from plan_knee_entry import create_context
from view_motion import load

# Reuse the existing independent convex distance and separating-plane checker
# without executing the original refiner's command-line entry point.
tree = ast.parse(Path(__file__).with_name("refine_climb_clearance.py").read_text())
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in ["closest", "gjk"]
            ],
            type_ignores=[],
        ),
        "existing_convex_distance",
        "exec",
    )
)


def audit(motion, output, margin=0.03, substeps=4):
    p = create_context()
    m = p.m
    d = p.d
    path, a = load(motion)
    fg = [m.geom("hiking_" + s + "_ankle_x_link_0").id for s in ["l", "r"]]
    fps = float(a["fps"])
    rows = []
    for i in range(len(a["timestamps"])):
        last = i == len(a["timestamps"]) - 1
        if not last:
            rot = Slerp(
                [0, 1], Rotation.from_quat(a["base_quat_wxyz"][i : i + 2], scalar_first=True)
            )
        for k in range(1 if last else substeps):
            u = k / substeps
            j = min(i + 1, len(a["timestamps"]) - 1)
            d.qpos[:3] = (1 - u) * a["base_pos_w"][i] + u * a["base_pos_w"][j]
            d.qpos[3:7] = a["base_quat_wxyz"][i] if last else rot([u]).as_quat(scalar_first=True)[0]
            d.qpos[p.ids] = (1 - u) * a["joint_pos"][i] + u * a["joint_pos"][j]
            mujoco.mj_forward(m, d)
            A, B = [p.hulls[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in fg]
            distance, lower = gjk(A, B, certificate=True)  # noqa: F821 — loaded by AST above
            depth = 0.0
            if distance < 1e-7:
                h = ConvexHull((A[:, None, :] - B[None, :, :]).reshape(-1, 3))
                depth = max(0.0, float(-h.equations[:, 3].max()))
            rows.append(
                dict(
                    frame=i,
                    subframe=k,
                    time_s=(i + u) / fps,
                    phase=str(a["phase"][i]),
                    distance_m=distance,
                    separation_lower_bound_m=lower,
                    convex_overlap_depth_m=depth,
                )
            )
    summary = dict(
        status="PASS" if all(r["separation_lower_bound_m"] >= margin for r in rows) else "FAIL",
        samples=len(rows),
        sample_rate_hz=fps * substeps,
        required_margin_m=margin,
        minimum_gap_m=min(r["distance_m"] for r in rows),
        maximum_convex_overlap_depth_m=max(r["convex_overlap_depth_m"] for r in rows),
        margin_violation_samples=sum(r["separation_lower_bound_m"] < margin for r in rows),
        positive_overlap_samples=sum(r["convex_overlap_depth_m"] > 1e-7 for r in rows),
        scope="Both complete ankle/foot convex collision meshes; linear joint/root-position interpolation with quaternion slerp. Positive margin is required even for nonpenetrating poses. No dynamics claim.",
    )
    report = {
        "summary": summary,
        "motion": str(path),
        "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": rows,
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    audit(args.motion, args.output, args.margin)
