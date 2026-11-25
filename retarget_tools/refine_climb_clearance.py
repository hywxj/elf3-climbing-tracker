"""Refine the full ELF3 geometric demo: arm/leg clearance and compact timing.
The collision model, mesh-hull distance checks, and source demo are read-only.
This does not validate forces, actuator torques, or dynamic execution.
"""

from project_paths import TOOLS_DIR
import argparse, tempfile, shutil, subprocess, hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", default="elf3_full_elbow_climb_demo")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--reuse-states", type=Path, help="Audit/export already solved planner states")
args = parser.parse_args()
if args.output.exists() or args.output.with_suffix(".checks.json").exists():
    parser.error("Output already exists")
args.output.parent.mkdir(parents=True, exist_ok=True)
work = tempfile.TemporaryDirectory(prefix="elf3_clearance_")
work_dir = Path(work.name)
import sys, json, itertools
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(TOOLS_DIR))
from plan_knee_entry import create_context
from view_motion import load

p = create_context()
m = p.m
d = p.d
source_path, a = load(args.motion)
arms = [
    m.geom("hiking_" + s + "_" + part + "_link_0").id
    for s in ["l", "r"]
    for part in ["shoulder_z", "wrist_x", "wrist_y", "wrist_z"]
]
legs = [
    m.geom("hiking_" + s + "_" + part + "_link_0").id
    for s in ["l", "r"]
    for part in ["hip_z", "knee_y", "ankle_x"]
]
pairs = list(itertools.product(arms, legs))


def setframe(i):
    d.qpos[:7] = a["qpos"][i, :7]
    d.qpos[p.ids] = a["joint_pos"][i]
    mujoco.mj_forward(m, d)


def vertices(g):
    return p.hulls[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def closest(simplex):
    best = None
    for k in range(1, min(4, len(simplex)) + 1):
        for ids in itertools.combinations(range(len(simplex)), k):
            pts = simplex[list(ids)]
            if k == 1:
                weights = np.ones(1)
            else:
                B = (pts[1:] - pts[0]).T
                v = np.linalg.lstsq(B, -pts[0], rcond=None)[0]
                weights = np.r_[1 - v.sum(), v]
            if weights.min() < -1e-9:
                continue
            q = weights @ pts
            dist = q @ q
            if best is None or dist < best[0]:
                best = (dist, q, pts[weights > 1e-9])
    return best[1], best[2]


def gjk(A, B, certificate=False):
    def result(v):
        distance = float(np.linalg.norm(v))
        lower = (
            0.0
            if distance < 1e-12
            else max(0.0, float(np.min(A @ (v / distance)) - np.max(B @ (v / distance))))
        )
        return (distance, lower) if certificate else distance

    def support(v):
        return A[np.argmax(A @ v)] - B[np.argmin(B @ v)]

    v = A.mean(0) - B.mean(0)
    S = np.array([support(-v)])
    v = S[0]
    for _ in range(50):
        vv = v @ v
        if vv < 1e-16:
            return result(v)
        w = support(-v)
        if vv - v @ w < 1e-10:
            return result(v)
        if np.min(np.linalg.norm(S - w, axis=1)) < 1e-10:
            return result(v)
        v, S = closest(np.vstack((S, w)))
    return result(v)


for field in ["terrain_box_centers_w", "terrain_box_quat_wxyz", "terrain_box_half_sizes"]:
    if not np.allclose(a[field], p.a[field]):
        raise ValueError("This refiner requires the existing widened demo terrain")


def refine():
    import sys, json
    from pathlib import Path
    import numpy as np, mujoco
    from scipy.optimize import least_squares
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation

    names = p.names
    waist = m.body("waist_z_link").id
    iy = 6 + names.index("l_wrist_y_joint")
    lo = np.r_[[-1.8, -0.5, 0.6], [-0.8, -0.3, -0.8], m.jnt_range[p.ji, 0]]
    hi = np.r_[[0.5, 0.5, 2.1], [0.8, 1.8, 0.8], m.jnt_range[p.ji, 1]]
    kg = [m.geom("hiking_" + s + "_hip_z_link_0").id for s in ["l", "r"]]
    sg = [m.geom("hiking_" + s + "_knee_y_link_0").id for s in ["l", "r"]]
    fg = [m.geom("hiking_" + s + "_ankle_x_link_0").id for s in ["l", "r"]]
    eg = [m.geom("hiking_" + s + "_shoulder_z_link_0").id for s in ["l", "r"]]
    eb = [m.body(s + "_elbow_y_link").id for s in ["l", "r"]]

    def features(x):
        p.setq(x)
        en, ss, v = p.geometry()
        return dict(
            pelvis=d.xpos[waist].copy(),
            knees=d.xpos[p.knees].copy(),
            feet=np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ p.sole for b in p.feet]),
            feetrot=d.xmat[p.feet].reshape(2, 3, 3).copy(),
            armpts=np.array([p.armpts @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b in p.arms]),
            gripper_dir=d.xmat[m.body("l_wrist_z_link").id].reshape(3, 3)[:, 0].copy(),
            vertices=v,
            environment=en,
            selfdist=ss,
        )

    def region_gap(f, k, elbow=False):
        v = f["vertices"][eg[k]] if elbow else np.vstack([f["vertices"][g] for g in [kg[k], sg[k]]])
        c = d.xpos[eb[k]] if elbow else f["knees"][k]
        mask = np.linalg.norm(v - c, axis=1) < 0.105
        return float(v[mask, 2].min() - p.top) if mask.any() else 1.0

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    oldx = []
    for i in range(len(a["timestamps"])):
        xyz = p.R.T @ (a["base_pos_w"][i] - p.box["center"])
        xyz[2] = a["base_pos_w"][i, 2]
        rot = Rotation.from_matrix(
            p.R.T @ Rotation.from_quat(a["base_quat_wxyz"][i], scalar_first=True).as_matrix()
        ).as_rotvec()
        oldx.append(np.r_[xyz, rot, a["joint_pos"][i]])
    oldx = np.array(oldx)
    phase_names = list(dict.fromkeys(a["phase"].tolist()))
    starts = [int(np.flatnonzero(a["phase"] == phase)[0]) for phase in phase_names]
    oldtimes = np.r_[np.array(starts) / 30.0, a["timestamps"][-1]]
    durations = {
        "READY": 0.3,
        "GRIPPER_FLIP": 1.15,
        "PREPARED": 0.2,
        "WALK_LEFT_CLEAR": 0.45,
        "WALK_LEFT_LAND": 0.45,
        "WALK_RIGHT_CLEAR": 0.45,
        "APPROACH_READY": 0.45,
        "ARM_HOVER": 0.8,
        "FOREARMS": 0.7,
        "FOREARM_HOLD": 0.30,
        "RAISE_AND_TUCK": 0.9,
        "RIGHT_KNEE": 0.9,
        "LEFT_KNEE_CLEAR": 0.9,
        "BOTH_KNEES": 0.9,
        "KNEEL_UP": 1.0,
        "RIGHT_FOOT_CLEAR": 0.85,
        "RIGHT_FOOT": 0.65,
        "LEFT_FOOT_CLEAR": 0.85,
        "CROUCH": 0.65,
        "STAND": 1.0,
        "FINISHED": 0.6,
    }
    newtimes = np.r_[0.0, np.cumsum([durations[n] for n in phase_names])]
    N = round(newtimes[-1] * 30) + 1
    times = np.arange(N) / 30.0
    newtimes *= times[-1] / newtimes[-1]
    source_times = PchipInterpolator(newtimes, oldtimes)(times)
    refs = np.array([np.interp(source_times, a["timestamps"], oldx[:, j]) for j in range(35)]).T
    orig_ix = np.minimum(
        np.searchsorted(a["timestamps"], source_times, side="right") - 1, len(oldx) - 1
    )
    phases = a["phase"][orig_ix]
    T = {phase: float(a["timestamps"][start]) for phase, start in zip(phase_names, starts)}
    states = []
    records = []
    previous = None
    for i, (t, ref, phase) in enumerate(zip(source_times, refs, phases)):
        target = features(ref)
        outward = 0.10 * smooth((t - T["ARM_HOVER"]) / (T["FOREARMS"] - T["ARM_HOVER"]))
        outward *= 1 - 0.4 * smooth((t - T["KNEEL_UP"]) / (T["RIGHT_FOOT_CLEAR"] - T["KNEEL_UP"]))
        outward *= 1 - smooth((t - T["CROUCH"]) / (T["STAND"] + 1.2 - T["CROUCH"]))
        targetarms = target["armpts"].copy()
        for k, sign in enumerate([1, -1]):
            targetarms[k] += p.R[:, 1] * sign * outward
        targetknees = target["knees"].copy()
        inward = (
            0.018
            * smooth((t - T["RAISE_AND_TUCK"]) / 0.6)
            * (1 - smooth((t - T["KNEEL_UP"]) / 1.0))
        )
        for k, sign in enumerate([1, -1]):
            targetknees[k] -= p.R[:, 1] * sign * inward
        feetplane = {}
        armplane = []
        kneeplane = []
        elbowplane = []
        for k, g in enumerate(fg):
            z = target["vertices"][g][:, 2].min()
            if abs(z - p.floor) < 0.0015:
                feetplane[k] = p.floor
            elif abs(z - p.top) < 0.0015:
                feetplane[k] = p.top
        for k, g in enumerate(p.armgeoms):
            if (
                abs(target["vertices"][g][:, 2].min() - p.top) < 0.0015
                and T["FOREARMS"] <= t < T["KNEEL_UP"]
            ):
                armplane.append(k)
        for k in range(2):
            if abs(region_gap(target, k)) < 0.0015 and T["RIGHT_KNEE"] < t < T["LEFT_FOOT_CLEAR"]:
                kneeplane.append(k)
            if phase == "FOREARM_HOLD":
                elbowplane.append(k)
        currentdist = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.15, None) for g, h in pairs])
        need = (
            outward > 1e-7
            or currentdist.min() < 0.035
            or max(0, -target["environment"].min(), -target["selfdist"].min()) > 0.0009
        )
        if need:

            def residual(x):
                f = features(x)
                out = list(200 * (f["pelvis"] - target["pelvis"]))
                out.extend((230 * (f["feet"] - target["feet"])).ravel())
                out.extend((65 * (f["knees"] - targetknees)).ravel())
                out.extend((140 * (f["armpts"] - targetarms)).ravel())
                out.extend(6 * (x[3:6] - ref[3:6]))
                out.extend(8 * (f["gripper_dir"] - target["gripper_dir"]))
                for k in range(2):
                    out.extend(
                        15
                        * Rotation.from_matrix(target["feetrot"][k].T @ f["feetrot"][k]).as_rotvec()
                    )
                for k, z in feetplane.items():
                    out.append(11000 * (f["vertices"][fg[k]][:, 2].min() - z + 0.0004))
                for k in armplane:
                    out.append(15000 * (f["vertices"][p.armgeoms[k]][:, 2].min() - p.top + 0.0005))
                for k in kneeplane:
                    out.append(11000 * (region_gap(f, k) + 0.0004))
                for k in elbowplane:
                    out.append(11000 * (region_gap(f, k, True) + 0.0004))
                distances = np.array(
                    [mujoco.mj_geomDistance(m, d, g, h, 0.15, None) for g, h in pairs]
                )
                return np.r_[
                    out,
                    5000 * np.minimum(distances - 0.035, 0),
                    9000 * np.minimum(f["environment"] + 0.0001, 0),
                    7000 * np.minimum(f["selfdist"] + 0.0001, 0),
                    1.0 * (x[6:] - ref[6:]),
                    0.3 * (x - (previous if previous is not None else ref)),
                ]

            low = lo.copy()
            high = hi.copy()
            low[iy] = -1.309
            high[iy] = -1.3089
            seed = ref.copy() if previous is None else previous + (ref - refs[i - 1])
            sol = least_squares(
                residual,
                np.clip(seed, low + 1e-8, high - 1e-8),
                bounds=(low, high),
                max_nfev=85,
                diff_step=1e-4,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            x = sol.x
            nfev = sol.nfev
        else:
            x = ref
            nfev = 0
        f = features(x)
        ds = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.25, None) for g, h in pairs])
        row = dict(
            frame=i,
            source_time_s=float(t),
            phase=str(phase),
            nfev=nfev,
            min_arm_leg_distance_m=float(ds.min()),
            environment_penetration_m=float(max(0, -f["environment"].min())),
            self_penetration_m=float(max(0, -f["selfdist"].min())),
        )
        states.append(x.copy())
        records.append(row)
        previous = x.copy()
        if i % 20 == 0:
            print(json.dumps(row), flush=True)
    np.savez_compressed(
        (work_dir / "refined_states.npz"),
        states=np.array(states),
        phase=phases,
        fps=30.0,
        source_times=source_times,
    )
    Path((work_dir / "refinement.json")).write_text(
        json.dumps(
            dict(
                target_clearance_m=0.035,
                minimum_acceptance_clearance_m=0.025,
                original_duration_s=float(a["timestamps"][-1]),
                duration_s=float(times[-1]),
                records=records,
            ),
            indent=2,
        )
    )
    print("DONE", N, flush=True)


def check_clearance():
    import sys, json
    from pathlib import Path
    import numpy as np, mujoco

    f = np.load((work_dir / "refined_states.npz"))
    rows = []
    allgs = sorted(set(g for pair in pairs for g in pair))
    min_native = 1.0
    min_independent = 1.0
    min_certificate = 1.0
    max_bound_gap = 0.0
    disagreement = 0
    for i, x in enumerate(f["states"]):
        p.setq(x)
        vs = {g: vertices(g) for g in allgs}
        bounds = {g: (v.min(0), v.max(0)) for g, v in vs.items()}
        ds = np.array([mujoco.mj_geomDistance(m, d, g, h, 0.3, None) for g, h in pairs])
        j = int(np.argmin(ds))
        g, h = pairs[j]
        ind, cert = gjk(vs[g], vs[h], certificate=True)
        arg = j
        frame_cert = cert
        frame_bound_gap = ind - cert
        for k, (g, h) in enumerate(pairs):
            if k == j:
                continue
            lower = float(
                np.linalg.norm(
                    np.maximum(
                        np.maximum(bounds[g][0] - bounds[h][1], bounds[h][0] - bounds[g][1]), 0
                    )
                )
            )
            if lower >= ind:
                continue
            value, cert = gjk(vs[g], vs[h], certificate=True)
            frame_cert = min(frame_cert, cert)
            frame_bound_gap = max(frame_bound_gap, value - cert)
            if value < ind:
                ind = value
                arg = k
        native = float(ds.min())
        difference = max(0.0, 0.3 - ind) if native >= 0.3 - 1e-9 else abs(max(0, native) - ind)
        disagreement += difference > 0.0001
        min_certificate = min(min_certificate, frame_cert)
        max_bound_gap = max(max_bound_gap, frame_bound_gap)
        rows.append(
            dict(
                frame=i,
                phase=str(f["phase"][i]),
                native_min_distance_m=float(ds.min()),
                independent_min_distance_m=ind,
                separation_certificate_lower_bound_m=frame_cert,
                gjk_bound_gap_m=frame_bound_gap,
                native_query_saturated=bool(native >= 0.3 - 1e-9),
                native_pair=[m.geom(g).name for g in pairs[j]],
                independent_pair=[m.geom(g).name for g in pairs[arg]],
                difference_m=difference,
            )
        )
        min_native = min(min_native, float(ds.min()))
        min_independent = min(min_independent, ind)
        if i % 100 == 0:
            print(i, min_native, min_independent, flush=True)
    summary = dict(
        scope="All combinations of shoulder/elbow housing, forearm and wrist collision meshes versus both thigh, shin and foot collision meshes.",
        pair_count=len(pairs),
        frames=len(rows),
        minimum_native_distance_m=min_native,
        minimum_independent_convex_distance_m=min_independent,
        threshold_m=0.025,
        frames_below_threshold=sum(r["independent_min_distance_m"] < 0.025 for r in rows),
        cross_check_disagreements_over_0_1mm=int(disagreement),
        minimum_separation_certificate_m=min_certificate,
        max_gjk_upper_lower_bound_gap_m=max_bound_gap,
        primary_method="Convex-hull closest point with separating-plane certificate; capped/native anomalies retained as diagnostics",
        status="PASS" if min_certificate >= 0.025 and max_bound_gap < 0.0001 else "FAIL",
    )
    Path((work_dir / "clearance.json")).write_text(
        json.dumps(dict(summary=summary, frames=rows), indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


def smooth_wrist():
    from scipy.signal import savgol_filter

    f = np.load(work_dir / "refined_states.npz")
    data = {k: f[k].copy() for k in f.files}
    if bool(data.get("wrist_smoothing_applied", False)):
        return
    x = data["states"]
    j = 6 + p.names.index("r_wrist_x_joint")
    old = x[:, j].copy()
    bad = np.flatnonzero(abs(np.diff(old)) * 30 > 20.0)
    weight = np.zeros(len(old))
    filtered = savgol_filter(old, 11, 2)
    for center in bad:
        u = np.clip((10 - abs(np.arange(len(old)) - center)) / 5, 0, 1)
        weight = np.maximum(weight, u * u * (3 - 2 * u))
    x[:, j] = (1 - weight) * old + weight * filtered
    data["wrist_smoothing_applied"] = np.array(True)
    np.savez_compressed(work_dir / "refined_states.npz", **data)


if args.reuse_states:
    shutil.copyfile(args.reuse_states, work_dir / "refined_states.npz")
else:
    refine()
smooth_wrist()
check_clearance()
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("build_full_elbow_demo.py")),
        "--reuse-states",
        str(work_dir / "refined_states.npz"),
        "--output",
        str(args.output),
    ],
    check=True,
)
report = json.loads(args.output.with_suffix(".checks.json").read_text())
clearance = json.loads((work_dir / "clearance.json").read_text())
with np.load(args.output) as f:
    data = {k: f[k].copy() for k in f.files}
states = np.load(work_dir / "refined_states.npz")
data["source_reference_times_s"] = states["source_times"]
data["source_id"] = np.array(args.output.stem)
data["motion_scope"] = np.array(
    "Full geometric demo with explicit arm-leg clearance and shorter phase timing."
)
summary = report["summary"]
hold = next(row for row in report["phases"] if row["phase"] == "FOREARM_HOLD")
summary["arm_leg_clearance_status"] = clearance["summary"]["status"]
summary["minimum_arm_leg_clearance_m"] = clearance["summary"][
    "minimum_independent_convex_distance_m"
]
summary["reference_duration_s"] = float(a["timestamps"][-1])
summary["forearm_hold_gripper_angle_to_up_deg_range"] = hold["gripper_angle_to_up_deg_range"]
summary["limitations"] = [s for s in summary["limitations"] if "15.5 degrees" not in s]
summary["limitations"].append(
    "Positive arm-leg clearance is checked on the model collision meshes; forces and physical execution remain NOT RUN."
)
summary["refinement_status"] = (
    "PASS"
    if summary["demo_geometry_status"] == "PASS" and clearance["summary"]["status"] == "PASS"
    else "FAIL"
)
report["arm_leg_clearance"] = clearance
report["refiner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report["reference_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
data["validation_state"] = np.array(
    summary["refinement_status"] + "_DEMO_GEOMETRY_AND_ARM_LEG_CLEARANCE_NO_DYNAMICS"
)
data["contact_plan_metadata_json"] = np.array(
    json.dumps({k: v for k, v in summary.items() if k != "candidate_sha256"})
)
np.savez_compressed(args.output, **data)
summary["candidate_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_suffix(".checks.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "output": str(args.output),
            "duration_s": summary["duration_s"],
            "clearance": clearance["summary"],
            "geometry": summary["demo_geometry_status"],
            "joint_velocity_violations": summary["velocity_violations"],
        },
        indent=2,
    )
)
work.cleanup()
