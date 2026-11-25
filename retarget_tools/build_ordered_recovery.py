"""Recovery revision with ordered leg corridors and explicit weight shifts.
Kinematic candidate only. Read-only source motion/model; no hardware commands.
"""

import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from refine_recovery_demo import Recovery


class OrderedRecovery(Recovery):
    def solve_ordered(self, ref, t, s, seed=None, previous=None, max_nfev=100):
        p = self.p
        armidx = [
            6 + j
            for j, n in enumerate(self.names)
            if any(v in n for v in ["shoulder", "elbow", "wrist"])
        ]

        def residual(x):
            f = self.features(x)
            feet = self.local(f["feet"])
            knees = self.local(f["knees"])
            out = []
            out.extend(240 * (f["pelvis"] - t["pelvis"]))
            out.extend(4500 * (f["com"][:2] - t["com"][:2]))
            out.extend(650 * Rotation.from_matrix(t["rootrot"].T @ f["rootrot"]).as_rotvec())
            for k in range(2):
                out.extend(
                    (5000 if k in s.get("feet", {}) else 1300) * (f["feet"][k] - t["feet"][k])
                )
                out.extend(
                    (5000 if k in s.get("feet", {}) else 220)
                    * Rotation.from_matrix(t["feetrot"][k].T @ f["feetrot"][k]).as_rotvec()
                )
                out.extend(
                    (14000 if k in s.get("knees", []) else 180) * (f["knees"][k] - t["knees"][k])
                )
                out.extend(
                    (2000 if k in s.get("arms", []) else 25)
                    * (f["armpts"][k] - t["armpts"][k]).ravel()
                )
                sign = 1 if k == 0 else -1
                out.append(16000 * min(0, sign * feet[k, 1] - s.get("foot_corridor", 0.085)))
                out.append(
                    7000 * max(0, abs(knees[k, 1] - feet[k, 1]) - s.get("knee_offset", 0.10))
                )
            for k, z in s.get("feet", {}).items():
                out.append(100000 * (f["v"][self.fg[k]][:, 2].min() - z + 0.00012))
            for k in s.get("knees", []):
                out.append(100000 * (self.knee_gap(f, k) + 0.00012))
            for k in s.get("arms", []):
                out.append(100000 * (f["v"][p.armgeoms[k]][:, 2].min() - p.top + 0.00012))
            for k, z in s.get("air", {}).items():
                out.append(30000 * min(0, f["v"][self.fg[k]][:, 2].min() - p.top - z))
            for k in s.get("outside", []):
                out.append(20000 * max(0, self.local(f["v"][self.fg[k]])[:, 0].max() + 0.035))
            for k in s.get("inside", []):
                out.append(20000 * min(0, self.local(f["v"][self.fg[k]])[:, 0].min() - 0.06))
            out.append(300 * min(0, f["gripper_dir"][2] - s.get("gripper_up", 0.80)))
            gaps = np.array([self.sep(f, g, h) for g, h in self.pairs])
            out.extend(18000 * np.minimum(gaps - self.margins, 0))
            out.extend(80000 * np.minimum(f["en"] + 0.00006, 0))
            out.extend(40000 * np.minimum(f["ss"] + 0.00004, 0))
            out.extend(90 * (x[6:] - ref[6:]))
            out.extend(s.get("free_arm_reg", 160) * (x[armidx] - ref[armidx]))
            if previous is not None:
                out.extend(70 * (x - previous))
            return np.asarray(out)

        seed = ref if seed is None else seed
        low = self.lo.copy()
        high = self.hi.copy()
        if previous is not None:
            step = np.r_[np.full(3, 0.025), np.full(3, 0.08), np.full(29, 0.16)]
            low = np.maximum(low, previous - step)
            high = np.minimum(high, previous + step)
        if s.get("quiet_arms"):
            low[armidx] = np.maximum(low[armidx], ref[armidx] - 0.025)
            high[armidx] = np.minimum(high[armidx], ref[armidx] + 0.025)
        sol = least_squares(
            residual,
            np.clip(seed, low + 1e-8, high - 1e-8),
            bounds=(low, high),
            max_nfev=max_nfev,
            diff_step=1e-4,
            ftol=2e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        f = self.features(sol.x)
        feet = self.local(f["feet"])
        knees = self.local(f["knees"])
        met = {
            "nfev": sol.nfev,
            "env_m": float(max(0, -f["en"].min())),
            "self_m": float(max(0, -f["ss"].min())),
            "foot_span_m": float(feet[0, 1] - feet[1, 1]),
            "knee_foot_offset_m": abs(knees[:, 1] - feet[:, 1]).tolist(),
            "feet_error_m": np.linalg.norm(f["feet"] - t["feet"], axis=1).tolist(),
            "com_error_m": float(np.linalg.norm(f["com"][:2] - t["com"][:2])),
            "pelvis_local": self.local(f["pelvis"]).tolist(),
            "com_local": self.local(f["com"]).tolist(),
            "knees_local": knees.tolist(),
            "feet_local": feet.tolist(),
        }
        return sol.x, met


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--keys-only", action="store_true")
    ap.add_argument("--reuse-keys", type=Path)
    ap.add_argument("--resume-states", type=Path)
    ap.add_argument("--resume-after", default="RIGHT_KNEE_ADVANCE")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    r = OrderedRecovery("elf3_full_elbow_climb_demo_v6_candidate")
    p = r.p
    v = r.refs
    plans = []

    def add(name, idx, targets, support, seconds, trans=None):
        plans.append(
            dict(
                name=name,
                ref=v[idx].copy(),
                target=r.desired(v[idx], **targets),
                support=support,
                duration=seconds,
                trans=support if trans is None else trans,
            )
        )

    knees = {0: [0.15, 0.17, 0.055], 1: [0.075, -0.154, 0.055]}
    back = {0: ([-0.20, 0.17, 0.025], 80), 1: ([-0.274, -0.17, 0.025], 80)}
    add(
        "LEFT_KNEE_CLEAR",
        228,
        dict(
            feet={0: ([-0.26, 0.17, 0.06], 80), 1: back[1]},
            knees={0: [0.02, 0.17, 0.14], 1: [0.075, -0.154, 0.055]},
            com=[0.13, -0.025],
        ),
        {"arms": [0, 1], "knees": [1], "outside": [0, 1], "knee_offset": 0.12},
        0.9,
    )
    add(
        "BOTH_KNEES",
        255,
        dict(feet=back, knees=knees, pelvis=[0.06, 0, 0.27], com=[0.15, 0]),
        {"arms": [0, 1], "knees": [0, 1], "outside": [0, 1]},
        0.8,
        {"arms": [0, 1], "knees": [1], "outside": [0, 1]},
    )
    add(
        "RIGHT_KNEE_ADVANCE",
        255,
        dict(
            feet={0: back[0], 1: ([-0.20, -0.17, 0.025], 80)},
            knees={0: knees[0], 1: [0.15, -0.154, 0.055]},
            pelvis=[0.09, 0, 0.27],
            com=[0.18, 0],
        ),
        {"arms": [0, 1], "knees": [0, 1], "outside": [0, 1]},
        0.5,
        {"arms": [0, 1], "knees": [0], "outside": [0, 1]},
    )
    knees[1] = [0.15, -0.154, 0.055]
    back[1] = ([-0.20, -0.17, 0.025], 80)
    add(
        "KNEEL_UP",
        286,
        dict(feet=back, knees=knees, pelvis=[0.13, 0, 0.46], com=[0.065, 0], pitch=12),
        {"knees": [0, 1], "outside": [0, 1]},
        0.9,
    )
    add(
        "SHIFT_LEFT",
        286,
        dict(feet=back, knees=knees, pelvis=[0.13, 0.095, 0.45], com=[0.065, 0.17], pitch=8),
        {"knees": [0, 1], "outside": [0, 1], "knee_offset": 0.14},
        0.5,
    )
    add(
        "RIGHT_FOOT_LIFT",
        300,
        dict(
            feet={0: back[0], 1: ([-0.22, -0.15, 0.11], 65)},
            knees={0: knees[0], 1: [0.15, -0.15, 0.15]},
            pelvis=[0.13, 0.09, 0.47],
            com=[0.065, 0.17],
            pitch=8,
        ),
        {"knees": [0], "outside": [0, 1], "knee_offset": 0.14},
        0.4,
    )
    add(
        "RIGHT_FOOT_CLEAR",
        308,
        dict(
            feet={0: back[0], 1: ([0.27, -0.14, 0.065], 0)},
            knees={0: knees[0], 1: [0.43, -0.14, 0.30]},
            pelvis=[0.13, 0.09, 0.48],
            com=[0.065, 0.17],
            pitch=8,
        ),
        {"knees": [0], "outside": [0], "air": {1: 0.035}, "knee_offset": 0.14},
        0.65,
        {"knees": [0], "outside": [0], "air": {1: 0.025}, "knee_offset": 0.14},
    )
    add(
        "RIGHT_FOOT",
        331,
        dict(
            feet={0: back[0], 1: ([0.27, -0.14, 0], 0)},
            knees={0: knees[0], 1: [0.43, -0.14, 0.29]},
            pelvis=[0.13, 0.07, 0.48],
            com=[0.18, 0.10],
            pitch=12,
        ),
        {"knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1], "knee_offset": 0.14},
        0.4,
        {"knees": [0], "outside": [0], "inside": [1], "knee_offset": 0.14},
    )
    add(
        "SHIFT_RIGHT",
        331,
        dict(
            feet={0: back[0], 1: ([0.27, -0.14, 0], 0)},
            knees={0: knees[0]},
            pelvis=[0.20, -0.12, 0.48],
            com=[0.27, -0.13],
            pitch=15,
        ),
        {"knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1], "knee_offset": 0.14},
        0.65,
    )
    flat = {1: ([0.27, -0.14, 0], 0)}
    add(
        "LEFT_FOOT_LIFT",
        342,
        dict(
            feet={0: ([-0.20, 0.16, 0.12], 65), **flat},
            knees={0: [0.13, 0.15, 0.17]},
            pelvis=[0.23, -0.12, 0.52],
            com=[0.27, -0.13],
            pitch=15,
        ),
        {"feet": {1: p.top}, "outside": [0], "inside": [1], "knee_offset": 0.14},
        0.4,
    )
    add(
        "LEFT_FOOT_CLEAR",
        358,
        dict(
            feet={0: ([0.27, 0.16, 0.07], 0), **flat},
            knees={0: [0.43, 0.15, 0.30]},
            pelvis=[0.23, -0.12, 0.55],
            com=[0.27, -0.13],
            pitch=15,
        ),
        {"feet": {1: p.top}, "air": {0: 0.035}, "inside": [1], "knee_offset": 0.14},
        0.65,
        {"feet": {1: p.top}, "air": {0: 0.025}, "inside": [1], "knee_offset": 0.14},
    )
    add(
        "BOTH_FEET_LOAD",
        379,
        dict(
            feet={0: ([0.27, 0.16, 0], 0), **flat},
            pelvis=[0.23, -0.08, 0.55],
            com=[0.27, -0.08],
            pitch=15,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1], "knee_offset": 0.14},
        0.4,
        {"feet": {1: p.top}, "inside": [0, 1], "knee_offset": 0.14},
    )
    add(
        "CENTER",
        385,
        dict(
            feet={0: ([0.27, 0.16, 0], 0), **flat},
            pelvis=[0.23, 0.01, 0.56],
            com=[0.27, 0.01],
            pitch=12,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
        0.4,
    )
    add(
        "STAND",
        409,
        dict(
            feet={0: ([0.27, 0.16, 0], 0), **flat},
            pelvis=[0.26, 0.01, 0.825],
            com=[0.28, 0.01],
            pitch=0,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
        1.0,
    )
    # Use the same quiet arm-joint posture throughout kneeling and rising.
    armidx = [
        6 + j for j, n in enumerate(r.names) if any(w in n for w in ["shoulder", "elbow", "wrist"])
    ]
    for plan in plans[:3]:
        plan["target"]["armpts"] = r.features(v[201])["armpts"].copy()
    for plan in plans[3:]:
        plan["ref"][armidx] = v[286, armidx]
        plan["support"]["quiet_arms"] = True
        plan["trans"]["quiet_arms"] = True
    known = {}
    if args.reuse_keys:
        z = np.load(args.reuse_keys)
        known = dict(zip(z["names"].tolist(), z["states"]))
    keys = []
    logs = []
    for plan in plans:
        if plan["name"] in known:
            x = known[plan["name"]]
            met = {"reused": True}
        else:
            seed = plan["ref"].copy()
            for side in ["l", "r"]:
                for j in ["hip_z", "ankle_x"]:
                    seed[6 + r.names.index(side + "_" + j + "_joint")] = 0
                seed[6 + r.names.index(side + "_ankle_y_joint")] = -0.6
            x, met = r.solve_ordered(
                plan["ref"], plan["target"], plan["support"], seed=seed, max_nfev=160
            )
        keys.append(x)
        logs.append({"name": plan["name"], **met})
        print(json.dumps(logs[-1]), flush=True)
        np.savez_compressed(
            args.work / "keys.npz", states=keys, names=[v["name"] for v in plans[: len(keys)]]
        )
        (args.work / "key_checks.json").write_text(json.dumps(logs, indent=2))
    if args.keys_only:
        return
    states = list(v[:202])
    phases = list(r.a["phase"][:202])
    detail = [""] * len(states)
    source = list(r.a["timestamps"][:202])
    original = list(range(202))
    logs = []
    previous = v[201].copy()
    fa = r.features(previous)
    if args.resume_states:
        saved = np.load(args.resume_states)
        cut = int(np.flatnonzero(saved["phase"] == args.resume_after)[-1]) + 1
        states = list(saved["states"][:cut])
        phases = list(saved["phase"][:cut])
        detail = list(saved["recovery_stage"][:cut])
        source = list(saved["source_times"][:cut])
        original = list(saved["original_indices"][:cut])
        previous = states[-1]
        fa = r.features(previous)
        offset = [plan["name"] for plan in plans].index(args.resume_after) + 1
        plans = plans[offset:]
        keys = keys[offset:]
    for plan, end in zip(plans, keys):
        fb = r.features(end)
        count = round(plan["duration"] * 30)
        for j in range(1, count + 1):
            u = j / count
            u = u**3 * (10 - 15 * u + 6 * u * u)
            ref = (1 - u) * previous + u * end
            t = {
                k: (1 - u) * fa[k] + u * fb[k]
                for k in ["pelvis", "knees", "feet", "armpts", "com", "hand"]
            }
            for k in ["rootrot", "handrot"]:
                t[k] = (
                    fa[k]
                    @ Rotation.from_rotvec(
                        u * Rotation.from_matrix(fa[k].T @ fb[k]).as_rotvec()
                    ).as_matrix()
                )
            t["feetrot"] = np.array(
                [
                    fa["feetrot"][k]
                    @ Rotation.from_rotvec(
                        u * Rotation.from_matrix(fa["feetrot"][k].T @ fb["feetrot"][k]).as_rotvec()
                    ).as_matrix()
                    for k in range(2)
                ]
            )
            s = plan["support"] if j == count else dict(plan["trans"])
            for k in s.get("feet", {}):
                t["feetrot"][k] = p.R
            if plan["name"] == "LEFT_KNEE_CLEAR":
                s["foot_corridor"] = 0.045 + 0.04 * u
            if plan["name"] == "RIGHT_KNEE_ADVANCE":
                lift = 0.025 * np.sin(np.pi * j / count) ** 2
                t["knees"][1, 2] += lift
                t["feet"][1, 2] += lift
            x, met = r.solve_ordered(ref, t, s, seed=ref, previous=states[-1], max_nfev=65)
            states.append(x)
            phases.append(plan["name"])
            detail.append(plan["name"])
            source.append(201 / 30)
            original.append(-1)
            logs.append({"frame": len(states) - 1, "name": plan["name"], **met})
            if j % 10 == 0 or j == count:
                print(json.dumps(logs[-1]), flush=True)
        previous = states[-1]
        fa = r.features(previous)
        np.savez_compressed(
            args.work / "states.npz",
            states=states,
            phase=phases,
            fps=30.0,
            source_times=source,
            wrist_smoothing_applied=True,
            recovery_stage=detail,
            original_indices=original,
        )
    for _ in range(15):
        states.append(previous.copy())
        phases.append("FINISHED")
        detail.append("FINISHED")
        source.append(201 / 30)
        original.append(-1)
    np.savez_compressed(
        args.work / "states.npz",
        states=states,
        phase=phases,
        fps=30.0,
        source_times=source,
        wrist_smoothing_applied=True,
        recovery_stage=detail,
        original_indices=original,
    )
    (args.work / "solver.json").write_text(json.dumps(logs, indent=2))
    print(args.work / "states.npz", flush=True)


if __name__ == "__main__":
    main()
