"""Resolve remaining frame-interpolation corner penetration with constrained IK.
Keeps the ordered recovery and contact targets. No physical execution claim.
"""

import argparse, json
from pathlib import Path
import numpy as np
from build_ordered_recovery import OrderedRecovery


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Refusing overwrite")
    z = np.load(args.states)
    a = {k: z[k].copy() for k in z.files}
    r = OrderedRecovery("elf3_full_elbow_climb_demo_v6_candidate")
    p = r.p
    logs = []
    for iteration in range(5):
        x = a["states"]
        bad = []
        for i in range(202, len(x) - 1):
            dep = 0.0
            for u in [0.25, 0.5, 0.75]:
                f = r.features((1 - u) * x[i] + u * x[i + 1])
                dep = max(dep, -f["en"].min(), -f["ss"].min())
            if dep > 0.00085:
                bad.append((i, float(dep)))
        print("iteration", iteration, "bad", bad, flush=True)
        if not bad:
            break
        inserts = {}
        for i, dep in bad:
            if str(a["phase"][i]) != "LEFT_KNEE_CLEAR":
                raise RuntimeError(
                    "Unexpected phase needs dedicated planning: " + str(a["phase"][i])
                )
            ref = (x[i] + x[i + 1]) / 2
            f = r.features(ref)
            t = {
                k: f[k].copy()
                for k in [
                    "pelvis",
                    "knees",
                    "feet",
                    "feetrot",
                    "armpts",
                    "com",
                    "rootrot",
                    "hand",
                    "handrot",
                ]
            }
            t["knees"][0, 2] += 0.022
            t["feet"][0, 2] += 0.015
            s = {"arms": [0, 1], "knees": [1], "outside": [0, 1], "knee_offset": 0.14}
            sol, met = r.solve_ordered(ref, t, s, seed=ref, max_nfev=130)
            inserts[i] = sol
            logs.append({"iteration": iteration, "after_frame": i, "before_depth_m": dep, **met})
            print(json.dumps(logs[-1]), flush=True)
        new = {
            k: [] for k in ["states", "phase", "source_times", "recovery_stage", "original_indices"]
        }
        for i in range(len(x)):
            for k in new:
                new[k].append(a[k][i])
            if i in inserts:
                new["states"].append(inserts[i])
                new["phase"].append(a["phase"][i])
                new["source_times"].append(a["source_times"][i])
                new["recovery_stage"].append(a["recovery_stage"][i])
                new["original_indices"].append(-1)
        for k, v in new.items():
            a[k] = np.array(v)
    else:
        raise RuntimeError("Interpolation remains unqualified; no final states exported")
    np.savez_compressed(args.output, **a)
    args.output.with_suffix(".json").write_text(
        json.dumps({"source": str(args.states), "repairs": logs}, indent=2)
    )


if __name__ == "__main__":
    main()
