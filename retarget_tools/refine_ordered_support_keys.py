from project_paths import TOOLS_DIR
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(TOOLS_DIR))
from build_ordered_recovery import OrderedRecovery

r = OrderedRecovery("elf3_full_elbow_climb_demo_v6_candidate")
p = r.p
ap = argparse.ArgumentParser(
    description="Refine recovery keys using actual knee support-patch targets."
)
ap.add_argument("--work", type=Path, required=True)
args = ap.parse_args()
w = args.work
z = np.load(w / "keys.npz")
keys = z["states"].copy()
names = z["names"].tolist()
logs = []
knees = {0: [0.15, 0.17, 0.055], 1: [0.15, -0.154, 0.055]}
back = {0: ([-0.20, 0.17, 0.025], 80), 1: ([-0.20, -0.17, 0.025], 80)}
for name in ["KNEEL_UP", "SHIFT_LEFT", "RIGHT_FOOT_LIFT", "RIGHT_FOOT_CLEAR", "LEFT_FOOT_LIFT"]:
    idx = names.index(name)
    ref = keys[idx].copy()
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
    s = {"quiet_arms": True, "knee_offset": 0.14}
    if name in ["KNEEL_UP", "SHIFT_LEFT"]:
        t = r.desired(
            ref, feet=back, knees=knees, com=[0.065, 0 if name == "KNEEL_UP" else 0.15], pitch=5
        )
        s.update(knees=[0, 1], outside=[0, 1])
        seed = keys[names.index("RIGHT_FOOT_LIFT")].copy()
        seed[1] = 0
        seed[3] = seed[5] = 0
        for suffix in ["hip_y", "hip_x", "hip_z", "knee_y", "ankle_y", "ankle_x"]:
            l = 6 + r.names.index("l_" + suffix + "_joint")
            rr = 6 + r.names.index("r_" + suffix + "_joint")
            seed[rr] = seed[l] * (-1 if suffix.endswith(("x", "z")) else 1)
    else:
        seed = ref.copy()
        if name.startswith("RIGHT"):
            s.update(knees=[0], outside=[0])
            t["com"] = r.world([0.057, 0.19, 0.4])
        else:
            s.update(feet={1: p.top}, outside=[0], inside=[1])
            t["com"] = r.world([0.27, -0.14, 0.5])
            t["feetrot"][1] = p.R
    x, met = r.solve_ordered(ref, t, s, seed=seed, max_nfev=180)
    keys[idx] = x
    logs.append({"name": name, **met})
    print(json.dumps(logs[-1]), flush=True)
np.savez_compressed(w / "refined_keys.npz", states=keys, names=names)
(w / "refined_key_checks.json").write_text(json.dumps(logs, indent=2))
