"""Build a geometric recovery candidate with separated feet and right-arm support.

Read-only source model/motion. No training or hardware control. Whole-body
inverse dynamics is audited separately; geometry alone never accepts dynamics.
"""

from project_paths import OUTPUT
import argparse, json, itertools
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plan_knee_entry import create_context
from view_motion import load
from audit_foot_separation import gjk


class Recovery:
    def __init__(self, motion):
        self.p = p = create_context()
        self.m = m = p.m
        self.d = d = p.d
        self.path, self.a = load(motion)
        self.names = p.names
        self.waist = m.body("waist_z_link").id
        self.rh = m.body("r_wrist_z_link").id
        self.hg = m.geom("hiking_r_wrist_z_link_0").id
        self.lo = np.r_[[-1.8, -0.65, 0.6], [-1, -0.3, -1], m.jnt_range[p.ji, 0]]
        self.hi = np.r_[[0.7, 0.65, 2.1], [1, 1.8, 1], m.jnt_range[p.ji, 1]]
        self.iy = 6 + p.names.index("l_wrist_y_joint")
        self.lo[self.iy] = -1.309
        self.hi[self.iy] = -1.3089
        self.fg = [m.geom("hiking_" + s + "_ankle_x_link_0").id for s in ["l", "r"]]
        self.kg = [
            [m.geom("hiking_" + s + "_" + part + "_link_0").id for part in ["hip_z", "knee_y"]]
            for s in ["l", "r"]
        ]
        self.gripper = m.geom("left_gripper_envelope").id
        self.torso = m.geom("hiking_torso_link_0").id
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
        self.pairs = (
            list(itertools.product(arms, legs))
            + [(self.fg[0], self.fg[1])]
            + [(self.torso, g) for row in self.kg for g in row[1:]]
            + [(self.gripper, g) for g in legs]
        )
        self.margins = np.r_[
            np.full(len(arms) * len(legs), 0.035), 0.04, np.full(2, 0.02), np.full(len(legs), 0.02)
        ]
        refs = []
        for i in range(len(self.a["timestamps"])):
            xyz = p.R.T @ (self.a["base_pos_w"][i] - p.box["center"])
            xyz[2] = self.a["base_pos_w"][i, 2]
            rot = Rotation.from_matrix(
                p.R.T
                @ Rotation.from_quat(self.a["base_quat_wxyz"][i], scalar_first=True).as_matrix()
            ).as_rotvec()
            refs.append(np.r_[xyz, rot, self.a["joint_pos"][i]])
        self.refs = np.array(refs)

    def world(self, v):
        return self.p.world(self.p.edge + v[0], v[1], self.p.top + v[2])

    def local(self, v):
        q = (v - self.p.box["center"]) @ self.p.R
        q[..., 0] -= self.p.edge
        q[..., 2] -= self.p.box["half_size"][2]
        return q

    def features(self, x):
        p = self.p
        m = self.m
        d = self.d
        d.qpos[:3] = p.world(*x[:3])
        d.qpos[3:7] = Rotation.from_matrix(p.R @ Rotation.from_rotvec(x[3:6]).as_matrix()).as_quat(
            scalar_first=True
        )
        d.qpos[p.ids] = x[6:]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        en, ss, v = p.geometry()
        return dict(
            hand=d.xpos[self.rh] + d.xmat[self.rh].reshape(3, 3) @ np.array([0.05, 0, 0]),
            handrot=d.xmat[self.rh].reshape(3, 3).copy(),
            pelvis=d.xpos[self.waist].copy(),
            knees=d.xpos[p.knees].copy(),
            feet=np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ p.sole for b in p.feet]),
            feetrot=d.xmat[p.feet].reshape(2, 3, 3).copy(),
            armpts=np.array([p.armpts @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b in p.arms]),
            com=d.subtree_com[m.body("torso_link").id].copy(),
            gripper_dir=d.xmat[m.body("l_wrist_z_link").id].reshape(3, 3)[:, 0].copy(),
            rootrot=d.xmat[m.body("torso_link").id].reshape(3, 3).copy(),
            aabb={g: (pts.min(0), pts.max(0)) for g, pts in v.items()},
            v=v,
            en=en,
            ss=ss,
        )

    def knee_gap(self, f, k):
        v = np.vstack([f["v"][g] for g in self.kg[k]])
        near = np.linalg.norm(v - f["knees"][k], axis=1) < 0.105
        return v[near, 2].min() - self.p.top

    def sep(self, f, g, h):
        A = f["v"][g]
        B = f["v"][h]
        bound = np.linalg.norm(
            np.maximum(
                0, np.maximum(f["aabb"][g][0] - f["aabb"][h][1], f["aabb"][h][0] - f["aabb"][g][1])
            )
        )
        if bound > 0.045:
            return bound
        gap = gjk(A, B)
        return mujoco.mj_geomDistance(self.m, self.d, g, h, 0.08, None) if gap < 1e-7 else gap

    def solve(self, ref, target, support, seed=None, max_nfev=100, approach=False):
        p = self.p
        m = self.m
        d = self.d
        seed = ref if seed is None else seed

        def residual(x):
            f = self.features(x)
            out = list(180 * (f["pelvis"] - target["pelvis"]))
            out.extend((120 * (f["knees"] - target["knees"])).ravel())
            for k in range(2):
                out.extend(
                    (7000 if k in support.get("feet", {}) else 400)
                    * (f["feet"][k] - target["feet"][k])
                )
                out.extend(
                    (120 if approach else (1000 if k in support.get("feet", {}) else 55))
                    * Rotation.from_matrix(target["feetrot"][k].T @ f["feetrot"][k]).as_rotvec()
                )
                out.extend(
                    (
                        0
                        if k == 1 and support.get("wrist_task")
                        else (
                            1800
                            if k in support.get("arms", [])
                            else getattr(self, "free_arm_weight", 180)
                        )
                    )
                    * (f["armpts"][k] - target["armpts"][k]).ravel()
                )
            if support.get("wrist_task"):
                out.extend(
                    (4000 * (f["hand"] - target["hand"]))[:2]
                    if support.get("hand_contact")
                    else 4000 * (f["hand"] - target["hand"])
                )
                out.extend(
                    30 * Rotation.from_matrix(target["handrot"].T @ f["handrot"]).as_rotvec()
                )
            if support.get("hand_contact"):
                out.append(100000 * (f["v"][self.hg][:, 2].min() - p.top + 0.00012))
            if not approach:
                out.extend(1800 * (f["com"][:2] - target["com"][:2]))
                out.extend(
                    150
                    * np.maximum(0, getattr(self, "gripper_min_up", 0.91) - f["gripper_dir"][2])
                    * np.ones(1)
                )
            for k, z in support.get("feet", {}).items():
                out.append(100000 * (f["v"][self.fg[k]][:, 2].min() - z + 0.00012))
            for k in support.get("arms", []):
                out.append(100000 * (f["v"][p.armgeoms[k]][:, 2].min() - p.top + 0.00012))
            for k in support.get("knees", []):
                out.append(100000 * (self.knee_gap(f, k) + 0.00012))
                out.extend(800 * (f["knees"][k] - target["knees"][k]))
            for k, z in support.get("air", {}).items():
                out.append(14000 * min(0.0, f["v"][self.fg[k]][:, 2].min() - p.top - z))
            for k in support.get("outside", []):
                out.append(14000 * max(0.0, self.local(f["v"][self.fg[k]])[:, 0].max() + 0.04))
            for k in support.get("inside", []):
                out.append(14000 * min(0.0, self.local(f["v"][self.fg[k]])[:, 0].min() - 0.06))
            out.extend(
                getattr(self, "root_rotation_weight", 20)
                * Rotation.from_matrix(target["rootrot"].T @ f["rootrot"]).as_rotvec()
            )
            if approach:
                strength = target.get("yaw_strength", 1.0)
                for name in ["waist_z_joint", "l_hip_z_joint", "r_hip_z_joint"]:
                    out.append(70 * strength * x[6 + self.names.index(name)])
            gaps = np.array([self.sep(f, g, h) for g, h in self.pairs])
            out.extend(20000 * np.minimum(gaps - self.margins, 0))
            return np.r_[
                out,
                80000 * np.minimum(f["en"] + 0.00006, 0),
                40000 * np.minimum(f["ss"] + 0.00004, 0),
                getattr(self, "regularization", 2.0) * (x[6:] - ref[6:]),
                getattr(self, "continuity_weight", 0.0)
                * (x - getattr(self, "continuity_target", ref)),
            ]

        sol = least_squares(
            residual,
            np.clip(seed, self.lo + 1e-8, self.hi - 1e-8),
            bounds=(self.lo, self.hi),
            max_nfev=max_nfev,
            diff_step=1e-4,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        f = self.features(sol.x)
        gaps = [self.sep(f, g, h) for g, h in self.pairs]
        metrics = dict(
            nfev=sol.nfev,
            pelvis_error_m=float(np.linalg.norm(f["pelvis"] - target["pelvis"])),
            feet_error_m=np.linalg.norm(f["feet"] - target["feet"], axis=1).tolist(),
            arm_error_m=np.linalg.norm(
                (f["armpts"] - target["armpts"]).reshape(2, -1), axis=1
            ).tolist(),
            com_xy_error_m=float(np.linalg.norm(f["com"][:2] - target["com"][:2])),
            foot_gap_m=float(gaps[len(self.pairs) - 9]),
            minimum_arm_leg_gap_m=float(min(gaps[:48])),
            environment_penetration_m=float(max(0, -f["en"].min())),
            self_penetration_m=float(max(0, -f["ss"].min())),
        )
        return sol.x, metrics

    def desired(
        self, ref, feet=None, knees=None, pelvis=None, com=None, arms=None, pitch=None, hand=None
    ):
        f = self.features(ref)
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
        if feet:
            for k, (pos, angle) in feet.items():
                t["feet"][k] = self.world(pos)
                t["feetrot"][k] = (
                    self.p.R @ Rotation.from_euler("y", angle, degrees=True).as_matrix()
                )
        if knees:
            for k, pos in knees.items():
                t["knees"][k] = self.world(pos)
        if pelvis is not None:
            t["pelvis"] = self.world(pelvis)
        if com is not None:
            t["com"] = self.world([*com, 0.25])
        if arms:
            for k, pos in arms.items():
                t["armpts"][k] = (
                    np.asarray(pos).copy()
                    if np.asarray(pos).shape == (2, 3)
                    else (self.p.armpts - self.p.armpts.mean(0)) @ self.p.R.T + self.world(pos)
                )
        if hand is not None:
            t["hand"] = self.world(hand)
            t["handrot"] = np.column_stack(([0, 0, -1], self.p.R[:, 1], self.p.R[:, 0]))
        if pitch is not None:
            t["rootrot"] = self.p.R @ Rotation.from_euler("y", pitch, degrees=True).as_matrix()
        return t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default="elf3_full_elbow_climb_demo_v4_candidate")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--keys-only", action="store_true")
    parser.add_argument("--reuse-keys", type=Path)
    parser.add_argument("--polish-keys", action="store_true")
    parser.add_argument("--polish-after")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    r = Recovery(args.motion)
    p = r.p
    refs = r.refs
    a = r.a
    start = 200
    probe = np.load((OUTPUT / "validation/elf3_right_forearm_left_knee_probe_state.npz"))["state"]
    Rarm = r.features(probe)["armpts"][1].copy()
    Lfree = [0.45, 0.35, 0.30]
    knees = {0: [0.12, 0.17, 0.055], 1: [0.10, -0.17, 0.055]}
    back = {0: ([-0.20, 0.20, 0.02], 80), 1: ([-0.23, -0.20, 0.02], 80)}
    # name, source reference, targets, contacts at endpoint, transition seconds,
    # constraints that must hold during the incoming transition.
    plans = []

    def add(name, ref, t, s, d, trans):
        plans.append(
            dict(name=name, ref=ref, target=t, support=s, duration=d, transition_support=trans)
        )

    add(
        "LEFT_KNEE_CLEAR",
        refs[227],
        r.desired(
            refs[227],
            feet={0: ([-0.26, 0.20, 0.06], 80), 1: ([-0.23, -0.20, 0.02], 80)},
            knees={0: [0.02, 0.18, 0.14], 1: [0.075, -0.154, 0.055]},
            com=[0.12, -0.03],
        ),
        {"arms": [0, 1], "knees": [1], "outside": [0, 1]},
        0.8,
        {"arms": [0, 1], "knees": [1]},
    )
    add(
        "BOTH_KNEES",
        refs[254],
        r.desired(refs[254], feet=back, knees=knees, com=[0.18, 0]),
        {"arms": [0, 1], "knees": [0, 1], "outside": [0, 1]},
        0.65,
        {"arms": [0, 1], "knees": [1], "outside": [0, 1]},
    )
    add(
        "RIGHT_ARM_LOAD",
        probe,
        r.desired(
            probe,
            feet=back,
            knees=knees,
            pelvis=[0.12, -0.04, 0.29],
            com=[0.20, -0.10],
            arms={0: Lfree, 1: Rarm},
        ),
        {"arms": [1], "knees": [0, 1], "outside": [0, 1]},
        0.65,
        {"arms": [1], "knees": [0, 1], "outside": [0, 1]},
    )
    add(
        "RIGHT_FOOT_LIFT_OUTSIDE",
        probe,
        r.desired(
            probe,
            feet={0: back[0], 1: ([-0.23, -0.20, 0.20], 80)},
            knees={0: knees[0]},
            pelvis=[0.13, -0.06, 0.34],
            com=[0.21, -0.11],
            arms={0: Lfree, 1: Rarm},
        ),
        {"arms": [1], "knees": [0], "outside": [0, 1], "air": {1: 0.065}},
        0.5,
        {"arms": [1], "knees": [0], "outside": [0, 1]},
    )
    add(
        "RIGHT_FOOT_CLEAR",
        probe,
        r.desired(
            probe,
            feet={0: back[0], 1: ([0.26, -0.18, 0.14], 0)},
            knees={0: knees[0]},
            pelvis=[0.13, -0.06, 0.34],
            com=[0.21, -0.11],
            arms={0: Lfree, 1: Rarm},
        ),
        {"arms": [1], "knees": [0], "air": {1: 0.065}, "outside": [0]},
        0.6,
        {"arms": [1], "knees": [0], "outside": [0], "air": {1: 0.065}},
    )
    toe = {0: back[0], 1: ([0.26, -0.18, 0.03945109], 20)}
    flat = {0: back[0], 1: ([0.26, -0.18, 0.0], 0)}
    hand = [0.45, -0.38, 0.0]
    highfree = [0.45, 0.32, 0.45]
    add(
        "RIGHT_FOOT",
        probe,
        r.desired(
            probe,
            feet=toe,
            knees={0: knees[0]},
            pelvis=[0.13, -0.07, 0.36],
            com=[0.23, -0.08],
            arms={0: Lfree, 1: Rarm},
        ),
        {"arms": [1], "knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1]},
        0.5,
        {"arms": [1], "knees": [0], "outside": [0], "inside": [1]},
    )
    add(
        "RIGHT_HAND_HOVER",
        refs[329],
        r.desired(
            refs[329],
            feet=toe,
            knees={0: knees[0]},
            pelvis=[0.19, 0, 0.44],
            com=[0.25, 0],
            arms={0: highfree},
            pitch=60,
            hand=[0.45, -0.38, 0.08],
        ),
        {"knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1], "wrist_task": True},
        0.6,
        {"knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1], "wrist_task": True},
    )
    add(
        "RIGHT_HAND_PLANT",
        refs[329],
        r.desired(
            refs[329],
            feet=toe,
            knees={0: knees[0]},
            pelvis=[0.19, 0, 0.44],
            com=[0.25, 0],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {
            "knees": [0],
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
        0.35,
        {"knees": [0], "feet": {1: p.top}, "outside": [0], "inside": [1], "wrist_task": True},
    )
    add(
        "RIGHT_FOOT_FLAT",
        refs[329],
        r.desired(
            refs[329],
            feet=flat,
            knees={0: knees[0]},
            pelvis=[0.20, -0.04, 0.47],
            com=[0.26, -0.07],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {
            "knees": [0],
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
        0.35,
        {
            "knees": [0],
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
    )
    add(
        "SHIFT_RIGHT",
        refs[329],
        r.desired(
            refs[329],
            feet=flat,
            knees={0: knees[0]},
            pelvis=[0.22, -0.10, 0.48],
            com=[0.28, -0.16],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {
            "knees": [0],
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
        0.45,
        {
            "knees": [0],
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
    )
    add(
        "LEFT_FOOT_LIFT_OUTSIDE",
        refs[340],
        r.desired(
            refs[340],
            feet={0: ([-0.21, 0.20, 0.20], 80), 1: flat[1]},
            pelvis=[0.22, -0.10, 0.50],
            com=[0.28, -0.16],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "air": {0: 0.065},
            "wrist_task": True,
            "hand_contact": True,
        },
        0.6,
        {
            "feet": {1: p.top},
            "outside": [0],
            "inside": [1],
            "wrist_task": True,
            "hand_contact": True,
        },
    )
    add(
        "LEFT_FOOT_OVER_EDGE",
        refs[355],
        r.desired(
            refs[355],
            feet={0: ([0.26, 0.18, 0.15], 0), 1: flat[1]},
            pelvis=[0.23, -0.10, 0.50],
            com=[0.28, -0.16],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {
            "feet": {1: p.top},
            "inside": [0, 1],
            "air": {0: 0.065},
            "wrist_task": True,
            "hand_contact": True,
        },
        0.55,
        {
            "feet": {1: p.top},
            "inside": [1],
            "air": {0: 0.065},
            "wrist_task": True,
            "hand_contact": True,
        },
    )
    add(
        "LEFT_FOOT_LAND",
        refs[374],
        r.desired(
            refs[374],
            feet={0: ([0.26, 0.18, 0.0], 0), 1: flat[1]},
            pelvis=[0.23, -0.10, 0.50],
            com=[0.28, -0.12],
            arms={0: highfree},
            pitch=60,
            hand=hand,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1], "wrist_task": True, "hand_contact": True},
        0.4,
        {"feet": {1: p.top}, "inside": [0, 1], "wrist_task": True, "hand_contact": True},
    )
    add(
        "BOTH_FEET_LOAD",
        refs[374],
        r.desired(
            refs[374],
            feet={0: ([0.26, 0.18, 0.0], 0), 1: flat[1]},
            pelvis=[0.23, 0, 0.50],
            com=[0.28, 0],
            arms={0: highfree},
            pitch=55,
            hand=hand,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1], "wrist_task": True, "hand_contact": True},
        0.4,
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1], "wrist_task": True, "hand_contact": True},
    )
    add(
        "RELEASE_RIGHT_ARM",
        refs[385],
        r.desired(
            refs[385],
            feet={0: ([0.26, 0.18, 0.0], 0), 1: ([0.26, -0.18, 0.0], 0)},
            pelvis=[0.25, 0, 0.58],
            com=[0.28, 0],
            pitch=12,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
        0.7,
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
    )
    add(
        "STAND",
        refs[404],
        r.desired(
            refs[404],
            feet={0: ([0.26, 0.18, 0.0], 0), 1: ([0.26, -0.18, 0.0], 0)},
            pelvis=[0.26, 0, 0.825],
            com=[0.28, 0],
            pitch=0,
        ),
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
        1.0,
        {"feet": {0: p.top, 1: p.top}, "inside": [0, 1]},
    )
    keys = []
    logs = []
    previous = refs[start]
    known = {}
    if args.reuse_keys:
        z = np.load(args.reuse_keys)
        known = dict(zip(z["names"].tolist(), z["states"]))
    for plan in plans:
        if plan["name"] in known and not (
            args.polish_keys
            or (
                args.polish_after
                and [v["name"] for v in plans].index(plan["name"])
                >= [v["name"] for v in plans].index(args.polish_after)
            )
        ):
            x = known[plan["name"]]
            metrics = {"reused": True}
        else:
            x, metrics = r.solve(
                plan["ref"],
                plan["target"],
                plan["support"],
                seed=known.get(plan["name"], plan["ref"]),
                max_nfev=120,
            )
        keys.append(x)
        previous = x
        logs.append({"stage": plan["name"], **metrics})
        print(json.dumps(logs[-1]), flush=True)
        np.savez_compressed(
            args.work / "keys.npz", states=keys, names=[v["name"] for v in plans[: len(keys)]]
        )
        (args.work / "key_checks.json").write_text(json.dumps(logs, indent=2))
    if args.keys_only:
        return
    states = list(refs[: start + 1])
    phases = list(a["phase"][: start + 1])
    detail = [""] * len(states)
    source = list(a["timestamps"][: start + 1])
    original = list(range(start + 1))
    records = []
    # Neutralize unnecessary waist/hip yaw in the procedural approach; ease the
    # objective out before the existing forearm-contact segment.
    for i in range(44, 149):
        target = r.features(refs[i])
        target = {
            k: target[k].copy()
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
        target["yaw_strength"] = 1.0 if i <= 103 else max(0.0, (140 - i) / 37)
        if i <= 103:
            target["feetrot"] = np.array([p.R, p.R])
            target["rootrot"] = p.R
        contacts = {k: p.floor for k in range(2) if a["geometric_contact_mask"][i, k]}
        support = {"feet": contacts}
        x, met = r.solve(refs[i], target, support, seed=refs[i], max_nfev=65, approach=True)
        states[i] = x
        records.append({"frame": i, "stage": "STRAIGHT_APPROACH", **met})
        if i % 15 == 0:
            print(json.dumps(records[-1]), flush=True)
            np.savez_compressed(args.work / "approach_checkpoint.npz", states=states, frame=i)
    previous = refs[start]
    fa = r.features(previous)
    for plan, end in zip(plans, keys):
        fb = r.features(end)
        count = round(plan["duration"] * 30)
        support = plan["transition_support"]
        for j in range(1, count + 1):
            raw = j / count
            u = raw * raw * (3 - 2 * raw)
            ref = (1 - u) * previous + u * end
            t = {
                field: (1 - u) * fa[field] + u * fb[field]
                for field in ["pelvis", "knees", "feet", "armpts", "com", "hand"]
            }
            t["feetrot"] = np.array(
                [
                    fa["feetrot"][k]
                    @ Rotation.from_rotvec(
                        u * Rotation.from_matrix(fa["feetrot"][k].T @ fb["feetrot"][k]).as_rotvec()
                    ).as_matrix()
                    for k in range(2)
                ]
            )
            t["handrot"] = (
                fa["handrot"]
                @ Rotation.from_rotvec(
                    u * Rotation.from_matrix(fa["handrot"].T @ fb["handrot"]).as_rotvec()
                ).as_matrix()
            )
            t["rootrot"] = (
                fa["rootrot"]
                @ Rotation.from_rotvec(
                    u * Rotation.from_matrix(fa["rootrot"].T @ fb["rootrot"]).as_rotvec()
                ).as_matrix()
            )
            if j == count:
                x = end
                met = {"endpoint": True}
            else:
                x, met = r.solve(ref, t, support, seed=ref, max_nfev=65)
            states.append(x)
            phases.append(plan["name"])
            detail.append(plan["name"])
            source.append(start / 30)
            original.append(-1)
            records.append({"frame": len(states) - 1, "stage": plan["name"], **met})
            if j == count or j % 12 == 0:
                print(json.dumps(records[-1]), flush=True)
        previous = end
        fa = fb
        np.savez_compressed(
            args.work / "trajectory_checkpoint.npz",
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
        source.append(float(a["timestamps"][-1]))
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
    (args.work / "solver.json").write_text(
        json.dumps(
            {
                "records": records,
                "keys": logs,
                "source": str(r.path),
                "scope": "Geometric recovery candidate; dynamic execution not accepted.",
            },
            indent=2,
        )
    )
    print(args.work / "states.npz", flush=True)


if __name__ == "__main__":
    main()
