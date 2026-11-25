"""Generate the scoped right-knee entry preview; no dynamics or full-climb claim.
The five seed poses were solved against the ELF3 model. Forearm and left-foot
contacts are preserved during frame-wise IK. The final left-foot liftoff gate
is intentionally reported independently from segment geometric acceptance.
"""

from project_paths import TOOLS_DIR
import argparse
from pathlib import Path

KEYPOSES = [
    [
        -0.719927919563145,
        0.004375636905507901,
        0.9808659449054736,
        -8.53002274271677e-05,
        0.9076878959145609,
        0.011454471676144462,
        -0.11179356832539476,
        0.0032771679312325976,
        -0.009170175595770914,
        -0.6557141568037608,
        0.06244896277294425,
        0.00778386353242473,
        -0.059559578741703645,
        -0.080429965545556,
        -0.0543637388619524,
        -0.6559001851436674,
        -0.07282246217346372,
        -0.007857323803292616,
        -0.05873445424295371,
        -0.08105033476223168,
        0.05642605572499568,
        -0.8674848021197075,
        0.012881242949586766,
        -0.10214045594419632,
        -0.021538162856795138,
        0.1835218673841702,
        -1.3089552910022175,
        -6.875546540935514e-16,
        -0.8366141265662059,
        -0.042453841012341086,
        0.09261916286398139,
        -0.09252761386819867,
        -0.4357932621009082,
        0.00026498173883389507,
        -0.0011871284187442905,
    ],
    [
        -0.7023280709461247,
        0.09365032998479618,
        0.9805916162876801,
        0.011285713114728797,
        0.9084443610293192,
        -0.017560294209115677,
        -0.10637943017131257,
        0.008182009482066531,
        -0.007949321689903066,
        -0.6306394621990494,
        -0.05883920942384129,
        -0.013067039112118978,
        -0.05371327222276665,
        -0.11685636882362752,
        -0.04674805639565214,
        -0.6766621566585868,
        -0.20340273988832266,
        -0.011154603355829424,
        -0.010761674481478295,
        -0.04890541495853899,
        0.05515066380040626,
        -0.8578068043113094,
        -0.002743861317414828,
        -0.13473225047255769,
        -0.0321407885873716,
        0.19048721569833105,
        -1.3089556976194678,
        -6.87419611384861e-16,
        -0.8308455945962262,
        -0.07742179523540844,
        0.055758244301036736,
        -0.09841412386220011,
        -0.4360945404199801,
        0.000903895605826338,
        -0.0010336952153071936,
    ],
    [
        -0.7756962509490369,
        0.05478201111170746,
        0.9873866811883444,
        0.04545713631235171,
        0.9923779253621258,
        -0.08237826235265942,
        -0.08213291047605954,
        0.13621495262300354,
        -0.4118963010214558,
        -0.8389141591135659,
        0.14792715891088515,
        0.4351567054096448,
        -0.015465999071229355,
        -0.07368941243965195,
        -0.002199340384149421,
        -2.677282053541459,
        -0.16433826620398428,
        -0.03983855375514028,
        2.4000000000163073,
        -0.08105033476220483,
        0.05642605572497861,
        -1.0443045390517234,
        -0.035752916635239336,
        -0.19223827862571663,
        0.07718270291686251,
        0.22152599163030617,
        -1.3089587292778029,
        4.510573692045665e-17,
        -1.0082949242592854,
        -0.09177615569602104,
        0.061355971313313364,
        -0.004674266802039969,
        -0.44232492418687425,
        -0.0730522907028418,
        -0.0011871284187442905,
    ],
    [
        -0.6873261843283305,
        0.028644503060370235,
        0.9890359196195022,
        -0.03565015220759441,
        1.0181085881065892,
        0.07887131692375164,
        -0.20238526553936861,
        0.0043358846089310175,
        -0.38162851584072344,
        -0.5579215510402754,
        0.28510278360545666,
        0.2647835295363236,
        -0.05957256089400036,
        -0.14538088130064097,
        -0.019463834500444276,
        -2.533859761982893,
        -0.040671020394939986,
        -0.05566548313544502,
        2.399999999999996,
        -0.08105033476221006,
        0.05642605572498047,
        -0.9087436498321612,
        -0.09684267266437317,
        -0.25226380050160296,
        -0.053021373728237696,
        0.3604948000076682,
        -1.3089600843587401,
        -1.0598038112231646e-15,
        -0.9361749748487135,
        -0.03039201930575726,
        -0.19967589146921969,
        -0.09617272456521432,
        -0.44997222871876535,
        -0.42348906745160764,
        -0.0011871284187442905,
    ],
    [
        -0.6699882788436612,
        0.016663829015877055,
        0.9870506334998634,
        -0.03389358879810577,
        1.0529288845620774,
        0.03635844710435125,
        -0.034048307604345886,
        -0.09751419362855092,
        -0.5488674126927279,
        -0.6886452725510073,
        0.596587713842905,
        0.32475496661099956,
        -0.029151756318473762,
        -0.13252051274592574,
        -0.12125053800456032,
        -2.3168365707475918,
        0.14171213713404376,
        -0.4462488698284939,
        2.422301835463425,
        -0.0845773758321696,
        0.050208111187103055,
        -0.8023597596221506,
        -0.050232618835891175,
        -0.08276203825469315,
        -0.22117058691913455,
        0.29488020081962163,
        -1.3089613584823576,
        -1.911136139372764e-15,
        -0.924212500253823,
        -0.03884378769119636,
        -0.06912768800934531,
        -0.1437226110422291,
        -0.4060691344761486,
        0.0022462763972703006,
        -0.0011871284187442905,
    ],
]


def create_context():
    import sys, itertools, json, tempfile
    from pathlib import Path
    import numpy as np, mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull

    sys.path.insert(0, str(TOOLS_DIR))
    from view_motion import load, model_file, terrain
    from check_motion import limb

    _, a = load("omniretarget_climb_28_z_scale_0.8")
    a["terrain_box_half_sizes"][0, 1] = 0.7
    work = tempfile.TemporaryDirectory()
    m = mujoco.MjModel.from_xml_path(str(model_file(work.name, a, True)))
    d = mujoco.MjData(m)
    names = a["joint_names"].tolist()
    ids = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
    ji = [m.joint(n).id for n in names]
    box = terrain(a)[0]
    R = box["rotation"] @ np.diag([-1.0, -1.0, 1.0])
    edge = -box["half_size"][0]
    top = box["center"][2] + box["half_size"][2]
    floor = 2 * a["terrain_box_half_sizes"][1, 2]
    feet = [m.body(s + "_ankle_x_link").id for s in ["l", "r"]]
    arms = [m.body(s + "_wrist_x_link").id for s in ["l", "r"]]
    knees = [m.body(s + "_knee_y_link").id for s in ["l", "r"]]
    hips = [m.body(s + "_hip_z_link").id for s in ["l", "r"]]
    sole = np.array([0.03, 0, -0.04])
    armpts = np.array([[-0.183, 0, -0.0329], [-0.094, 0, -0.0329]])
    robot = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]]
    hulls = {}
    for g in robot:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mi] : m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            hulls[g] = v[ConvexHull(v).vertices]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            hulls[g] = np.array(list(itertools.product([-1, 1], repeat=3))) * m.geom_size[g]
    selfpairs = [
        (g, h)
        for g, h in itertools.combinations(robot, 2)
        if limb(m.body(m.geom_bodyid[g]).name) != limb(m.body(m.geom_bodyid[h]).name)
    ]
    selfindex = {tuple(sorted(pair)): i for i, pair in enumerate(selfpairs)}
    armgeoms = [m.geom("hiking_" + s + "_wrist_x_link_0").id for s in ["l", "r"]]

    def world(x, y, z):
        return box["center"] + R @ np.array([x, y, z - box["center"][2]])

    def setq(x):
        d.qpos[:3] = world(*x[:3])
        rot = R @ Rotation.from_rotvec(x[3:6]).as_matrix()
        d.qpos[3:7] = Rotation.from_matrix(rot).as_quat(scalar_first=True)
        d.qpos[ids] = x[6:]
        mujoco.mj_forward(m, d)

    def geometry():
        depths = []
        points = {}
        for g, v in hulls.items():
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            points[g] = w
            l = (w - box["center"]) @ R
            delta = abs(l) - box["half_size"]
            sdf = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(1), 0)
            depths.append(min(float(sdf.min()), float(w[:, 2].min() - floor)))
        selfdist = np.zeros(len(selfpairs))
        contact_depth = {g: 0.0 for g in hulls}
        for c in d.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            if pair in selfindex:
                selfdist[selfindex[pair]] = min(selfdist[selfindex[pair]], float(c.dist))
            if 0 in [m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]]:
                for g in pair:
                    if g in contact_depth:
                        contact_depth[g] = min(contact_depth[g], float(c.dist))
        depths = np.minimum(depths, [contact_depth[g] for g in hulls])
        return np.array(depths), selfdist, points

    from types import SimpleNamespace

    return SimpleNamespace(**locals())


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New candidate NPZ path; existing files are never overwritten",
    )
    args = parser.parse_args()
    out = args.output
    if out.suffix != ".npz":
        parser.error("--output must end in .npz")
    if out.exists() or out.with_suffix(".checks.json").exists():
        parser.error("Output or checks file already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    import sys, json
    from pathlib import Path
    import numpy as np, mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull

    p = create_context()
    names = p.names
    m = p.m
    d = p.d
    fps = 30.0
    xs = [np.array(x) for x in KEYPOSES]
    counts = [45, 60, 45, 30]
    coms = []
    knees = []
    for x in xs:
        p.setq(x)
        coms.append(d.subtree_com[m.body("torso_link").id].copy())
        knees.append(d.xpos[p.knees[1]].copy())
    p.setq(xs[0])
    feetgoal = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ p.sole for b in p.feet]
    feetrot = [d.xmat[b].reshape(3, 3).copy() for b in p.feet]
    arm_local = []
    arm_goal = []
    for k, b in enumerate(p.arms):
        gg = p.armgeoms[k]
        pts = [
            c.pos.copy()
            for c in d.contact
            if gg in (c.geom1, c.geom2) and 0 in [m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]]
        ]
        if not pts:
            raise RuntimeError("Initial forearm contact missing")
        pt = np.mean(pts, axis=0)
        arm_goal.append(pt)
        arm_local.append(d.xmat[b].reshape(3, 3).T @ (pt - d.xpos[b]))
    refs = []
    for _ in range(15):
        refs.append((xs[0], coms[0], knees[0], "BOTH_FEET"))
    for seg, count in enumerate(counts):
        for raw in np.linspace(0, 1, count + 1)[1:]:
            u = raw * raw * (3 - 2 * raw)
            refs.append(
                (
                    (1 - u) * xs[seg] + u * xs[seg + 1],
                    (1 - u) * coms[seg] + u * coms[seg + 1],
                    (1 - u) * knees[seg] + u * knees[seg + 1],
                    ["UNLOAD_RIGHT", "LIFT_RIGHT_KNEE", "CLEAR_EDGE", "LAND_RIGHT_KNEE"][seg],
                )
            )
    for _ in range(30):
        refs.append((xs[-1], coms[-1], knees[-1], "RIGHT_KNEE_HOLD"))
    low = np.r_[[-1.4, -0.3, 0.7], [-0.6, 0.05, -0.7], m.jnt_range[p.ji, 0]]
    high = np.r_[[-0.4, 0.3, 1.25], [0.6, 1.6, 0.7], m.jnt_range[p.ji, 1]]
    iy = 6 + names.index("l_wrist_y_joint")
    low[iy] = -1.309
    high[iy] = -1.3089
    previous = None
    states = []
    records = []
    for f, (ref, comgoal, kneegoal, phase) in enumerate(refs):

        def residual(x):
            p.setq(x)
            en, ss, v = p.geometry()
            out = []
            for k, b in enumerate(p.feet):
                if k == 0 or phase in ["BOTH_FEET", "UNLOAD_RIGHT"]:
                    rr = d.xmat[b].reshape(3, 3)
                    out.extend(500 * (d.xpos[b] + rr @ p.sole - feetgoal[k]))
                    out.extend(35 * Rotation.from_matrix(feetrot[k].T @ rr).as_rotvec())
                    gg = m.geom("hiking_" + ["l", "r"][k] + "_ankle_x_link_0").id
                    out.append(4500 * (v[gg][:, 2].min() - p.floor + 0.0002))
            for k, b in enumerate(p.arms):
                point = d.xpos[b] + d.xmat[b].reshape(3, 3) @ arm_local[k]
                out.extend(300 * (point - arm_goal[k]))
                out.append(4500 * (v[p.armgeoms[k]][:, 2].min() - p.top + 0.0003))
            if phase not in ["BOTH_FEET", "UNLOAD_RIGHT"]:
                out.extend(160 * (d.xpos[p.knees[1]] - kneegoal))
            out.extend(100 * (d.subtree_com[m.body("torso_link").id][:2] - comgoal[:2]))
            return np.r_[
                out,
                2200 * np.minimum(en + 0.0001, 0),
                1500 * np.minimum(ss + 0.0001, 0),
                3 * (x[:6] - ref[:6]),
                0.5 * (x[6:] - ref[6:]),
            ]

        lo = low.copy()
        hi = high.copy()
        if previous is not None:
            step = np.r_[np.full(3, 0.02), np.full(3, 0.04), np.full(29, 0.10)]
            lo = np.maximum(lo, previous - step)
            hi = np.minimum(hi, previous + step)
        sol = least_squares(
            residual,
            np.clip(previous if previous is not None else ref, lo + 1e-9, hi - 1e-9),
            bounds=(lo, hi),
            max_nfev=70,
            diff_step=1e-4,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        previous = sol.x.copy()
        p.setq(sol.x)
        en, ss, v = p.geometry()
        states.append(sol.x.copy())
        row = {
            "frame": f,
            "phase": phase,
            "nfev": sol.nfev,
            "max_environment_penetration_m": float(max(0, -en.min())),
            "max_self_penetration_m": float(max(0, -ss.min())),
            "left_sole_error_m": float(
                np.linalg.norm(
                    d.xpos[p.feet[0]] + d.xmat[p.feet[0]].reshape(3, 3) @ p.sole - feetgoal[0]
                )
            ),
            "right_knee_error_m": float(np.linalg.norm(d.xpos[p.knees[1]] - kneegoal)),
        }
        records.append(row)
        if f % 15 == 0:
            print(json.dumps(row), flush=True)
    f = dict(
        states=np.array(states),
        phases=np.array([r[-1] for r in refs]),
        arm_local=np.array(arm_local),
        arm_targets=np.array(arm_goal),
        feet_targets=np.array(feetgoal),
    )
    import sys, json, hashlib
    from pathlib import Path
    import numpy as np, mujoco
    from scipy.spatial import ConvexHull

    sys.path.insert(0, str(TOOLS_DIR))
    from check_motion import longest_bout
    from gripper_geometry import specification

    states = f["states"]
    phases = f["phases"]
    m = p.m
    d = p.d
    N = len(states)
    fps = 30.0
    q = []
    roots = []
    quats = []
    rows = []
    coms = []
    footpoints = []
    kneepoints = []
    anchors = []
    masks = []
    region_positions = []
    expected = [
        "left_ground_foot",
        "right_ground_foot",
        "left_forearm",
        "right_forearm",
        "right_knee",
    ]

    def role(body, point):
        if body == "l_ankle_x_link":
            return "left_ground_foot"
        if body == "r_ankle_x_link":
            return "right_ground_foot"
        for k, side in enumerate(["l", "r"]):
            prefix = ["left", "right"][k]
            if body == side + "_wrist_x_link":
                return prefix + "_forearm"
            if body in [side + "_wrist_y_link", side + "_wrist_z_link"]:
                return prefix + "_wrist"
            if (
                body == side + "_shoulder_z_link"
                and np.linalg.norm(point - d.xpos[m.body(side + "_elbow_y_link").id]) < 0.10
            ):
                return prefix + "_elbow"
        if (
            body in ["r_hip_z_link", "r_knee_y_link"]
            and np.linalg.norm(point - d.xpos[p.knees[1]]) < 0.10
        ):
            return "right_knee"
        return "unexpected"

    def margin(points, com):
        points = np.unique(np.round(points, 9), axis=0)
        if len(points) < 3:
            return None
        try:
            h = ConvexHull(points)
            return float(
                np.min(
                    -(h.equations[:, :2] @ com + h.equations[:, 2])
                    / np.linalg.norm(h.equations[:, :2], axis=1)
                )
            )
        except Exception:
            return None

    for i, x in enumerate(states):
        p.setq(x)
        en, ss, v = p.geometry()
        q.append(d.qpos[p.ids].copy())
        roots.append(d.qpos[:3].copy())
        quats.append(d.qpos[3:7].copy())
        com = d.subtree_com[m.body("torso_link").id].copy()
        coms.append(com)
        feet = np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ p.sole for b in p.feet])
        footpoints.append(feet)
        kneepoints.append(d.xpos[p.knees].copy())
        anchor = np.array(
            [
                d.xpos[b] + d.xmat[b].reshape(3, 3) @ point
                for b, point in zip(p.arms, f["arm_local"])
            ]
        )
        anchors.append(anchor)
        region_points = {name: [] for name in expected}
        events = []
        found = set()
        support = []
        without_right = []
        top_support = []
        unplanned = []
        knee_top = False
        gdepth = 0.0
        for c in d.contact:
            gs = [int(c.geom1), int(c.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            if m.geom("left_gripper_envelope").id in gs:
                gdepth = max(gdepth, max(0.0, -float(c.dist)))
            if 0 not in bs:
                continue
            idx = 0 if bs[0] else 1
            body = m.body(bs[idx]).name
            env = m.geom(gs[1 - idx]).name
            r = role(body, c.pos)
            if r == "unexpected":
                unplanned.append(body)
                continue
            if c.dist > 0.001:
                continue
            if r.endswith("ground_foot") and env == "preview_box_0":
                continue
            if not r.endswith("ground_foot") and env != "preview_box_0":
                continue
            if abs(c.frame[2]) < 0.8:
                continue
            found.add(r)
            support.append(c.pos[:2])
            if r in region_points:
                region_points[r].append(c.pos.copy())
            events.append(
                {
                    "region": r,
                    "body": body,
                    "robot_geom": m.geom(gs[idx]).name,
                    "surface_geom": env,
                    "point_w": c.pos.tolist(),
                    "normal_w": c.frame[:3].tolist(),
                    "distance_m": float(c.dist),
                }
            )
            if r != "right_ground_foot":
                without_right.append(c.pos[:2])
            if env == "preview_box_0":
                top_support.append(c.pos[:2])
            if r == "right_knee":
                knee_top = True
        region_positions.append(
            [
                np.mean(region_points[name], axis=0) if region_points[name] else np.zeros(3)
                for name in expected
            ]
        )
        masks.append([s in found for s in expected])
        row = {
            "frame": i,
            "phase": str(phases[i]),
            "max_environment_penetration_m": float(max(0, -en.min())),
            "max_non_neighbor_self_penetration_m": float(max(0, -ss.min())),
            "gripper_penetration_m": gdepth,
            "contact_events": events,
            "contacts": sorted(found),
            "unexpected_contacts": sorted(set(unplanned)),
            "support_com_margin_m": margin(support, com[:2]),
            "support_com_margin_without_right_ground_foot_m": margin(without_right, com[:2]),
            "top_only_com_margin_m": margin(top_support, com[:2]),
            "left_sole_displacement_m": float(np.linalg.norm(feet[0] - f["feet_targets"][0])),
            "forearm_anchor_displacement_m": np.linalg.norm(
                anchor - f["arm_targets"], axis=1
            ).tolist(),
        }
        rows.append(row)
    q = np.array(q)
    roots = np.array(roots)
    quats = np.array(quats)
    masks = np.array(masks)
    vel = np.vstack((np.diff(q, axis=0) * fps, np.zeros((1, 29))))
    import xml.etree.ElementTree as ET
    from view_motion import ASSET

    urdf = ET.parse(ASSET / "urdf/elf3.urdf").getroot()
    limits = np.array(
        [
            [
                float(urdf.find(f'joint[@name="{n}"]/limit').get(k))
                for k in ["lower", "upper", "velocity"]
            ]
            for n in p.names
        ]
    )
    limit_bad = int(np.count_nonzero((q < limits[:, 0] - 1e-5) | (q > limits[:, 1] + 1e-5)))
    speed_bad = int(np.count_nonzero(abs(vel) > limits[:, 2] + 1e-5))
    summary = {
        "scope": "right-knee entry segment only; necessary geometric support checks, not dynamics",
        "frames": N,
        "fps": fps,
        "joint_limit_violations": limit_bad,
        "velocity_violations": speed_bad,
        "left_ground_contact_frames": int(masks[:, 0].sum()),
        "both_forearms_contact_frames": int(masks[:, 2:4].all(1).sum()),
        "right_knee_top_contact_frames": int(masks[:, 4].sum()),
        "right_knee_longest_contact_s": longest_bout(masks[:, 4], fps),
        "max_environment_penetration_m": max(r["max_environment_penetration_m"] for r in rows),
        "max_non_neighbor_self_penetration_m": max(
            r["max_non_neighbor_self_penetration_m"] for r in rows
        ),
        "max_gripper_penetration_m": max(r["gripper_penetration_m"] for r in rows),
        "max_left_sole_displacement_m": max(r["left_sole_displacement_m"] for r in rows),
        "max_forearm_anchor_displacement_m": max(
            max(r["forearm_anchor_displacement_m"]) for r in rows
        ),
        "minimum_support_com_margin_m": min(
            r["support_com_margin_m"] if r["support_com_margin_m"] is not None else -999
            for r in rows
        ),
        "final_top_only_com_margin_m": rows[-1]["top_only_com_margin_m"],
        "dynamics_status": "NOT RUN",
        "full_climb_status": "NOT RUN",
        "training_ready": False,
    }
    fail = []
    if limit_bad or speed_bad:
        fail.append("joint or speed limit")
    if (
        summary["max_environment_penetration_m"] > 0.003
        or summary["max_non_neighbor_self_penetration_m"] > 0.003
    ):
        fail.append("penetration >3mm")
    if summary["max_gripper_penetration_m"] > 0.001:
        fail.append("gripper penetration >1mm")
    if not masks[:, 0].all():
        fail.append("left ground contact lost")
    if not masks[:, 2:4].all():
        fail.append("forearm contact lost")
    if summary["minimum_support_com_margin_m"] < 0.01:
        fail.append("support projection margin <1cm")
    if summary["right_knee_longest_contact_s"] < 0.5:
        fail.append("right knee top contact too short")
    if (
        summary["max_left_sole_displacement_m"] > 0.005
        or summary["max_forearm_anchor_displacement_m"] > 0.01
    ):
        fail.append("support anchor drift")
    summary["segment_geometry_status"] = "FAIL" if fail else "PASS"
    summary["reasons"] = fail
    summary["left_leg_liftoff_gate"] = (
        "FAIL"
        if summary["final_top_only_com_margin_m"] is None
        or summary["final_top_only_com_margin_m"] < 0.01
        else "PASS"
    )
    print(json.dumps(summary, indent=2), flush=True)
    base = {
        key: p.a[key].copy()
        for key in [
            "joint_names",
            "terrain_box_centers_w",
            "terrain_box_quat_wxyz",
            "terrain_box_half_sizes",
        ]
    }
    base.update(
        fps=np.array(fps),
        timestamps=np.arange(N) / fps,
        qpos=np.c_[roots, quats, q],
        base_pos_w=roots,
        base_quat_wxyz=quats,
        joint_pos=q,
        joint_vel=vel,
        root_link=np.array("torso_link"),
        source_id=np.array("elf3_right_knee_entry_contact_plan"),
        motion_origin=np.array("contact_plan"),
        support_mode=np.array("left_foot_ground_forearms_then_right_knee"),
        phase=phases,
        com_w=np.array(coms),
        feet_sole_points_w=np.array(footpoints),
        knee_centers_w=np.array(kneepoints),
        forearm_material_points_w=np.array(anchors),
        contact_region_names=np.array(expected),
        geometric_contact_mask=masks,
        geometric_contact_positions_w=np.array(region_positions),
        validation_state=np.array(
            "SEGMENT_GEOMETRY_" + summary["segment_geometry_status"] + "_NO_DYNAMICS_NO_FULL_CLIMB"
        ),
        preview_camera_distance=np.array(2.7),
        preview_camera_lookat_w=p.world(p.edge - 0.12, 0, 0.65),
    )
    meta = {
        "method": "model-based multi-contact keyposes with per-frame contact-constrained IK",
        "scope": "first right knee entry only; does not claim a complete climb",
        "gripper": specification(),
        "mass_model": "original ELF3 model; gripper envelope has no added mass",
        "com_check": "projection on measured contact polygon; no forces, torque limits or dynamics certified",
        "terrain_width_m": 1.4,
        "terrain_height_m": p.top,
        "training_ready": False,
    }
    base["contact_plan_metadata_json"] = np.array(json.dumps(meta))
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **base)
    summary["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["solver_records"] = records
    summary["candidate_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    summary["file"] = out.name
    out.with_suffix(".checks.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=2)
    )
    print("Saved", out)


if __name__ == "__main__":
    main()
