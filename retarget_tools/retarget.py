"""Source adapters and the retained legacy wrist-support GMR baseline.

Uses existing GMR and ELF3 assets read-only. Does not train or run dynamics.
The CLI defaults to the forearm task. --baseline explicitly selects the
user-authorized first-stage wrist-support references; no training is started.
"""

from pathlib import Path
import sys, json, xml.etree.ElementTree as ET, tempfile, time, hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco, mink
from bvh import load_bvh

from project_paths import ROOT, SOURCE, OUTPUT, PROJECT, GMR

sys.path[:0] = [str(PROJECT), str(GMR)]
from legged_lab.motion.elf3_contract import JOINT_NAMES
from legged_lab.motion.official_hiking_retarget import UrdfKinematics
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


class GMRWithExplicitLimits(GeneralMotionRetargeting):
    """Local API compatibility: Mink 1.3 inserted safety_break before limits.

    Keep the existing GMR task setup/targets and iterative IK; pass limits by
    keyword so the installed Mink actually enforces them. No GMR source edits.
    """

    def update_targets(self, human_data, offset_to_ground=False, ground_clearance=0.1):
        super().update_targets(human_data, offset_to_ground, ground_clearance)
        offsets = getattr(self, "anatomical_offsets", None)
        if offsets is None:
            return
        frames = {
            name: self.scaled_human_data[entry[0]] for name, entry in self.ik_match_table1.items()
        }
        rotations = {
            name: Rotation.from_quat(pose[1], scalar_first=True).as_matrix()
            for name, pose in frames.items()
        }
        self.reference_pelvis_rotation = rotations["waist_z_link"].copy()
        # Preserve the torso and the nearest reachable pelvis orientation.
        # Otherwise a position-only compromise can twist the waist by ~180deg.
        if hasattr(self, "waist_limits"):
            relative = rotations["torso_link"].T @ rotations["waist_z_link"]
            angles = Rotation.from_matrix(relative).as_euler("YXZ")
            angles = np.clip(
                angles, self.waist_limits[:, 0] + 0.001, self.waist_limits[:, 1] - 0.001
            )
            rotations["waist_z_link"] = (
                rotations["torso_link"] @ Rotation.from_euler("YXZ", angles).as_matrix()
            )
            frames["waist_z_link"][1] = Rotation.from_matrix(rotations["waist_z_link"]).as_quat(
                scalar_first=True
            )
        positions = {"waist_z_link": frames["waist_z_link"][0].copy()}
        positions["torso_link"] = (
            positions["waist_z_link"]
            - rotations["torso_link"] @ offsets["torso_link", "waist_z_link"]
        )
        for side in ("l", "r"):
            hip, knee, ankle = [
                side + "_" + part + "_link" for part in ("hip_z", "knee_y", "ankle_x")
            ]
            shoulder, elbow, wrist = [
                side + "_" + part + "_link" for part in ("shoulder_z", "elbow_y", "wrist_z")
            ]
            for parent, child in [
                ("waist_z_link", hip),
                (hip, knee),
                (knee, ankle),
                ("torso_link", shoulder),
                (shoulder, elbow),
                (elbow, wrist),
            ]:
                positions[child] = positions[parent] + rotations[parent] @ offsets[parent, child]
        for target, point in positions.items():
            src = self.ik_match_table1[target][0]
            self.scaled_human_data[src][0] = point
            self.human_body_to_task1[src].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(frames[target][1]), point)
            )

    def retarget(self, human_data, offset_to_ground=False, ground_clearance=0.1):
        self.update_targets(human_data, offset_to_ground, ground_clearance)
        for enabled, tasks, error in (
            (self.use_ik_match_table1, self.tasks1, self.error1),
            (self.use_ik_match_table2, self.tasks2, self.error2),
        ):
            if not enabled:
                continue
            previous_error = error()
            for iteration in range(self.max_iter + 1):
                dt = self.configuration.model.opt.timestep
                base_constraints = getattr(self, "pose_constraints", [])
                contacts = getattr(self, "contact_constraints", [])
                constraints = base_constraints + contacts
                try:
                    velocity = mink.solve_ik(
                        self.configuration,
                        tasks,
                        dt,
                        self.solver,
                        damping=self.damping,
                        limits=self.ik_limits,
                        constraints=constraints,
                    )
                except mink.NoSolutionFound:
                    if contacts:
                        self.infeasible_contact_steps = (
                            getattr(self, "infeasible_contact_steps", 0) + 1
                        )
                    velocity = mink.solve_ik(
                        self.configuration,
                        tasks,
                        dt,
                        "proxqp",
                        damping=self.damping,
                        limits=self.ik_limits,
                        constraints=base_constraints,
                    )
                self.configuration.integrate_inplace(velocity, dt)
                next_error = error()
                constraint_error = max(
                    [
                        np.linalg.norm(c.compute_error(self.configuration))
                        for c in (constraints or [])
                    ]
                    or [0]
                )
                if previous_error - next_error <= 0.001 and constraint_error < 0.002:
                    break
                previous_error = next_error
        return self.configuration.data.qpos.copy()


class WorldPointTask(mink.Task):
    """A position-only world constraint; does not constrain hand orientation."""

    def __init__(self, model, name):
        super().__init__(cost=np.zeros(3), gain=0.35, lm_damping=1.0)
        self.model = model
        self.site = model.site(name).id
        self.target = np.zeros(3)

    def set_position_cost(self, cost):
        self.cost[:] = cost

    def set_target(self, target):
        self.target = np.asarray(
            target.translation() if hasattr(target, "translation") else target
        ).copy()

    def compute_error(self, configuration):
        return configuration.data.site_xpos[self.site] - self.target

    def compute_jacobian(self, configuration):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, configuration.data, jac, None, self.site)
        return jac


class OrientationOnlyTask(mink.Task):
    """Preserve reference torso orientation without fixing its translation."""

    def __init__(self, frame_task):
        super().__init__(cost=np.ones(3), gain=0.4, lm_damping=1.0)
        self.frame_task = frame_task

    def compute_error(self, configuration):
        return self.frame_task.compute_error(configuration)[3:]

    def compute_jacobian(self, configuration):
        return self.frame_task.compute_jacobian(configuration)[3:, :]


INDEX = json.loads((SOURCE / "index.json").read_text())
G1_JOINTS = [n.removeprefix("g1_29dof_").removesuffix("_q") for n in INDEX["G1_qpos_order"][7:]]


class G1Kinematics:
    """URDF FK with the published Drake state order, world Z-up/metres."""

    def __init__(self):
        r = ET.parse(SOURCE / "models/g1_29dof_spherehand_kinematics.urdf").getroot()
        self.joints = []
        for j in r.findall("joint"):
            origin = j.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            axis = j.find("axis")
            axis = (
                np.fromstring(axis.get("xyz"), sep=" ")
                if axis is not None
                else np.array([1.0, 0, 0])
            )
            self.joints.append(
                (
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    j.get("name"),
                    j.get("type"),
                    xyz,
                    Rotation.from_euler("xyz", rpy).as_matrix(),
                    axis,
                )
            )
        children = {x[1] for x in self.joints}
        self.root = next(l.get("name") for l in r.findall("link") if l.get("name") not in children)
        self.names = ["pelvis", "torso_link"] + [
            f"{s}_{n}"
            for s in ("left", "right")
            for n in (
                "thigh_frame",
                "knee_link",
                "ankle_roll_link",
                "upper_arm_frame",
                "elbow_link",
                "wrist_yaw_link",
            )
        ]
        self.rest = self.forward(np.r_[1.0, np.zeros(35)])
        self.rest_poses = self.poses

    def forward(self, q):
        poses = {self.root: (q[4:7], Rotation.from_quat(q[:4], scalar_first=True).as_matrix())}
        angles = dict(zip(G1_JOINTS, q[7:]))
        pending = self.joints.copy()
        while pending:
            progress = False
            for j in pending[:]:
                parent, child, name, typ, xyz, R0, axis = j
                if parent not in poses:
                    continue
                pp, RR = poses[parent]
                Rj = (
                    Rotation.from_rotvec(axis * angles.get(name, 0)).as_matrix()
                    if typ in ("revolute", "continuous")
                    else np.eye(3)
                )
                poses[child] = (pp + RR @ xyz, RR @ R0 @ Rj)
                pending.remove(j)
                progress = True
            if not progress:
                raise ValueError("Disconnected source URDF")
        # Anatomical origins paired with the complete three-axis joint frame.
        # The first pitch link alone omits hip/shoulder roll and yaw.
        for side in ("left", "right"):
            poses[f"{side}_thigh_frame"] = (
                poses[f"{side}_hip_pitch_link"][0],
                poses[f"{side}_hip_yaw_link"][1],
            )
            poses[f"{side}_upper_arm_frame"] = (
                poses[f"{side}_shoulder_pitch_link"][0],
                poses[f"{side}_shoulder_yaw_link"][1],
            )
        self.poses = poses
        return np.array([poses[n][0] for n in self.names]), np.array(
            [poses[n][1] for n in self.names]
        )


def pairs(human=False):
    # The existing local ELF3 BVH configs map torso_link to Spine2. With the
    # measured leg scale its height above Hips also matches ELF3's 0.2265m.
    p = [
        ("waist_z_link", "Hips" if human else "pelvis", 100, 20),
        ("torso_link", "Spine2" if human else "torso_link", 50, 30),
    ]
    for side, hs, gs in [("l", "Left", "left"), ("r", "Right", "right")]:
        for target, hbone, gbone, pw, rw in [
            ("hip_z", "UpLeg", "thigh_frame", 10, 5),
            ("knee_y", "Leg", "knee_link", 15, 5),
            ("ankle_x", "Foot", "ankle_roll_link", 100, 20),
            ("shoulder_z", "Arm", "upper_arm_frame", 10, 5),
            ("elbow_y", "ForeArm", "elbow_link", 15, 5),
            ("wrist_z", "Hand", "wrist_yaw_link", 20, 5),
        ]:
            p.append((f"{side}_{target}_link", hs + hbone if human else f"{gs}_{gbone}", pw, rw))
    return p


def source_motion(row, g1):
    if row["dataset"] == "OmniRetarget":
        with np.load(SOURCE / row["file"], allow_pickle=False) as a:
            q = a["qpos"].copy()
            fps = float(a["fps"])
        vals = []
        feet = []
        hands = []
        for x in q:
            vals.append(g1.forward(x))
            feet.append(
                [
                    g1.poses[s + "_ankle_roll_link"][0]
                    + g1.poses[s + "_ankle_roll_link"][1] @ np.array([0.035, 0, -0.035])
                    for s in ("left", "right")
                ]
            )
            hands.append([g1.poses[s + "_sphere_hand_tip_link"][0] for s in ("left", "right")])
        rest_feet = np.array(
            [
                g1.rest_poses[s + "_ankle_roll_link"][0]
                + g1.rest_poses[s + "_ankle_roll_link"][1] @ np.array([0.035, 0, -0.035])
                for s in ("left", "right")
            ]
        )
        return (
            g1.names,
            np.stack([x[0] for x in vals]),
            np.stack([x[1] for x in vals]),
            g1.rest,
            fps,
            False,
            {
                "feet": np.array(feet),
                "hands": np.array(hands),
                "rest_feet": rest_feet,
                "world_quat": [1, 0, 0, 0],
            },
        )
    a = load_bvh(SOURCE / row["file"], stride=4)
    names = [x[1] for x in pairs(True)]
    ids = [a["names"].index(n) for n in names]
    # HOLD rest pose: left=-X, forward=+Y, up=+Z. ELF3: left=+Y,
    # forward=+X, up=+Z. Rotate the whole source world, including terrain.
    C = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    feet = (
        np.stack(
            [
                np.mean(
                    a["positions"][
                        :,
                        [
                            a["names"].index(s + "SoleHeel_Estimated"),
                            a["names"].index(s + "SoleToe_Estimated"),
                        ],
                    ],
                    axis=1,
                )
                for s in ("Left", "Right")
            ],
            axis=1,
        )
        @ C.T
    )
    hands = (
        np.stack(
            [a["positions"][:, a["names"].index(s + "Hand")] for s in ("Left", "Right")], axis=1
        )
        @ C.T
    )
    rest_feet = (
        np.array(
            [
                np.mean(
                    a["rest_positions"][
                        [
                            a["names"].index(s + "SoleHeel_Estimated"),
                            a["names"].index(s + "SoleToe_Estimated"),
                        ]
                    ],
                    axis=0,
                )
                for s in ("Left", "Right")
            ]
        )
        @ C.T
    )
    return (
        names,
        a["positions"][:, ids] @ C.T,
        C @ a["rotations"][:, ids],
        (a["rest_positions"][ids] @ C.T, C @ a["rest_rotations"][ids]),
        a["fps"],
        True,
        {
            "feet": feet,
            "hands": hands,
            "rest_feet": rest_feet,
            "world_quat": Rotation.from_matrix(C).as_quat(scalar_first=True).tolist(),
        },
    )


def calibration(m, kin, names, rest, human):
    ps = pairs(human)
    sp, sR = rest
    d = mujoco.MjData(m)
    d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for side, sign in [("l", 1), ("r", -1)]:
        # G1 zero already has a forward forearm. Human T-pose has a straight
        # arm, requiring ELF3 elbow=90deg after shoulder abduction.
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_elbow_y_joint").id]] = np.pi / 2 if human else 0.0
        if human:
            d.qpos[m.jnt_qposadr[m.joint(f"{side}_shoulder_x_joint").id]] = sign * np.pi / 2
    mujoco.mj_forward(m, d)
    rp = d.xpos[m.body("waist_z_link").id]
    mp = {t: s for t, s, _, _ in ps}
    src_leg = sum(
        np.linalg.norm(sp[names.index(mp[b])] - sp[names.index(mp[a])])
        for a, b in [("l_hip_z_link", "l_knee_y_link"), ("l_knee_y_link", "l_ankle_x_link")]
    )
    kin.forward(np.zeros(29))
    dst_leg = sum(
        np.linalg.norm(kin.pose(b)[0] - kin.pose(a)[0])
        for a, b in [("l_hip_y_link", "l_knee_y_link"), ("l_knee_y_link", "l_ankle_x_link")]
    )
    scale = float(dst_leg / src_leg)
    config = {
        "robot_root_name": "torso_link",
        "human_root_name": ps[0][1],
        "ground_height": 0,
        "human_height_assumption": 1.7,
        "root_horizontal_scale": scale,
        "use_ik_match_table1": True,
        "use_ik_match_table2": False,
        "human_scale_table": {},
        "ik_match_table1": {},
        "ik_match_table2": {},
    }
    for target, src, pw, rw in ps:
        si = names.index(src)
        bid = m.body(target).id
        tr = d.xmat[bid].reshape(3, 3)
        offset = tr.T @ (d.xpos[bid] - rp - scale * (sp[si] - sp[0]))
        R = sR[si].T @ tr
        config["human_scale_table"][src] = scale
        config["ik_match_table1"][target] = [
            src,
            pw,
            rw,
            offset.tolist(),
            Rotation.from_matrix(R).as_quat(scalar_first=True).tolist(),
        ]
    return config, d.qpos.copy(), scale


def contact_anchors(points, fps, boxes, hand=False):
    """Geometric, low-speed intervals; anchors are frozen within each bout."""
    points = np.asarray(points)
    T = len(points)
    mask = np.zeros((T, 2), bool)
    anchors = points.copy()
    speed = np.linalg.norm(np.gradient(points, axis=0) * fps, axis=-1)
    for side in range(2):
        surface = np.zeros(T)
        nearest = points[:, side].copy()
        distance = np.full(T, np.inf)
        if not hand:
            distance = np.abs(points[:, side, 2])
            nearest[:, 2] = 0.0
        for box in boxes:
            center, R, size = box["center"], box["rotation"], box["half_size"]
            top = center[2] + R[2, 2] * size[2]
            if hand and top < 0.15:
                continue
            local = (points[:, side] - center) @ R
            p_local = np.clip(local, -size, size)
            p_local[:, 2] = size[2]
            p = p_local @ R.T + center
            dist = np.linalg.norm(points[:, side] - p, axis=1)
            if hand:
                use = dist < distance
            else:
                inside = np.all(np.abs(local[:, :2]) <= size[:2], axis=1)
                use = inside & (top >= surface)
                surface[use] = top
            if hand:
                p_local[:, :2] = np.clip(p_local[:, :2], -size[:2] + 0.025, size[:2] - 0.025)
                p = p_local @ R.T + center
            nearest[use] = p[use]
            distance[use] = dist[use]
        threshold = 0.06 if hand else np.where(surface > 0.15, 0.04, 0.10)
        raw = (distance < threshold) & (speed[:, side] < (0.10 if hand else 0.08))
        padded = np.pad(raw, (1, 1))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end in zip(starts, ends):
            if end - start < max(4, round(0.16 * fps)):
                continue
            mask[start:end, side] = True
            # A slow-moving point is not necessarily a sticking contact.
            # Freeze only genuinely stationary patches; otherwise preserve
            # the source path instead of pinning a whole walking phase.
            excursion = np.linalg.norm(nearest[start:end, :2] - nearest[start, :2], axis=1).max()
            anchors[start:end, side] = (
                np.median(nearest[start:end], axis=0) if excursion < 0.02 else nearest[start:end]
            )
    return mask, anchors


def convert(row, g1, folder):
    names, pos, rot, rest, fps, human, aux = source_motion(row, g1)
    asset = PROJECT / "legged_lab/assets/elf3_lite"
    urdf = asset / "urdf/elf3.urdf"
    tree = ET.parse(asset / "xml/elf3.xml").getroot()
    for inc in list(tree.findall("include")):
        tree.remove(inc)
    tree.find("compiler").set("meshdir", str(asset / "meshes"))
    for tag in ("keyframe", "equality", "contact"):
        for node in list(tree.findall(tag)):
            tree.remove(node)
    # Reuse the project's source STL geometry instead of legacy capsules.
    from legged_lab.hiking.mujoco_scene import _robot_mesh_geometry
    from view_motion import terrain

    _robot_mesh_geometry(tree)
    from gripper_geometry import apply as apply_gripper

    gripper_config = apply_gripper(tree)
    xml = folder / "model.xml"
    xml.write_text(ET.tostring(tree, encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(xml))
    kin = UrdfKinematics(urdf, JOINT_NAMES)
    config, seed, scale = calibration(m, kin, names, rest, human)
    boxes = terrain(
        {
            "source_id": np.array(row["id"]),
            "morphology_scale": scale,
            "source_to_elf3_world_rotation_wxyz": aux["world_quat"],
        }
    )
    world = tree.find("worldbody")
    ET.SubElement(
        world, "geom", name="retarget_floor", type="plane", size="0 0 .1", rgba=".22 .24 .26 1"
    )
    for i, box in enumerate(boxes):
        ET.SubElement(
            world,
            "geom",
            name=f"retarget_box_{i}",
            type="box",
            pos=" ".join(map(str, box["center"])),
            size=" ".join(map(str, box["half_size"])),
            quat=" ".join(
                map(str, Rotation.from_matrix(box["rotation"]).as_quat(scalar_first=True))
            ),
            rgba=".4 .6 .8 .8",
        )
    feet_mask, feet_anchor = contact_anchors(aux["feet"] * scale, fps, boxes)
    hands_mask, hands_anchor = contact_anchors(aux["hands"] * scale, fps, boxes, True)
    hand_control_mask = hands_mask.copy()
    hand_plan = hands_anchor.copy()
    # Anticipate support: approach above the edge before lowering the surface
    # point. Merely enforcing the endpoint can trap the hand on the box face.
    for k in range(2):
        padded = np.pad(hands_mask[:, k], (1, 1))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        for start in starts:
            begin = max(0, start - round(0.4 * fps))
            n = start - begin
            if not n:
                continue
            initial = aux["hands"][begin, k] * scale
            final = hands_anchor[start, k]
            for step, tpre in enumerate(range(begin, start)):
                if hands_mask[tpre, k]:
                    continue
                u = (step + 1) / (n + 1)
                blend = u * u * (3 - 2 * u)
                point = (1 - blend) * initial + blend * final
                point[2] += 0.14 * np.sin(np.pi * u)
                hand_plan[tpre, k] = point
                hand_control_mask[tpre, k] = True
    import trimesh

    for side, k in [("l", 0), ("r", 1)]:
        foot = tree.find(f'.//body[@name="{side}_ankle_x_link"]')
        ET.SubElement(foot, "site", name=f"{side}_sole_target", pos=".03 0 -.04", size=".004")
        wrist = tree.find(f'.//body[@name="{side}_wrist_z_link"]')
        link = ET.parse(urdf).getroot().find(f'link[@name="{side}_wrist_z_link"]')
        mesh = link.find("collision/geometry/mesh")
        vertices = np.asarray(trimesh.load(urdf.parent / mesh.get("filename")).vertices)
        src = dict((t, s) for t, s, _, _ in pairs(human))[f"{side}_wrist_z_link"]
        j = names.index(src)
        off = Rotation.from_quat(
            config["ik_match_table1"][f"{side}_wrist_z_link"][4], scalar_first=True
        ).as_matrix()
        Rs = rot[:, j] @ off
        selected = Rs[hands_mask[:, k]]
        Ravg = Rotation.from_matrix(selected).mean().as_matrix() if len(selected) else np.eye(3)
        height = (vertices @ Ravg.T)[:, 2]
        local = vertices[height < height.min() + 0.0005].mean(0)
        ET.SubElement(
            wrist, "site", name=f"{side}_hand_target", pos=" ".join(map(str, local)), size=".004"
        )
    xml.write_text(ET.tostring(tree, encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    d.qpos[:] = seed
    mujoco.mj_forward(m, d)
    target_rest = np.mean(
        [
            d.site_xpos[m.site(s + "_sole_target").id, 2] - d.xpos[m.body("waist_z_link").id, 2]
            for s in ("l", "r")
        ]
    )
    source_rest = float(np.mean(aux["rest_feet"][:, 2] - rest[0][0, 2])) * scale
    height_offset = source_rest - target_rest
    # Foot/body positions are soft priors; surface sites enforce geometric support.
    for target, entry in config["ik_match_table1"].items():
        if "ankle_x" in target:
            entry[1] = 0
        if "wrist_z" in target:
            entry[1] = 5
        if target in ("waist_z_link", "torso_link"):
            entry[1] *= 0.4
            entry[2] *= 2
    cfg = folder / "config.json"
    cfg.write_text(json.dumps(config))
    key = "elf3_box_data"
    ROBOT_XML_DICT[key] = xml
    IK_CONFIG_DICT["box_source"] = {key: cfg}
    engine = GMRWithExplicitLimits(
        "box_source", key, solver="daqp", verbose=False, use_velocity_limit=False
    )
    engine.configuration.update(seed)
    engine.max_iter = 35
    engine.set_ground_offset(-height_offset)
    kin.forward(np.zeros(29))
    bone_pairs = [("torso_link", "waist_z_link")]
    for side in ("l", "r"):
        hip, knee, ankle = [side + "_" + part + "_link" for part in ("hip_z", "knee_y", "ankle_x")]
        shoulder, elbow, wrist = [
            side + "_" + part + "_link" for part in ("shoulder_z", "elbow_y", "wrist_z")
        ]
        bone_pairs.extend(
            [
                ("waist_z_link", hip),
                (hip, knee),
                (knee, ankle),
                ("torso_link", shoulder),
                (shoulder, elbow),
                (elbow, wrist),
            ]
        )
    engine.anatomical_offsets = {
        (a, b): kin.pose(a)[1].T @ (kin.pose(b)[0] - kin.pose(a)[0]) for a, b in bone_pairs
    }
    engine.waist_limits = kin.limits[
        [list(JOINT_NAMES).index(n) for n in ("waist_y_joint", "waist_x_joint", "waist_z_joint")]
    ]
    engine.pose_constraints = [
        OrientationOnlyTask(engine.human_body_to_task1[config["ik_match_table1"][name][0]])
        for name in ("torso_link", "waist_z_link")
    ]
    qids = np.array([engine.model.jnt_qposadr[engine.model.joint(n).id] for n in JOINT_NAMES])
    vids = np.array([engine.model.jnt_dofadr[engine.model.joint(n).id] for n in JOINT_NAMES])
    jids = np.array([engine.model.joint(n).id for n in JOINT_NAMES])
    hard = kin.limits.copy()
    engine.model.jnt_range[jids] = hard
    previous = None
    qp = []
    dt = 1 / fps
    errors = []
    penetrations = []
    site_positions = []
    pelvis_errors = []
    orientation_errors = []
    self_penetrations = []
    support_tasks = []
    for side in ("l", "r"):
        ft = mink.FrameTask(
            frame_name=side + "_sole_target",
            frame_type="site",
            position_cost=180,
            orientation_cost=15,
            lm_damping=1,
        )
        ht = WorldPointTask(engine.model, side + "_hand_target")
        engine.tasks1.extend([ft, ht])
        support_tasks.append((ft, ht))
    robot_geoms = [
        i
        for i in range(engine.model.ngeom)
        if engine.model.geom_bodyid[i] and engine.model.geom_contype[i]
    ]
    env_geoms = [
        i
        for i in range(engine.model.ngeom)
        if engine.model.geom_bodyid[i] == 0 and engine.model.geom_contype[i]
    ]
    collision = mink.CollisionAvoidanceLimit(
        engine.model,
        [(robot_geoms, env_geoms)],
        gain=0.8,
        minimum_distance_from_collisions=-0.0005,
        collision_detection_distance=0.04,
    )
    # Mink's generic parent-body filter otherwise omits torso/world pairs.
    collision.geom_id_pairs = [(min(a, b), max(a, b)) for a in robot_geoms for b in env_geoms]
    collision.max_num_contacts = len(collision.geom_id_pairs)
    collision.broadphase = False
    self_collision = mink.CollisionAvoidanceLimit(
        engine.model,
        [(robot_geoms, robot_geoms)],
        gain=0.8,
        minimum_distance_from_collisions=-0.0005,
        collision_detection_distance=0.035,
    )
    excluded = {
        tuple(sorted((engine.model.body(e.get("body1")).id, engine.model.body(e.get("body2")).id)))
        for e in tree.findall("contact/exclude")
    }
    self_collision.geom_id_pairs = [
        (a, b)
        for a, b in self_collision.geom_id_pairs
        if tuple(sorted((int(engine.model.geom_bodyid[a]), int(engine.model.geom_bodyid[b]))))
        not in excluded
    ]
    # Coincident axes form one mechanical shoulder/hip/wrist assembly. Ignore
    # overlap within that assembly, as with the normal parent-link filtering.
    nominal = mujoco.MjData(engine.model)
    nominal.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    nominal.qpos[7:] = 0
    mujoco.mj_forward(engine.model, nominal)
    self_collision.geom_id_pairs = [
        (a, b)
        for a, b in self_collision.geom_id_pairs
        if np.linalg.norm(
            nominal.xpos[engine.model.geom_bodyid[a]] - nominal.xpos[engine.model.geom_bodyid[b]]
        )
        > 0.001
    ]

    def limb_group(body):
        name = engine.model.body(body).name
        if name.startswith(("l_", "r_")):
            part = name[2:]
            if part.startswith(("shoulder", "elbow", "wrist")):
                return name[:2] + "arm"
            if part.startswith(("hip", "knee", "ankle")):
                return name[:2] + "leg"
        return name

    # Empty intermediate joint bodies must not turn physically adjacent
    # segments into non-neighbors (e.g. shoulder_z -> empty elbow -> forearm).
    self_collision.geom_id_pairs = [
        (a, b)
        for a, b in self_collision.geom_id_pairs
        if limb_group(int(engine.model.geom_bodyid[a]))
        != limb_group(int(engine.model.geom_bodyid[b]))
    ]
    self_collision.max_num_contacts = len(self_collision.geom_id_pairs)
    self_collision.broadphase = False
    monitored_self_bodies = {
        tuple(sorted((int(engine.model.geom_bodyid[a]), int(engine.model.geom_bodyid[b]))))
        for a, b in self_collision.geom_id_pairs
    }
    check = mujoco.MjData(engine.model)
    line_search_rejections = 0
    warming = False

    def collision_depths(q):
        check.qpos[:] = q
        mujoco.mj_forward(engine.model, check)
        env = []
        self_values = []
        for c in check.contact:
            bodies = tuple(
                sorted(int(engine.model.geom_bodyid[int(g)]) for g in (c.geom1, c.geom2))
            )
            if 0 in bodies:
                env.append(max(0, -c.dist))
            elif bodies in monitored_self_bodies:
                self_values.append(max(0, -c.dist))
        return max(env or [0]), max(self_values or [0])

    def depth_at(q, environment_only=False):
        a, b = collision_depths(q)
        return a if environment_only else max(a, b)

    def safe_integrate(velocity, solver_dt):
        nonlocal line_search_rejections
        initial = engine.configuration.q.copy()
        old_env, old_self = collision_depths(initial)
        for backtrack in range(14):
            candidate = initial.copy()
            mujoco.mj_integratePos(engine.model, candidate, velocity, solver_dt * (0.5**backtrack))
            env_depth, self_depth = collision_depths(candidate)
            if (not warming and env_depth > max(0.001, old_env) + 1e-7) or self_depth > max(
                0.001, old_self
            ) + 1e-7:
                line_search_rejections += 1
                continue
            if previous is not None and np.linalg.norm(candidate[:3] - previous[:3]) > 0.06:
                continue
            if (
                previous is not None
                and (
                    Rotation.from_quat(previous[3:7], scalar_first=True).inv()
                    * Rotation.from_quat(candidate[3:7], scalar_first=True)
                ).magnitude()
                > 0.20
            ):
                continue
            engine.configuration.update(candidate)
            return
        engine.configuration.update(initial)

    engine.configuration.integrate_inplace = safe_integrate
    source_frames = list(range(len(pos)))
    dwell_counts = {}
    settled_counts = {}
    temporal_dwell = {}
    joint_speed = np.minimum(kin.velocity_limits, 9.0)
    for output_frame, t in enumerate(source_frames):
        h = {
            name: [pos[t, i].copy(), Rotation.from_matrix(rot[t, i]).as_quat(scalar_first=True)]
            for i, name in enumerate(names)
            if name in config["human_scale_table"]
        }
        if previous is not None:
            bounds = hard.copy()
            bounds[:, 0] = np.maximum(bounds[:, 0], previous[qids] - joint_speed * dt)
            bounds[:, 1] = np.minimum(bounds[:, 1], previous[qids] + joint_speed * dt)
            engine.model.jnt_range[jids] = bounds
        engine.update_targets(h)
        if previous is None:
            initial = seed.copy()
            initial[:3] = engine.scaled_human_data[config["ik_match_table1"]["torso_link"][0]][0]
            initial[3:7] = engine.scaled_human_data[config["ik_match_table1"]["torso_link"][0]][1]
            engine.configuration.update(initial)
        for k, side in enumerate(("l", "r")):
            ft, ht = support_tasks[k]
            foot_src = config["ik_match_table1"][side + "_ankle_x_link"][0]
            pfoot, qfoot = engine.scaled_human_data[foot_src]
            Rfoot = Rotation.from_quat(qfoot, scalar_first=True)
            fgoal = feet_anchor[t, k] if feet_mask[t, k] else aux["feet"][t, k] * scale
            if feet_mask[t, k]:
                Rfoot = Rotation.from_euler("z", Rfoot.as_euler("xyz")[2])
            ft.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(Rfoot.as_quat(scalar_first=True)), fgoal
                )
            )
            ht.set_position_cost(220 if hand_control_mask[t, k] else 0)
            ht.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), hand_plan[t, k])
            )
        if previous is None:
            # Fit the initial leg/arm posture before introducing environment
            # constraints; a neutral articulated pose with a tilted free root
            # is not an initial pose representative of the source motion.
            warming = True
            engine.configuration.integrate_inplace = safe_integrate
            engine.ik_limits = [mink.ConfigurationLimit(engine.model), self_collision]
            for _ in range(5):
                engine.retarget(h, offset_to_ground=False)
            initial = engine.configuration.q.copy()
            for _ in range(10):
                depth = depth_at(initial, environment_only=True)
                if depth < 0.0001:
                    break
                initial[2] += depth + 0.002
            engine.configuration.update(initial)
            warming = False
        engine.ik_limits = [mink.ConfigurationLimit(engine.model), collision, self_collision]
        engine.contact_constraints = [support_tasks[k][1] for k in range(2) if hands_mask[t, k]]
        for attempt in range(8 if previous is None else 3):
            q = engine.retarget(h, offset_to_ground=False)
            mujoco.mj_forward(engine.model, engine.configuration.data)
            depth = max(
                [
                    max(0, -c.dist)
                    for c in engine.configuration.data.contact
                    if any(engine.model.geom_bodyid[int(g)] == 0 for g in (c.geom1, c.geom2))
                ]
                or [0]
            )
            if depth < 0.003 and depth_at(q) < 0.003:
                break
        step_ratio = (
            float(np.max(np.abs(q[qids] - previous[qids]) / (joint_speed * dt)))
            if previous is not None
            else 0.0
        )
        engine.model.jnt_range[jids] = hard
        qp.append(q)
        previous = q.copy()
        penetrations.append(depth)
        sites = np.array(
            [
                engine.configuration.data.site_xpos[
                    engine.model.site(s + "_" + part + "_target").id
                ].copy()
                for part in ("sole", "hand")
                for s in ("l", "r")
            ]
        )
        site_positions.append(sites)
        wanted = np.concatenate((feet_anchor[t], hands_anchor[t]))
        active = np.r_[feet_mask[t], hands_mask[t]]
        errors.append(
            float(np.max(np.linalg.norm(sites[active] - wanted[active], axis=1)))
            if active.any()
            else 0.0
        )
        data = engine.configuration.data
        pelvis_id = engine.model.body("waist_z_link").id
        pelvis_key = config["human_root_name"]
        desired_p, desired_q = engine.scaled_human_data[pelvis_key]
        pelvis_errors.append(float(np.linalg.norm(data.xpos[pelvis_id] - desired_p)))
        orientation_errors.append(
            float(
                Rotation.from_matrix(
                    engine.reference_pelvis_rotation.T @ data.xmat[pelvis_id].reshape(3, 3)
                ).magnitude()
            )
        )
        self_penetrations.append(
            max(
                [
                    max(0, -c.dist)
                    for c in data.contact
                    if tuple(
                        sorted(int(engine.model.geom_bodyid[int(g)]) for g in (c.geom1, c.geom2))
                    )
                    in monitored_self_bodies
                ]
                or [0]
            )
        )
        # Give contact acquisition its own time instead of letting the source
        # body continue climbing while the larger target hand is still moving
        # around the edge. This changes timing explicitly and records mapping.
        new_hand = bool(t > 0 and np.any(hands_mask[t] & ~hands_mask[t - 1]))
        if new_hand:
            hand_error = max(
                [
                    np.linalg.norm(sites[k + 2] - hands_anchor[t, k])
                    for k in range(2)
                    if hands_mask[t, k]
                ]
                or [0]
            )
            settled_counts[t] = settled_counts.get(t, 0) + 1 if hand_error < 0.004 else 0
            if settled_counts[t] < 3 and dwell_counts.get(t, 0) < 60:
                source_frames.insert(output_frame + 1, t)
                dwell_counts[t] = dwell_counts.get(t, 0) + 1
        elif step_ratio > 0.95 and temporal_dwell.get(t, 0) < 12:
            source_frames.insert(output_frame + 1, t)
            temporal_dwell[t] = temporal_dwell.get(t, 0) + 1
        if output_frame % 100 == 0:
            print(
                row["id"],
                "source frame",
                t,
                "output frame",
                output_frame,
                "penetration",
                round(depth, 5),
                flush=True,
            )
    qp = np.array(qp)
    vel = np.zeros((len(qp), engine.model.nv))
    for t in range(len(qp) - 1):
        mujoco.mj_differentiatePos(engine.model, vel[t], dt, qp[t], qp[t + 1])
    vel[-1] = vel[-2]
    joints = qp[:, qids]
    jointvel = vel[:, vids]
    qout = np.concatenate((qp[:, :7], joints), axis=1)
    if not np.isfinite(qout).all() or not np.isfinite(jointvel).all():
        raise ValueError("Non-finite output")
    output = OUTPUT / f"{row['id']}.npz"
    source_frames = np.array(source_frames)
    intended_mask = np.concatenate((feet_mask, hands_mask), axis=1)[source_frames]
    targets = np.concatenate((feet_anchor, hands_anchor), axis=1)[source_frames]
    geometric_mask = intended_mask & (
        np.linalg.norm(np.array(site_positions) - targets, axis=-1) < 0.005
    )
    np.savez_compressed(
        output,
        fps=fps,
        timestamps=np.arange(len(qp)) / fps,
        qpos=qout,
        base_pos_w=qp[:, :3],
        base_quat_wxyz=qp[:, 3:7],
        joint_names=np.array(JOINT_NAMES),
        joint_pos=joints,
        joint_vel=jointvel,
        source_id=np.array(row["id"]),
        root_link=np.array("torso_link"),
        morphology_scale=np.array(scale),
        source_to_elf3_world_rotation_wxyz=np.array(aux["world_quat"]),
        retarget_version=np.array(4),
        support_mode=np.array("wrist_baseline"),
        gripper_envelope_config_json=np.array(json.dumps(gripper_config)),
        source_world_translation=np.array([0.0, 0.0, height_offset]),
        geometric_contact_mask=geometric_mask,
        intended_contact_mask=intended_mask,
        source_frame_indices=source_frames,
        contact_body_names=np.array(["l_sole", "r_sole", "l_wrist_surface", "r_wrist_surface"]),
        contact_frame_names=np.array(
            ["l_ankle_x_link", "r_ankle_x_link", "l_wrist_z_link", "r_wrist_z_link"]
        ),
        contact_local_positions=np.array(
            [
                engine.model.site(s + "_" + part + "_target").pos
                for part in ("sole", "hand")
                for s in ("l", "r")
            ]
        ),
        contact_targets_w=targets,
        contact_sites_w=np.array(site_positions),
        terrain_box_centers_w=np.array([b["center"] for b in boxes]),
        terrain_box_quat_wxyz=Rotation.from_matrix(
            np.array([b["rotation"] for b in boxes])
        ).as_quat(scalar_first=True),
        terrain_box_half_sizes=np.array([b["half_size"] for b in boxes]),
        environment_penetration_m=np.array(penetrations),
        intended_contact_error_m=np.array(errors),
        pelvis_tracking_error_m=np.array(pelvis_errors),
        pelvis_orientation_error_rad=np.array(orientation_errors),
        self_penetration_m=np.array(self_penetrations),
        validation_state=np.array("kinematic_contact_constrained_no_dynamics_validation"),
    )
    return {
        "id": row["id"],
        "file": output.name,
        "source_file": row["file"],
        "frames": len(qp),
        "fps": fps,
        "morphology_scale": scale,
        "status": "CONVERTED",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "retarget_version": 4,
        "height_offset_m": float(height_offset),
        "max_environment_penetration_m": float(max(penetrations)),
        "max_intended_contact_error_m": float(max(errors)),
        "collision_line_search_rejections": line_search_rejections,
        "infeasible_contact_ik_steps": getattr(engine, "infeasible_contact_steps", 0),
        "contact_acquisition_extra_frames": int(sum(dwell_counts.values())),
        "contact_acquisition_limit_hits": sum(v >= 60 for v in dwell_counts.values()),
        "joint_speed_cap_rad_s": 9.0,
        "temporal_extra_frames": int(sum(temporal_dwell.values())),
        "max_self_penetration_m": float(max(self_penetrations)),
        "pelvis_tracking_error_p95_m": float(np.percentile(pelvis_errors, 95)),
        "pelvis_orientation_error_p95_deg": float(
            np.degrees(np.percentile(orientation_errors, 95))
        ),
        "geometric_contact_frames": geometric_mask.sum(0).tolist(),
        "intended_contact_frames": intended_mask.sum(0).tolist(),
        "dual_wrist_geometry_contact_s": float(np.all(geometric_mask[:, 2:], axis=1).sum() / fps),
        "quality": "contact-constrained kinematic reference; actual gripper/elbow loading and dynamics not certified",
    }


def legacy_wrist_main(motion_ids):
    global OUTPUT
    from check_motion import inspect
    import shutil

    target = OUTPUT
    target.mkdir(exist_ok=True)
    chosen = set(motion_ids)
    rows = [r for r in INDEX["motions"] if not chosen or r["id"] in chosen]
    if chosen - set(r["id"] for r in rows):
        raise ValueError("Unknown source motion ID")
    previous = (
        json.loads((target / "index.json").read_text()) if (target / "index.json").exists() else {}
    )
    from gripper_geometry import specification

    active_gripper = specification()
    accepted = {
        r["id"]: r
        for r in previous.get("motions", [])
        if r.get("retarget_version") == 4
        and r.get("status") == "BASELINE_KINEMATIC_PASS"
        and r.get("checks", {}).get("validation_revision") == 5
        and r.get("checks", {}).get("left_gripper_envelope_check", {}).get("config_sha256")
        == active_gripper["config_sha256"]
        and (target / r["file"]).is_file()
        and hashlib.sha256((target / r["file"]).read_bytes()).hexdigest() == r.get("sha256")
    }
    rejected = {
        r["id"]: r
        for r in previous.get("rejected", [])
        if r.get("required_support") == "wrist_baseline"
    }
    g1 = G1Kinematics()
    started = time.monotonic()
    metadata = {
        **previous,
        "source_count": INDEX["count"],
        "retarget_version": 4,
        "validation_revision": 5,
        "robot": "ELF3",
        "joint_count": 29,
        "qpos_order": "base xyz + base quaternion wxyz + joint_names order",
        "root_link": "torso_link",
        "method": "local GMR with explicit Mink limits, anatomical bone frames, oriented terrain, non-neighbor collision constraints and contact retiming",
        "validation_scope": "Only the stated offline kinematic/geometry gates; not dynamics or full hardware/motion acceptance",
        "HOLD_downsampling_stride": 4,
        "reference_joint_speed_cap_rad_s": 9.0,
        "robot_urdf": str(PROJECT / "legged_lab/assets/elf3_lite/urdf/elf3.urdf"),
        "dynamics_validated": False,
        "elbow_contact_validated": False,
        "training_ready": False,
        "required_support": "wrist_baseline",
        "final_support_requirement": "dual_forearm",
        "forearm_qualified_count": 0,
        "reference_data_stage": "phase1_kinematic_baseline",
        "contact_labels": "inferred wrist/sole geometry; force labels unavailable",
    }
    if previous.get("required_support") != "wrist_baseline":
        metadata["forearm_rejections"] = previous.get("rejected", [])
    for key in ("model_check", "terrain_check", "source_integrity_check"):
        if key in previous:
            metadata[key] = previous[key]

    def save_index():
        pending = [
            {"id": r["id"], "status": "NOT RUN"}
            for r in INDEX["motions"]
            if r["id"] not in accepted and r["id"] not in rejected
        ]
        value = dict(
            metadata,
            count=len(accepted),
            rejected_count=len(rejected),
            motions=sorted(accepted.values(), key=lambda r: r["id"]),
            rejected=sorted(rejected.values(), key=lambda r: r["id"]),
            unprocessed=pending,
        )
        value["screening_outcome"] = "BASELINE_ONLY"
        value["retargeting_status"] = (
            "BASELINE_REFERENCES_AVAILABLE" if accepted else "BASELINE_IN_PROGRESS"
        )
        value["tool_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }
        temporary = target / ".index.json.tmp"
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        temporary.replace(target / "index.json")

    save_index()
    try:
        with tempfile.TemporaryDirectory(prefix="elf3_box_retarget_", dir="/tmp") as folder:
            OUTPUT = Path(folder) / "data"
            OUTPUT.mkdir()
            for i, row in enumerate(rows):
                checks = None
                try:
                    result = convert(row, g1, Path(folder))
                    checks = inspect(OUTPUT / result["file"], required_support="wrist_baseline")
                    result["checks"] = checks
                    if checks["status"] != "PASS":
                        raise ValueError("; ".join(checks["reasons"]))
                    result["status"] = "BASELINE_KINEMATIC_PASS"
                    result["support_mode"] = "wrist_baseline"
                    accepted[row["id"]] = result
                    rejected.pop(row["id"], None)
                    shutil.copy2(OUTPUT / result["file"], target / result["file"])
                except Exception as error:
                    accepted.pop(row["id"], None)
                    (target / (row["id"] + ".npz")).unlink(missing_ok=True)
                    rejected[row["id"]] = {
                        "id": row["id"],
                        "source_file": row["file"],
                        "status": "REJECTED_CURRENT_RETARGET",
                        "required_support": "wrist_baseline",
                        "reason": str(error),
                    }
                    if checks is not None:
                        rejected[row["id"]]["checks"] = checks
                save_index()
                print(
                    f"{i + 1}/{len(rows)} {row['id']}; accepted {len(accepted)}, rejected {len(rejected)}; elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        OUTPUT = target
        save_index()
        # Remove only this tool's known, unaccepted outputs. Preserve sources
        # and any unrelated files the user placed in the directory.
        for row in INDEX["motions"]:
            if row["id"] not in accepted:
                (target / (row["id"] + ".npz")).unlink(missing_ok=True)
    print("done", len(accepted), "accepted ELF3 motions", flush=True)


def main():
    if "--baseline" in sys.argv[1:]:
        legacy_wrist_main([s for s in sys.argv[1:] if s != "--baseline"])
        return
    chosen = set(sys.argv[1:])
    rows = [r for r in INDEX["motions"] if not chosen or r["id"] in chosen]
    if chosen - set(r["id"] for r in rows):
        raise SystemExit("Unknown source motion ID")
    # Merely replacing the legacy wrist site's name would preserve the wrong
    # support motion. A forearm source and contact IK are both prerequisites.
    if not any(r.get("forearm_geometry_screen", {}).get("geometric_candidate") for r in rows):
        raise SystemExit(
            "No selected source has a qualified dual-forearm placement. Legacy wrist retargeting was not run."
        )
    raise SystemExit(
        "Forearm contact IK is not implemented in the retained wrist baseline. No task-qualified data exported."
    )


if __name__ == "__main__":
    main()
