"""Inspect existing ELF3 NPZ references with the local GMR viewer. No training."""

from pathlib import Path
import argparse, json, sys, tempfile, time, xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from bvh import load_bvh


def oriented_box(vertices, basis=None):
    vertices = np.asarray(vertices)
    center = vertices.mean(0)
    if basis is None:
        from scipy.spatial import ConvexHull

        xy = np.unique(np.round(vertices[:, :2], 10), axis=0)
        hull = xy[ConvexHull(xy).vertices]
        axis = hull[1] - hull[0]
        axis /= np.linalg.norm(axis)
        basis = np.array([[axis[0], -axis[1], 0], [axis[1], axis[0], 0], [0, 0, 1.0]])
    basis = np.asarray(basis).copy()
    if basis[2, 2] < 0:
        # Object capture frames can point down; the geometric top is +world Z.
        basis[:, 0] *= -1
        basis[:, 2] *= -1
    local = (vertices - center) @ basis
    low = local.min(0)
    high = local.max(0)
    center = center + basis @ ((low + high) / 2)
    return {"center": center, "rotation": basis, "half_size": (high - low) / 2}


from project_paths import ROOT, SOURCE, OUTPUT, ASSET, GMR, PROJECT

sys.path[:0] = [str(GMR), str(PROJECT)]


def load(name):
    p = Path(name)
    if not p.is_file():
        p = OUTPUT / (name if name.endswith(".npz") else name + ".npz")
    with np.load(p, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    n = len(a["timestamps"])
    fps = float(a["fps"])
    for k, shape in [
        ("joint_pos", (n, 29)),
        ("joint_vel", (n, 29)),
        ("base_pos_w", (n, 3)),
        ("base_quat_wxyz", (n, 4)),
        ("qpos", (n, 36)),
    ]:
        if a[k].shape != shape or not np.isfinite(a[k]).all():
            raise ValueError(f"Invalid {k}: {a[k].shape}")
    if n < 2 or fps <= 0 or not np.allclose(np.diff(a["timestamps"]), 1 / fps):
        raise ValueError("Invalid timestamps/fps")
    if not np.allclose(np.linalg.norm(a["base_quat_wxyz"], axis=1), 1, atol=1e-5):
        raise ValueError("Nonunit base quaternion")
    if len(set(a["joint_names"].tolist())) != 29:
        raise ValueError("Joint names must be unique")
    if not np.allclose(
        a["qpos"],
        np.concatenate((a["base_pos_w"], a["base_quat_wxyz"], a["joint_pos"]), axis=1),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("qpos and named root/joint fields disagree")
    return p, a


def terrain(a):
    if "terrain_box_centers_w" in a:
        return [
            {
                "center": c,
                "rotation": Rotation.from_quat(q, scalar_first=True).as_matrix(),
                "half_size": s,
            }
            for c, q, s in zip(
                a["terrain_box_centers_w"], a["terrain_box_quat_wxyz"], a["terrain_box_half_sizes"]
            )
        ]
    row = next(
        r
        for r in json.loads((SOURCE / "index.json").read_text())["motions"]
        if r["id"] == str(a["source_id"])
    )
    scale = float(a["morphology_scale"])
    boxes = []
    if row["dataset"] == "OmniRetarget":
        p = SOURCE / row["terrain"]
        for c in ET.parse(p).getroot().findall(".//collision"):
            mesh = c.find("geometry/mesh")
            origin = c.find("origin")
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
            if np.any(rpy):
                raise ValueError("Rotated terrain requires explicit transform")
            vertices = np.array(
                [
                    list(map(float, line.split()[1:4]))
                    for line in (p.parent / mesh.get("filename")).read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            vertices = (vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ") + xyz) * scale
            boxes.append(oriented_box(vertices))
    else:
        b = load_bvh(SOURCE / row["file"], stride=4)
        names = b["names"]
        world = names.index("Z_UP_WORLD_ORIGIN")
        for node in b["bones"]:
            if node["parent"] != world or node["name"] == "Hips":
                continue
            ids = [
                i
                for i, n in enumerate(names)
                if n.startswith(node["name"] + "_") and n.endswith("__end")
            ]
            if ids:
                quat = a.get("source_to_elf3_world_rotation_wxyz", [1, 0, 0, 0])
                C = Rotation.from_quat(quat, scalar_first=True).as_matrix()
                corners = (b["positions"][:, ids] @ C.T) * scale
                propid = names.index(node["name"])
                basis = C @ Rotation.from_matrix(b["rotations"][:, propid]).mean().as_matrix()
                boxes.append(oriented_box(np.median(corners, axis=0), basis))
    return boxes


def model_file(folder, a, show_terrain, include_gripper=True):
    root = ET.parse(ASSET / "xml/elf3.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET / "meshes"))
    for tag in ("include", "keyframe", "equality", "contact"):
        for node in list(root.findall(tag)):
            root.remove(node)
    from legged_lab.hiking.mujoco_scene import _robot_mesh_geometry

    _robot_mesh_geometry(root)
    if include_gripper:
        from gripper_geometry import apply

        apply(root)
    world = root.find("worldbody")
    ET.SubElement(
        world, "geom", name="preview_floor", type="plane", size="0 0 .1", rgba=".22 .24 .26 1"
    )
    if show_terrain:
        for i, box in enumerate(terrain(a)):
            ET.SubElement(
                world,
                "geom",
                name=f"preview_box_{i}",
                type="box",
                pos=" ".join(map(str, box["center"])),
                size=" ".join(map(str, box["half_size"])),
                quat=" ".join(
                    map(str, Rotation.from_matrix(box["rotation"]).as_quat(scalar_first=True))
                ),
                rgba=".4 .6 .8 .8",
            )
    p = Path(folder) / "preview.xml"
    p.write_text(ET.tostring(root, encoding="unicode"))
    return p


def preview_source(args):
    """Preview a raw source before committing to any robot retargeting."""
    import retarget

    row = next((r for r in retarget.INDEX["motions"] if r["id"] == args.source), None)
    if row is None:
        raise ValueError("Unknown local source motion ID")
    names, points, rot, rest, fps, human, aux = retarget.source_motion(row, retarget.G1Kinematics())
    print(f"Original source: {row['id']}, {len(points)} frames, {fps:.5f} Hz", flush=True)
    print(
        f"First torso inclination: {np.degrees(np.arccos(np.clip(rot[0, 1, 2, 2], -1, 1))):.3f} deg",
        flush=True,
    )
    if args.check_only:
        return
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    ET.SubElement(world, "geom", type="plane", size="0 0 .1", rgba=".22 .24 .26 1")
    if not args.no_terrain:
        for box in terrain(
            {
                "source_id": row["id"],
                "morphology_scale": 1.0,
                "source_to_elf3_world_rotation_wxyz": aux["world_quat"],
            }
        ):
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=" ".join(map(str, box["center"])),
                size=" ".join(map(str, box["half_size"])),
                quat=" ".join(
                    map(str, Rotation.from_matrix(box["rotation"]).as_quat(scalar_first=True))
                ),
                rgba=".4 .6 .8 .8",
            )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = {"paused": False, "frame": 0}

    def key(code):
        if code == 32:
            state["paused"] = not state["paused"]
        elif code in (262, 263):
            state["paused"] = True
            state["frame"] = (state["frame"] + (1 if code == 262 else -1)) % len(points)

    from mujoco import viewer as mjviewer

    print(
        "Green: original skeleton; orange: elbow-to-wrist segments. Space: pause; left/right: one frame.",
        flush=True,
    )
    print(
        "Source props are shown at their median captured pose; numeric screening uses per-frame prop poses.",
        flush=True,
    )
    with mjviewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = np.median(points[:, 0], axis=0)
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -20
        while viewer.is_running():
            begin = time.monotonic()
            frame = state["frame"]
            scene = viewer.user_scn
            scene.ngeom = 0
            for first, last in [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (1, 5),
                (5, 6),
                (6, 7),
                (0, 8),
                (8, 9),
                (9, 10),
                (1, 11),
                (11, 12),
                (12, 13),
            ]:
                forearm = (first, last) in ((6, 7), (12, 13))
                color = (
                    np.array([1.0, 0.45, 0.05, 1.0]) if forearm else np.array([0.1, 0.95, 0.2, 1.0])
                )
                width = 0.018 if forearm else 0.009
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.ones(3) * width,
                    np.zeros(3),
                    np.eye(3).ravel(),
                    color,
                )
                mujoco.mjv_connector(
                    g,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    width,
                    points[frame, first],
                    points[frame, last],
                )
                scene.ngeom += 1
            viewer.sync()
            if not state["paused"]:
                state["frame"] = (frame + 1) % len(points)
            time.sleep(max(0.0, 1 / (fps * args.speed) - (time.monotonic() - begin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--motion")
    parser.add_argument(
        "--source", help="Preview a local raw source ID, e.g. hold_I13, without ELF3 retargeting"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check fields and model joint mapping without opening a GUI",
    )
    parser.add_argument("--no-terrain", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--no-gripper", action="store_true", help="Compare against the original bare-wrist model"
    )
    parser.add_argument(
        "--show-source", action="store_true", help="Overlay the scaled source skeleton in green"
    )
    parser.add_argument(
        "--once", action="store_true", help="Play one cycle, then hold the final frame"
    )
    parser.add_argument(
        "--view",
        choices=["default", "rear", "side", "rear-oblique"],
        default="default",
        help="Initial fixed camera relative to terrain orientation",
    )
    args = parser.parse_args()
    if args.list:
        motions = json.loads((OUTPUT / "index.json").read_text())["motions"]
        for r in motions:
            print(
                f"{r['id']:40s} {r['frames']:5d} frames  {r['fps']:.5f} Hz  {r.get('status', 'UNKNOWN')}"
            )
        if not motions:
            print("No ELF3 motions qualified for the required forearm support task.")
        return
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.source:
        preview_source(args)
        return
    if args.motion is None:
        motions = json.loads((OUTPUT / "index.json").read_text())["motions"]
        if not motions:
            parser.error(
                "No qualified ELF3 forearm motion. Use --source hold_I13 to inspect an original BVH."
            )
        args.motion = motions[0]["id"]
    p, a = load(args.motion)
    contact_plan = str(a.get("motion_origin", "")) == "contact_plan"
    if contact_plan and args.show_source:
        parser.error(
            "This model-generated contact plan has no captured source skeleton to overlay."
        )
    with tempfile.TemporaryDirectory(prefix="elf3_view_", dir="/tmp") as folder:
        xml = model_file(folder, a, not args.no_terrain, not args.no_gripper)
        m = mujoco.MjModel.from_xml_path(str(xml))
        d = mujoco.MjData(m)
        qids = np.array([m.jnt_qposadr[m.joint(str(n)).id] for n in a["joint_names"]])
        for t in (0, len(a["timestamps"]) - 1):
            d.qpos[:3] = a["base_pos_w"][t]
            d.qpos[3:7] = a["base_quat_wxyz"][t]
            d.qpos[qids] = a["joint_pos"][t]
            mujoco.mj_forward(m, d)
            if not np.isfinite(d.xpos).all():
                raise ValueError("Invalid FK")
        print(
            f"{p.name}: {len(a['timestamps'])} frames, {float(a['fps']):.5f} Hz, 29 joints; fields and joint names readable",
            flush=True,
        )
        if contact_plan:
            print(
                "Origin: model-generated contact plan.",
                str(a.get("motion_scope", "First right knee only; not a complete climb.")),
                flush=True,
            )
        else:
            print("Retarget version:", int(a.get("retarget_version", 1)), flush=True)
        print("Support reference:", str(a.get("support_mode", "legacy/unspecified")), flush=True)
        if not args.no_gripper:
            from gripper_geometry import specification

            cfg = specification()
            print("Orange: left-gripper envelope; dimensions mm:", cfg["dimensions_mm"], flush=True)
            print("Envelope mount assumption:", cfg["mount_assumption"], flush=True)
        if not contact_plan and int(a.get("retarget_version", 1)) < 4:
            print(
                "WARNING: legacy calibration; this file has not received the current correction.",
                flush=True,
            )
        print(
            "Kinematic reference preview only; no contact/dynamics qualification. HOLD props use static median bounds.",
            flush=True,
        )
        if args.check_only:
            return
        from general_motion_retargeting import RobotMotionViewer
        from general_motion_retargeting.params import ROBOT_XML_DICT

        ROBOT_XML_DICT["elf3"] = xml
        state = {"paused": False, "frame": 0, "source": args.show_source}

        def key(code):
            if code == 32:
                state["paused"] = not state["paused"]
            elif code in (262, 263):
                state["paused"] = True
                state["frame"] = (state["frame"] + (1 if code == 262 else -1)) % len(
                    a["timestamps"]
                )
            elif code in (83, 115):
                state["source"] = not state["source"]

        viewer = RobotMotionViewer(
            "elf3",
            motion_fps=float(a["fps"]) * args.speed,
            camera_follow=False,
            keyboard_callback=key,
        )
        viewer.viewer.cam.lookat[:] = a.get(
            "preview_camera_lookat_w", np.median(a["base_pos_w"], axis=0)
        )
        viewer.viewer.cam.distance = float(a.get("preview_camera_distance", 4.0))
        viewer.viewer.cam.elevation = -20
        viewer.viewer.cam.azimuth = 130
        if args.view != "default":
            basis = Rotation.from_quat(
                a["terrain_box_quat_wxyz"][0]
                if len(a.get("terrain_box_quat_wxyz", []))
                else a["base_quat_wxyz"][0],
                scalar_first=True,
            ).as_matrix()
            heading = float(np.degrees(np.arctan2(basis[1, 0], basis[0, 0])))
            camera = viewer.viewer.cam
            camera.azimuth = heading + {"rear": 0, "side": -90, "rear-oblique": 45}[args.view]
            camera.elevation = -13 if args.view == "rear" else -20
            camera.distance = 4.2 if args.view == "rear" else 3.6
            if len(a.get("terrain_box_centers_w", [])):
                camera.lookat[:] = a["terrain_box_centers_w"][0] + basis @ np.array(
                    [-a["terrain_box_half_sizes"][0, 0] - 0.05, 0, 0]
                )
                camera.lookat[2] = 0.9 if args.view == "rear" else 1.1
        # The GMR viewer expects native MuJoCo joint order; map by joint name.
        native = np.argsort(qids)
        source_points = None
        if args.show_source:
            import retarget

            row = next(r for r in retarget.INDEX["motions"] if r["id"] == str(a["source_id"]))
            source = retarget.source_motion(row, retarget.G1Kinematics())
            source_points = source[1] * float(a["morphology_scale"]) + a.get(
                "source_world_translation", np.zeros(3)
            )
            print(
                "Green: scaled source skeleton aligned to the body reference; contact targets stay on the terrain. Limb proportions differ.",
                flush=True,
            )
        print(
            "Space: pause/resume; left/right: one frame; S: source overlay; Ctrl+C: exit",
            flush=True,
        )
        try:
            while viewer.viewer.is_running():
                frame = state["frame"]
                viewer.viewer.user_scn.ngeom = 0
                if source_points is not None and state["source"]:
                    source_frame = (
                        int(a["source_frame_indices"][frame])
                        if "source_frame_indices" in a
                        else frame
                    )
                    points = source_points[source_frame]
                    for first, last in [
                        (0, 1),
                        (0, 2),
                        (2, 3),
                        (3, 4),
                        (1, 5),
                        (5, 6),
                        (6, 7),
                        (0, 8),
                        (8, 9),
                        (9, 10),
                        (1, 11),
                        (11, 12),
                        (12, 13),
                    ]:
                        scene = viewer.viewer.user_scn
                        g = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.ones(3) * 0.006,
                            np.zeros(3),
                            np.eye(3).ravel(),
                            np.array([0.1, 0.95, 0.2, 1.0]),
                        )
                        mujoco.mjv_connector(
                            g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, points[first], points[last]
                        )
                        scene.ngeom += 1
                viewer.step(
                    a["base_pos_w"][frame],
                    a["base_quat_wxyz"][frame],
                    a["joint_pos"][frame, native],
                    rate_limit=True,
                    follow_camera=False,
                )
                if not state["paused"]:
                    if args.once and frame == len(a["timestamps"]) - 1:
                        state["paused"] = True
                    else:
                        state["frame"] = (frame + 1) % len(a["timestamps"])
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
