"""Minimal read-only source loader and FK model builder, extracted from local tools."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .paths import ASSET, URDF


def load(name):
    p = Path(name)
    if not p.is_file():
        raise FileNotFoundError(p)
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
    raise ValueError("Explicit terrain boxes are required; prepare them before training")


def _robot_mesh_geometry(xml):
    """Replace the legacy capsule approximation with source robot meshes."""
    urdf_path = URDF
    source = ET.parse(urdf_path).getroot()
    bodies = {body.get("name"): body for body in xml.findall(".//worldbody//body")}
    assets = xml.find("asset")
    mesh_names = {mesh.get("name") for mesh in assets.findall("mesh")}
    for body in bodies.values():
        for geom in list(body.findall("geom")):
            if geom.get("class") != "visual":
                body.remove(geom)
    # The historical XML pairs refer to the removed foot capsules.
    for element in list(xml.findall("contact")):
        xml.remove(element)
    contact = ET.SubElement(xml, "contact")
    for link in source.findall("link"):
        name = link.get("name")
        if name not in bodies:
            raise ValueError(f"MuJoCo model is missing URDF body {name}")
        body = bodies[name]
        # Use source inertials, rather than rounded values in the legacy XML.
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            if origin is not None and any(
                float(v) != 0 for v in origin.get("rpy", "0 0 0").split()
            ):
                raise ValueError("Rotated URDF inertials require explicit conversion")
            for old in body.findall("inertial"):
                body.remove(old)
            inertia = inertial.find("inertia")
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0" if origin is None else origin.get("xyz", "0 0 0"),
                mass=inertial.find("mass").get("value"),
                fullinertia=" ".join(
                    inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ),
            )
        collisions = list(link.findall("collision"))
        if name in ("l_knee_y_link", "r_knee_y_link"):
            collision = ET.Element("collision")
            ET.SubElement(
                ET.SubElement(collision, "geometry"), "mesh", filename=f"../meshes/{name}.STL"
            )
            collisions.append(collision)
        for index, collision in enumerate(collisions):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Unsupported non-mesh source collision: {name}")
            mesh_name = Path(mesh.get("filename")).stem
            if mesh.get("scale", "1 1 1") != "1 1 1":
                raise ValueError("Scaled source collision needs a distinct mesh asset")
            if mesh_name not in mesh_names:
                ET.SubElement(assets, "mesh", name=mesh_name, file=f"{mesh_name}.STL")
                mesh_names.add(mesh_name)
            origin = collision.find("origin")
            kwargs = (
                {}
                if origin is None
                else {"pos": origin.get("xyz", "0 0 0"), "euler": origin.get("rpy", "0 0 0")}
            )
            ET.SubElement(
                body,
                "geom",
                name=f"hiking_{name}_{index}",
                type="mesh",
                mesh=mesh_name,
                group="3",
                contype="1",
                conaffinity="1",
                density="0",
                rgba="0.3 0.6 0.3 .25",
                friction="1 .005 .0001",
                **kwargs,
            )
    # Same-leg exclusions from the Hiking USD overlay. MuJoCo additionally
    # applies its normal parent-child collision filtering.
    for side in ("l", "r"):
        for adjacent in (f"{side}_hip_z_link", f"{side}_ankle_x_link"):
            ET.SubElement(contact, "exclude", body1=f"{side}_knee_y_link", body2=adjacent)


def model_file(folder, a, show_terrain, include_gripper=True):
    root = ET.parse(ASSET / "xml/elf3.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET / "meshes"))
    for tag in ("include", "keyframe", "equality", "contact"):
        for node in list(root.findall(tag)):
            root.remove(node)

    _robot_mesh_geometry(root)
    if include_gripper:
        from .gripper import apply

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
