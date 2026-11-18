"""Single configurable left-gripper box; no CAD conversion or dynamics changes."""

from .paths import GRIPPER_CONFIG
import hashlib, json, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

CONFIG = GRIPPER_CONFIG


def specification():
    raw = CONFIG.read_bytes()
    cfg = json.loads(raw)
    if cfg["body"] != "l_wrist_z_link":
        raise ValueError("Expected a left-wrist envelope")
    length = cfg["gripper_length_mm"] + cfg["adapter_axial_allowance_mm"]
    padding = cfg["padding_each_face_mm"]
    dimensions = (
        np.array([length, cfg["gripper_open_width_mm"], cfg["gripper_thickness_mm"]]) + 2 * padding
    )
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or padding < 0:
        raise ValueError("Invalid envelope dimensions")
    q = np.asarray(cfg["mount_quaternion_wxyz"], float)
    if not np.isclose(np.linalg.norm(q), 1.0):
        raise ValueError("Mount quaternion must be unit length")
    R = Rotation.from_quat(q, scalar_first=True).as_matrix()
    center = np.asarray(cfg["mount_plane_position_mm"]) + R @ np.array([length / 2, 0.0, 0.0])
    return {
        **cfg,
        "dimensions_mm": dimensions.tolist(),
        "center_in_wrist_m": (center / 1000).tolist(),
        "half_size_m": (dimensions / 2000).tolist(),
        "config_path": str(CONFIG),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
    }


def apply(root):
    cfg = specification()
    if not cfg["enabled"]:
        return cfg
    body = root.find(f'.//body[@name="{cfg["body"]}"]')
    if body is None:
        raise ValueError("Left wrist body not found")
    for geom in list(body.findall("geom")):
        if geom.get("name") == cfg["geom_name"]:
            body.remove(geom)
    text = lambda values: " ".join(format(float(v), ".12g") for v in values)
    ET.SubElement(
        body,
        "geom",
        name=cfg["geom_name"],
        type="box",
        pos=text(cfg["center_in_wrist_m"]),
        size=text(cfg["half_size_m"]),
        quat=text(cfg["mount_quaternion_wxyz"]),
        contype="1",
        conaffinity="1",
        group="1",
        density="0",
        rgba="1 .42 .04 .45",
    )
    return cfg
