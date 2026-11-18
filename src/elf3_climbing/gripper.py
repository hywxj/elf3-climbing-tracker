"""Left-gripper envelope and user-provided mass with approximate COM/inertia."""

from .paths import GRIPPER_CONFIG
from .inertia import combine
from pathlib import Path
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
    dynamics_path = Path(__file__).with_name("gripper_dynamics.json")
    dynamics_raw = dynamics_path.read_bytes()
    dynamics = json.loads(dynamics_raw)
    mass = float(dynamics["mass_kg"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Gripper mass must be finite and positive")
    if (
        dynamics["center_of_mass_model"] != "collision_envelope_center"
        or dynamics["inertia_model"] != "uniform_collision_envelope_box"
    ):
        raise ValueError("Unsupported gripper inertial approximation")
    squared = (dimensions / 1000) ** 2
    inertia = R @ np.diag(mass / 12 * (squared.sum() - squared)) @ R.T
    return {
        **cfg,
        **dynamics,
        "center_of_mass_in_wrist_m": (center / 1000).tolist(),
        "inertia_about_com_in_wrist_kg_m2": inertia.tolist(),
        "dynamics_config_sha256": hashlib.sha256(dynamics_raw).hexdigest(),
        "analysis_scope": "Mass included; COM and inertia approximated, not measured",
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
    if body.find(f'geom[@name="{cfg["geom_name"]}"]') is not None:
        raise ValueError(
            "Gripper already applied; rebuild from the original model to avoid double mass"
        )
    original = body.find("inertial")
    if original is None or original.get("fullinertia") is None:
        raise ValueError("Expected explicit full wrist inertia from the URDF model builder")
    xx, yy, zz, xy, xz, yz = np.fromstring(original.get("fullinertia"), sep=" ")
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    mass, center, inertia = combine(
        float(original.get("mass")),
        np.fromstring(original.get("pos", "0 0 0"), sep=" "),
        tensor,
        cfg["mass_kg"],
        cfg["center_of_mass_in_wrist_m"],
        cfg["inertia_about_com_in_wrist_kg_m2"],
    )
    original.set("mass", format(mass, ".16g"))
    original.set("pos", " ".join(format(x, ".16g") for x in center))
    original.set(
        "fullinertia",
        " ".join(
            format(x, ".16g")
            for x in [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ]
        ),
    )
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


def apply_usd_mass(prim):
    """Modify only the spawned wrist prim; source USD files are never saved."""
    from pxr import Gf, UsdPhysics

    cfg = specification()
    if not cfg["enabled"]:
        return {"enabled": False, "added_mass_kg": 0.0}
    if prim.GetCustomDataByKey("elf3GripperMassAppliedKg") is not None:
        raise ValueError("Gripper mass already applied to this prim")
    api = UsdPhysics.MassAPI(prim)
    mass_old = api.GetMassAttr().Get()
    center_old = api.GetCenterOfMassAttr().Get()
    diag = api.GetDiagonalInertiaAttr().Get()
    axes = api.GetPrincipalAxesAttr().Get()
    if mass_old is None or center_old is None or diag is None or axes is None:
        raise ValueError("Expected explicitly authored wrist mass, COM and inertia")
    q = [axes.GetReal(), *axes.GetImaginary()]
    rotation = Rotation.from_quat(q, scalar_first=True).as_matrix()
    inertia_old = rotation @ np.diag(diag) @ rotation.T
    mass, center, inertia = combine(
        mass_old,
        center_old,
        inertia_old,
        cfg["mass_kg"],
        cfg["center_of_mass_in_wrist_m"],
        cfg["inertia_about_com_in_wrist_kg_m2"],
    )
    values, basis = np.linalg.eigh(inertia)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    quat = Rotation.from_matrix(basis).as_quat(scalar_first=True)
    api.CreateMassAttr().Set(mass)
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center.tolist()))
    api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*values.tolist()))
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:].tolist())))
    prim.SetCustomDataByKey("elf3GripperMassAppliedKg", cfg["mass_kg"])
    return {
        "added_mass_kg": cfg["mass_kg"],
        "original_wrist_mass_kg": float(mass_old),
        "combined_wrist_mass_kg": mass,
        "combined_com_in_wrist_m": center.tolist(),
        "combined_inertia_in_wrist_kg_m2": inertia.tolist(),
        "assumptions": cfg["assumptions"],
        "dynamics_config_sha256": cfg["dynamics_config_sha256"],
    }
