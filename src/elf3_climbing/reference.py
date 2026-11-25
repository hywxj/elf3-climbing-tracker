"""Read-only bridge from local v7 candidates to a 50 Hz contact tracker reference."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .paths import DEFAULT_SOURCE, URDF, MJCF, GRIPPER_CONFIG

# Foot targets merge the source's ground/top slots. Forearm and elbow are two
# surface points on the same forearm body, not two independent rigid contacts.
SITE_NAMES = [
    "left_foot",
    "right_foot",
    "left_forearm",
    "right_forearm",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
]
SITE_BODIES = [
    "l_ankle_x_link",
    "r_ankle_x_link",
    "l_wrist_x_link",
    "r_wrist_x_link",
    "l_wrist_x_link",
    "r_wrist_x_link",
    "l_knee_y_link",
    "r_knee_y_link",
]
SITE_OFFSETS = np.array(
    [
        [0.03, 0, -0.04],
        [0.03, 0, -0.04],
        [-0.094, 0, -0.0329],
        [-0.094, 0, -0.0329],
        [-0.183, 0, -0.0329],
        [-0.183, 0, -0.0329],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
SEGMENT_PHASES = {"flip": ["READY", "GRIPPER_FLIP"], "stand": ["STAND", "FINISHED"]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_contact_commands(time, points, flags, horizon=1.5):
    """Use event onset anchors; never interpolate boolean contacts across edges."""
    targets = np.zeros_like(points)
    valid = np.zeros_like(flags)
    for b in range(flags.shape[1]):
        edges = np.diff(np.r_[False, flags[:, b], False].astype(int))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            # First eligible event wins, including when two events fall in the horizon.
            eligible = (time >= time[start] - horizon - 1e-9) & (time < time[stop - 1] + 1e-9)
            eligible &= ~valid[:, b]
            targets[eligible, b] = points[start, b]
            valid[eligible, b] = True
    return targets, valid


def fit_forearm_surface(model, data, arm, body, dz):
    """Match tabletop height and full forearm orientation, allowing tangential sliding."""
    import mujoco
    from scipy.optimize import least_squares

    indices, limits = arm
    initial = data.qpos[indices].copy()
    target = data.xpos[body].copy()
    target[2] += dz
    orientation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))

    def residual(values):
        data.qpos[indices] = values
        mujoco.mj_kinematics(model, data)
        position = data.xpos[body] - target
        current = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
        angle = (current * orientation.inv()).as_rotvec()
        return np.r_[position[2], 0.25 * angle, 0.01 * position[:2], 0.0005 * (values - initial)]

    solution = least_squares(residual, initial, bounds=tuple(limits.T), ftol=1e-10,
                             xtol=1e-10, gtol=1e-10, max_nfev=60)
    residual(solution.x)
    mujoco.mj_forward(model, data)
    offset = data.xpos[body] - target
    angle = (Rotation.from_matrix(data.xmat[body].reshape(3, 3)) * orientation.inv()).magnitude()
    return abs(float(offset[2])), float(angle), float(np.linalg.norm(offset[:2]))


def prepare(source, outdir, platform_height=None):
    from .model import load, model_file
    import mujoco

    source, src = load(str(source))
    before = digest(source)
    t0 = src["timestamps"]
    t = np.arange(int(np.floor((t0[-1] - t0[0]) * 50 + 1e-6)) + 1) / 50 + t0[0]
    q = np.stack([np.interp(t, t0, col) for col in src["joint_pos"].T], axis=1)
    pos = np.stack([np.interp(t, t0, col) for col in src["base_pos_w"].T], axis=1)
    rotation = Slerp(t0, Rotation.from_quat(src["base_quat_wxyz"], scalar_first=True))(t)
    quat = rotation.as_quat(scalar_first=True)
    # World-frame angular velocity, consistent with Isaac root velocity writes.
    ang = np.empty((len(t), 3))
    for i in range(len(t)):
        lo, hi = max(0, i - 1), min(len(t) - 1, i + 1)
        ang[i] = (rotation[hi] * rotation[lo].inv()).as_rotvec() / (t[hi] - t[lo])
    index = np.searchsorted(t0, t + 1e-10, side="right") - 1
    names = src["contact_region_names"].tolist()
    mask = src["geometric_contact_mask"][index]
    flags = np.stack(
        [
            mask[:, names.index("left_ground_foot")] | mask[:, names.index("left_top_foot")],
            mask[:, names.index("right_ground_foot")] | mask[:, names.index("right_top_foot")],
            *[mask[:, names.index(k)] for k in SITE_NAMES[2:]],
        ],
        axis=1,
    )
    phase = src["phase"][index]
    centres = src["terrain_box_centers_w"].copy()
    sizes = src["terrain_box_half_sizes"].copy()
    tops = centres[:, 2] + sizes[:, 2]
    upper = int(np.argmax(tops))
    source_rise = float(tops.max() - tops.min())
    arm_shift = np.zeros(len(t))
    height_error = angle_error = tangential_shift = 0.0
    if platform_height is not None:
        if not np.isfinite(platform_height) or platform_height <= 0 or len(tops) != 2:
            raise ValueError("Expected a positive rise and a two-level platform")
        delta = platform_height - source_rise
        centres[upper, 2] += delta / 2
        sizes[upper, 2] += delta / 2
        if sizes[upper, 2] <= 0:
            raise ValueError("Platform has nonpositive thickness")
        def ramp(begin, end):
            u = np.clip((np.arange(len(t)) - begin) / (end - begin), 0, 1)
            return u * u * (3 - 2 * u)
        first_arm = np.flatnonzero(flags[:, 2:6].any(1))[0]
        takeoff = np.flatnonzero((np.arange(len(t)) > first_arm) & ~flags[:, :2].any(1))[0]
        first_knee = np.flatnonzero(flags[:, 7])[0]
        # Feet stay on the ground through release. Move the arms during stance,
        # then raise the airborne body to match the knee landing.
        root_shift = delta * ramp(takeoff, first_knee)
        pos[:, 2] += root_shift
        arm_shift = delta * ramp(np.flatnonzero(phase == "ARM_HOVER")[0], first_arm) - root_shift
    points = np.empty((len(t), 8, 3))
    body_quat = np.empty((len(t), 8, 4))
    with tempfile.TemporaryDirectory() as tmp:
        m = mujoco.MjModel.from_xml_path(str(model_file(tmp, src, True)))
        d = mujoco.MjData(m)
        joints = [m.jnt_qposadr[m.joint(str(n)).id] for n in src["joint_names"]]
        bodies = [m.body(n).id for n in SITE_BODIES]
        arms = []
        for side in ("l", "r"):
            ids = [m.joint(f"{side}_{name}_joint").id for name in
                   ("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y", "wrist_x")]
            arms.append((m.jnt_qposadr[ids], m.jnt_range[ids]))
        for i in range(len(t)):
            d.qpos[:3] = pos[i]
            d.qpos[3:7] = quat[i]
            d.qpos[joints] = q[i]
            mujoco.mj_forward(m, d)
            if abs(arm_shift[i]) > 1e-8:
                for arm, body in zip(arms, bodies[2:4]):
                    height, angle, slide = fit_forearm_surface(m, d, arm, body, arm_shift[i])
                    height_error = max(height_error, height)
                    angle_error = max(angle_error, angle)
                    tangential_shift = max(tangential_shift, slide)
                q[i] = d.qpos[joints]
            points[i] = d.xpos[bodies] + np.einsum(
                "bij,bj->bi", d.xmat[bodies].reshape(-1, 3, 3), SITE_OFFSETS
            )
            body_quat[i] = d.xquat[bodies]
    if height_error > 0.0005 or angle_error > 0.002:
        raise ValueError(f"Forearm pose error after height adjustment: {height_error:.5f} m, {angle_error:.5f} rad")
    vel = np.gradient(q, 0.02, axis=0, edge_order=2)
    lin = np.gradient(pos, 0.02, axis=0, edge_order=2)
    segments = {}
    for name, labels in SEGMENT_PHASES.items():
        ix = np.flatnonzero(np.isin(phase, labels))
        assert len(ix) > 2 and np.all(np.diff(ix) == 1)
        segments[name] = [int(ix[0]), int(ix[-1])]
    if platform_height is not None:
        segments["knee_entry"] = [int(np.flatnonzero(phase == "FOREARM_HOLD")[0]),
                                 int(np.flatnonzero(phase == "RIGHT_KNEE")[-1])]
    command, valid = make_contact_commands(t, points, flags)
    pack = dict(
        schema=np.array("elf3_contact_tracker_v1"),
        fps=np.array(50.0),
        timestamps=t,
        joint_names=src["joint_names"],
        joint_pos=q,
        joint_vel=vel,
        root_pos=pos,
        root_quat=quat,
        root_lin_vel=lin,
        root_ang_vel=ang,
        site_names=np.array(SITE_NAMES),
        site_bodies=np.array(SITE_BODIES),
        site_offsets=SITE_OFFSETS,
        site_pos=points,
        site_quat=body_quat,
        contact_mask=flags,
        contact_targets=command,
        contact_valid=valid,
        phase=phase,
        segments_json=np.array(json.dumps(segments)),
        source_sha256=np.array(before),
        terrain_box_centers_w=centres,
        terrain_box_quat_wxyz=src["terrain_box_quat_wxyz"],
        terrain_box_half_sizes=sizes,
    )
    for k, v in pack.items():
        if v.dtype.kind in "fiu" and not np.isfinite(v).all():
            raise ValueError(k)
    outdir = Path(outdir).expanduser().resolve()
    if outdir.exists():
        raise FileExistsError(f"Choose a new output directory: {outdir}")
    outdir.mkdir(parents=True)
    np.savez_compressed(outdir / "reference.npz", **pack)
    meta = dict(
        source=os.path.relpath(source, outdir),
        source_sha256=before,
        pack_sha256=digest(outdir / "reference.npz"),
        hashes={os.path.relpath(p, outdir): digest(p) for p in [URDF, MJCF, GRIPPER_CONFIG]},
        segments=segments,
        samples=len(t),
        fps=50,
        labels="Geometric reference labels, NOT measured forces",
        contact_points="Foot sole / forearm surface offsets; knee link origin is a tracking marker, NOT a surface contact point",
        source_platform_rise_m=source_rise,
        platform_rise_m=float(np.ptp(centres[:, 2] + sizes[:, 2])),
        height_adaptation=("Match forearm surface height and orientation, preserve grounded legs, lift root after takeoff"
                           if platform_height is not None else "Source heights"),
        forearm_height_fit_error_max_m=height_error,
        forearm_angle_fit_error_max_rad=angle_error,
        forearm_tangential_shift_max_m=tangential_shift,
        scope="Kinematic prior for segment training; contact forces evaluated in simulation",
        full_motion_training_ready=False,
        real_lidar="NOT CONNECTED",
        resampling="Linear positions, quaternion SLERP, backward-held categorical labels; derivatives recomputed",
    )
    (outdir / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    assert digest(source) == before
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def read_pack(path):
    path = Path(path).expanduser().resolve()
    meta = json.loads(path.with_name("manifest.json").read_text())
    if digest(path) != meta["pack_sha256"]:
        raise ValueError("Reference hash mismatch")
    for source, h in meta["hashes"].items():
        asset_path = Path(source)
        if not asset_path.is_absolute():
            asset_path = path.parent / asset_path
        if digest(asset_path) != h:
            raise ValueError(f"Asset changed: {source}; rebuild reference")
    source_path = Path(meta["source"])
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    if digest(source_path) != meta["source_sha256"]:
        raise ValueError("Source motion hash mismatch")
    meta["source"] = str(source_path.resolve())
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    if str(a["schema"]) != "elf3_contact_tracker_v1" or float(a["fps"]) != 50:
        raise ValueError("Unknown contract")
    if len(set(a["joint_names"].tolist())) != 29:
        raise ValueError("Invalid joints")
    return a, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-height", type=float, help="Target rise above the approach floor in metres")
    args = parser.parse_args()
    prepare(args.source, args.output, args.platform_height)
