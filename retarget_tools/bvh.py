from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load_bvh(path, stride=1):
    """BVH intrinsic Euler FK; preserves native units/up axis. Includes end sites."""
    text = Path(path).read_text()
    header, motion = text.split("MOTION", 1)
    import re

    tokens = re.findall(r"[{}]|[^\s{}]+", header)
    at = 0
    bones = []
    stack = []
    pending = None
    channels = 0
    while at < len(tokens):
        t = tokens[at]
        at += 1
        if t in ("ROOT", "JOINT"):
            name = tokens[at]
            at += 1
            pending = len(bones)
            bones.append(
                {
                    "name": name,
                    "parent": stack[-1] if stack else -1,
                    "offset": [0, 0, 0],
                    "channels": [],
                }
            )
        elif t == "End":
            at += 1
            pending = len(bones)
            bones.append(
                {
                    "name": bones[stack[-1]]["name"] + "__end",
                    "parent": stack[-1],
                    "offset": [0, 0, 0],
                    "channels": [],
                }
            )
        elif t == "{":
            stack.append(pending)
        elif t == "}":
            stack.pop()
        elif t == "OFFSET":
            bones[stack[-1]]["offset"] = list(map(float, tokens[at : at + 3]))
            at += 3
        elif t == "CHANNELS":
            n = int(tokens[at])
            at += 1
            names = tokens[at : at + n]
            at += n
            bones[stack[-1]]["channels"] = [(name, channels + i) for i, name in enumerate(names)]
            channels += n
    lines = motion.strip().splitlines()
    count = int(lines[0].split(":")[1])
    dt = float(lines[1].split(":")[1])
    values = np.loadtxt(lines[2:], dtype=float)
    if values.shape != (count, channels):
        raise ValueError(f"BVH data shape mismatch {values.shape}, {(count, channels)}")
    values = values[::stride]
    T = len(values)
    positions = np.zeros((T, len(bones), 3))
    rots = np.zeros((T, len(bones), 3, 3))
    rest_p = np.zeros((len(bones), 3))
    rest_R = np.tile(np.eye(3), (len(bones), 1, 1))
    for i, b in enumerate(bones):
        p = np.tile(b["offset"], (T, 1)).astype(float)
        order = []
        angles = []
        for name, col in b["channels"]:
            if name.endswith("position"):
                p[:, "XYZ".index(name[0])] += values[:, col]
            elif name.endswith("rotation"):
                order.append(name[0])
                angles.append(values[:, col])
        R = (
            Rotation.from_euler("".join(order), np.stack(angles, axis=1), degrees=True).as_matrix()
            if order
            else np.tile(np.eye(3), (T, 1, 1))
        )
        parent = b["parent"]
        if parent < 0:
            positions[:, i] = p
            rots[:, i] = R
            rest_p[i] = b["offset"]
        else:
            positions[:, i] = positions[:, parent] + np.einsum("tij,tj->ti", rots[:, parent], p)
            rots[:, i] = rots[:, parent] @ R
            rest_p[i] = rest_p[parent] + b["offset"]
    return {
        "names": [b["name"] for b in bones],
        "bones": bones,
        "positions": positions,
        "rotations": rots,
        "rest_positions": rest_p,
        "rest_rotations": rest_R,
        "fps": 1 / (dt * stride),
        "native_fps": 1 / dt,
        "frames": count,
        "frame_time": dt,
    }
