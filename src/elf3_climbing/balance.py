"""Ground standing task derived in memory; never rewrites the source motion."""

import numpy as np
from scipy.spatial.transform import Rotation


def ground_standing_reference(source):
    pack = {key: value.copy() for key, value in source.items()}
    feet = pack["site_pos"][0, :2]
    candidates = []
    for index, (center, quat, half) in enumerate(
        zip(
            pack["terrain_box_centers_w"],
            pack["terrain_box_quat_wxyz"],
            pack["terrain_box_half_sizes"],
        )
    ):
        rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        local = (feet - center) @ rotation
        if (
            np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-6)
            and (abs(local[:, :2]) < half[:2]).all()
            and (abs(local[:, 2] - half[2]) < 0.005).all()
        ):
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError("Standing requires one horizontal surface under both initial feet")
    for key in ("terrain_box_centers_w", "terrain_box_quat_wxyz", "terrain_box_half_sizes"):
        pack[key] = pack[key][candidates].copy()
    for key in ("root_lin_vel", "root_ang_vel", "joint_vel"):
        pack[key][0] = 0
    for key in ("contact_mask", "contact_valid"):
        pack[key][0] = False
        pack[key][0, :2] = True
    pack["contact_targets"][0] = 0
    pack["contact_targets"][0, :2] = feet
    wrist = pack["joint_names"].tolist().index("l_wrist_y_joint")
    if abs(pack["joint_pos"][0, wrist]) > 0.05:
        raise ValueError("Initial pose is not an unflipped gripper pose")
    return pack, {
        "task": "balance_ground_v1",
        "reference_frame": 0,
        "source_terrain_index": candidates[0],
        "ground_top_m": float(
            pack["terrain_box_centers_w"][0, 2] + pack["terrain_box_half_sizes"][0, 2]
        ),
        "wrist_target_rad": float(pack["joint_pos"][0, wrist]),
        "source_files_modified": False,
        "flip_required": False,
        "scope": "Static initial ground pose, not the post-climb stand segment",
    }
