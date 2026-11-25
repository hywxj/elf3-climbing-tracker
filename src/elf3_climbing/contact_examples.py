"""Physical observations shared by the contact tracker and its offline teachers."""

import torch

from .isaac_knee_entry import KNEE_PATCH_RADIUS_M, top_face_points


CONTACT_NAMES = (
    "left_foot",
    "right_foot",
    "left_forearm",
    "right_forearm",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
)


def student_contact_state(env, *, with_points=False):
    """Eight positions in clone-local world axes and measured top-face support in N."""
    contacts = env.knee_contacts
    physical_sites = env.site_positions()
    wrist_ids = getattr(env, "_student_wrist_ids", None)
    if wrist_ids is None:
        wrist_ids = env.robot.find_bodies(
            ["l_wrist_y_link", "r_wrist_y_link"], preserve_order=True
        )[0]
        env._student_wrist_ids = wrist_ids
    sites = physical_sites.clone()
    sites[:, 4:6] = env.robot.data.body_link_pos_w[:, wrist_ids]
    origins = env.scene.env_origins
    forces = origins.new_zeros((len(origins), 8))
    moments = origins.new_zeros((len(origins), 8, 3)) if with_points else None
    bodies = [
        *[(side, name) for side, name in enumerate(contacts.foot_names)],
        *[(side + 2, name) for side, name in enumerate(contacts.arm_names)],
        *[
            (side + 4, f"{prefix}_wrist_{axis}_link")
            for side, prefix in enumerate(("l", "r"))
            for axis in ("y", "z")
        ],
        *[(side + 6, name) for side, name in enumerate(contacts.knee_names)],
    ]
    for channel, name in bodies:
        sensor = env.support_sensors[name]
        detail = sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        surfaces = [(contacts.upper, contacts.top_heights)]
        if channel < 2:
            surfaces.append((contacts.lower, contacts.floor_heights))
        for box, heights in surfaces:
            rows, xyz, nz, keep, normal = top_face_points(detail, box, heights)
            if channel >= 6:
                keep &= (xyz - sites[rows, channel]).norm(dim=1) <= KNEE_PATCH_RADIUS_M
            weight = torch.where(keep, normal * nz, 0.0)
            forces[:, channel].index_add_(0, rows, weight)
            if with_points:
                moments[:, channel].index_add_(0, rows, weight[:, None] * (xyz - origins[rows]))
    result = sites - origins[:, None], forces
    if with_points:
        centers = torch.where(
            forces[:, :, None] > 0, moments / forces[:, :, None].clamp_min(1e-8), torch.nan
        )
        return *result, centers
    return result
