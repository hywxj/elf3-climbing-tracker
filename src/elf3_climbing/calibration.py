"""Reset-only model calibration, independent of PD tracking performance."""


def check_model_frames(env):
    """Reset-only FK calibration; explicitly not a dynamics rollout."""
    import tempfile
    import numpy as np
    import torch
    import mujoco
    from .model import load, model_file
    from .gripper import specification

    _, source = load(env.reference_manifest["source"])
    original_start = env.segment_start
    original_random = env.cfg.random_start
    samples = sorted({original_start, (original_start + env.segment_end) // 2, env.segment_end})
    records = []
    cfg = specification()
    wrist_id = env.robot.find_bodies("l_wrist_z_link", preserve_order=True)[0][0]
    with tempfile.TemporaryDirectory() as folder:
        baseline = mujoco.MjModel.from_xml_path(
            str(model_file(folder, source, True, include_gripper=False))
        )
        m = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True)))
        d = mujoco.MjData(m)
        joints = [m.jnt_qposadr[m.joint(str(n)).id] for n in env.reference_np["joint_names"]]
        wrist = m.body("l_wrist_z_link").id
        all_masses = env.robot.root_physx_view.get_masses().cpu().numpy().astype(np.float64)
        actual_mass = all_masses[:, wrist_id]
        actual_com = env.robot.data.body_com_pos_b[:, wrist_id].cpu().numpy()
        actual_inertias = (
            env.robot.root_physx_view.get_inertias()[:, wrist_id].cpu().numpy().reshape(-1, 3, 3)
        )
        from scipy.spatial.transform import Rotation

        # PhysX get_inertias already uses the rigid-body-prim frame at COM.
        # Its flattened matrices are column-major; do not rotate by principal axes again.
        actual_link_inertia = actual_inertias.transpose(0, 2, 1).astype(np.float64)
        mj_rotation = Rotation.from_quat(m.body_iquat[wrist], scalar_first=True).as_matrix()
        mj_link_inertia = mj_rotation @ np.diag(m.body_inertia[wrist]) @ mj_rotation.T
        expected_added = cfg["mass_kg"] if cfg["enabled"] else 0.0
        added = float(m.body_mass.sum() - baseline.body_mass.sum())
        mass_error = float(np.abs(actual_mass - m.body_mass[wrist]).max())
        com_error = float(np.linalg.norm(actual_com - m.body_ipos[wrist], axis=-1).max())
        inertia_error = float(np.abs(actual_link_inertia - mj_link_inertia).max())
        inertial_check = {
            "configured_added_mass_kg": expected_added,
            "baseline_total_mass_kg": float(baseline.body_mass.sum()),
            "mujoco_total_mass_kg": float(m.body_mass.sum()),
            "physx_total_mass_kg": float(all_masses[0].sum()),
            "mujoco_added_total_mass_kg": added,
            "mujoco_wrist_mass_kg": float(m.body_mass[wrist]),
            "physx_wrist_mass_min_kg": float(actual_mass.min()),
            "physx_wrist_mass_max_kg": float(actual_mass.max()),
            "wrist_mass_error_kg": mass_error,
            "wrist_com_error_m": com_error,
            "wrist_inertia_error_kg_m2": inertia_error,
            "physx_wrist_inertia_in_link_kg_m2": actual_link_inertia[0].tolist(),
            "mujoco_wrist_inertia_in_link_kg_m2": mj_link_inertia.tolist(),
        }
        if (
            abs(added - expected_added) > 1e-8
            or mass_error > 1e-5
            or com_error > 1e-5
            or inertia_error > 1e-6
        ):
            raise RuntimeError(f"Gripper inertial mismatch: {inertial_check}")
        inertial_check["status"] = "PASS"
        try:
            env.cfg.random_start = False
            for frame in samples:
                env.segment_start = frame
                env.reset()
                d.qpos[:3] = env.reference_np["root_pos"][frame]
                d.qpos[3:7] = env.reference_np["root_quat"][frame]
                d.qpos[joints] = env.reference_np["joint_pos"][frame]
                mujoco.mj_forward(m, d)
                actual_q = env.robot.data.body_link_quat_w[0, wrist_id].cpu().numpy()
                actual_p = (
                    (env.robot.data.body_link_pos_w[0, wrist_id] - env.scene.env_origins[0])
                    .cpu()
                    .numpy()
                )
                from scipy.spatial.transform import Rotation

                actual_R = Rotation.from_quat(actual_q, scalar_first=True).as_matrix()
                expected_R = d.xmat[wrist].reshape(3, 3)
                center = np.array(cfg["center_in_wrist_m"])
                actual_center = actual_p + actual_R @ center
                expected_center = d.xpos[wrist] + expected_R @ center
                site_error = torch.linalg.vector_norm(
                    env.site_positions()
                    - env.scene.env_origins[:, None]
                    - env.ref["site_pos"][frame],
                    dim=-1,
                )
                records.append(
                    {
                        "reference_frame": frame,
                        "site_error_max_m": float(site_error.max()),
                        "gripper_center_error_m": float(
                            np.linalg.norm(actual_center - expected_center)
                        ),
                        "gripper_orientation_error_rad": float(
                            (
                                Rotation.from_matrix(actual_R)
                                * Rotation.from_matrix(expected_R).inv()
                            ).magnitude()
                        ),
                    }
                )
        finally:
            env.segment_start = original_start
            env.cfg.random_start = original_random
            env.reset()
    if any(
        r["site_error_max_m"] > 0.002
        or r["gripper_center_error_m"] > 0.002
        or r["gripper_orientation_error_rad"] > 0.01
        for r in records
    ):
        raise RuntimeError(f"MuJoCo/Isaac calibration mismatch: {records}")
    return {
        "status": "PASS",
        "scope": "Reset-only FK and loaded inertial parameters, not dynamics acceptance",
        "frames": records,
        "gripper_inertials": inertial_check,
    }
