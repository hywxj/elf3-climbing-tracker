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
        m = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True)))
        d = mujoco.MjData(m)
        joints = [m.jnt_qposadr[m.joint(str(n)).id] for n in env.reference_np["joint_names"]]
        wrist = m.body("l_wrist_z_link").id
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
    return {"status": "PASS", "scope": "Reset-only FK, not dynamics", "frames": records}
