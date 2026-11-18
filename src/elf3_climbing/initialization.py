"""One-time static action-head initialization; no inverse dynamics in policy rollout."""

import tempfile

import mujoco
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .model import load, model_file


def static_action_bias(env):
    """Balance gravity with unilateral foot contacts, then offset position targets by tau/Kp.

    This only initializes the network's constant output bias. No reference actions,
    support forces or inverse dynamics are injected while the learned policy runs.
    """
    _, source = load(env.reference_manifest["source"])
    with tempfile.TemporaryDirectory() as folder:
        m = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True)))
        d = mujoco.MjData(m)
        i = env.segment_end
        names = env.reference_np["joint_names"].tolist()
        qids = [m.jnt_qposadr[m.joint(n).id] for n in names]
        vids = [m.jnt_dofadr[m.joint(n).id] for n in names]
        q = env.reference_np["joint_pos"][i].copy()
        d.qpos[:3] = env.reference_np["root_pos"][i]
        d.qpos[3:7] = env.reference_np["root_quat"][i]
        d.qpos[qids] = q
        mujoco.mj_forward(m, d)
        feet = {m.body("l_ankle_x_link").id, m.body("r_ankle_x_link").id}
        points, body_ids = [], []
        for c in d.contact:
            a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if ((a in feet and b == 0) or (b in feet and a == 0)) and abs(c.dist) <= 0.002:
                if abs(c.frame[2]) < 0.999:
                    raise ValueError("Static seed requires horizontal foot contacts")
                points.append(c.pos.copy())
                body_ids.append(a if a in feet else b)
        if set(body_ids) != feet or len(points) < 4:
            raise ValueError(
                "Static seed needs contact points under both feet; use --no-pose-init or a verified checkpoint"
            )
        jacobians = []
        for point, body in zip(points, body_ids):
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, point, body)
            jacobians.append(jp)
        J = np.concatenate(jacobians)
        weight = m.body_mass.sum() * np.linalg.norm(m.opt.gravity)
        n = len(points)
        guess = np.zeros((n, 3))
        guess[:, 2] = weight / n
        A = J[:, :6].T
        target = d.qfrc_bias[:6].copy()
        friction = np.zeros((4 * n, 3 * n))
        # Inscribed square: each feasible force also lies inside the mu=0.6 cone.
        mu = 0.6 / np.sqrt(2)
        for j in range(n):
            for k, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
                friction[4 * j + k, 3 * j + axis] = sign
                friction[4 * j + k, 3 * j + 2] = mu
        result = minimize(
            lambda force: 0.5 * np.sum(((force - guess.ravel()) / weight) ** 2),
            guess.ravel(),
            jac=lambda force: (force - guess.ravel()) / weight**2,
            method="SLSQP",
            bounds=Bounds(np.tile([-weight, -weight, 0], n), np.full(3 * n, weight)),
            constraints=[
                LinearConstraint(A, target, target),
                LinearConstraint(friction, 0, np.inf),
            ],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        force = result.x
        residual = np.linalg.norm(A @ force - target)
        if not result.success or residual > 1e-6 or (friction @ force).min() < -1e-7:
            raise ValueError(f"No valid static seed: {result.message}; residual={residual}")
        torque = d.qfrc_bias[vids] - J[:, vids].T @ force
        stiffness = env.robot.data.joint_stiffness[0, env.joint_ids].cpu().numpy().astype(float)
        effort = env.robot.data.joint_effort_limits[0, env.joint_ids].cpu().numpy()
        if (stiffness <= 0).any() or (abs(torque) > effort).any():
            raise ValueError("Static seed exceeds actuator capability")
        unconstrained = q + torque / stiffness
        limits = env.limits[0].cpu().numpy()
        desired = np.clip(unconstrained, limits[:, 0], limits[:, 1])
        clipped = np.flatnonzero(abs(desired - unconstrained) > 1e-8)
        # The wrist flip reference is at its limit. A bounded initial bias cannot
        # exactly compensate its gravity torque; record that residual, never move
        # the mechanical limit or claim this is a static equilibrium controller.
        unbalanced_torque = torque - stiffness * (desired - q)
        action = (desired - env.nominal[0].cpu().numpy()) / env.cfg.action_scale
        if abs(action).max() > 3:
            raise ValueError("Static action bias exceeds policy action range")
        return action, {
            "method": "static gravity-aware constant action-head bias",
            "reference_frame": int(i),
            "joint_names": names,
            "joint_target": desired.tolist(),
            "position_compensation_rad": (desired - q).tolist(),
            "clipped_joint_names": [names[j] for j in clipped],
            "uncompensated_torque_Nm": unbalanced_torque.tolist(),
            "estimated_static_joint_torque_Nm": torque.tolist(),
            "contact_points_m": np.asarray(points).tolist(),
            "contact_forces_N": force.reshape(-1, 3).tolist(),
            "floating_base_residual": float(residual),
            "max_torque_limit_ratio": float((abs(torque) / effort).max()),
            "scope": "Initialization estimate only; no runtime force injection or guarantee of stability",
        }
