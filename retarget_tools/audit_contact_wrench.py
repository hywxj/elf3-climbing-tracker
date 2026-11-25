"""Necessary inverse-dynamics contact-force screen; not closed-loop execution.

Uses finite-difference derivatives, current ELF3 inertias, non-tensile forces at
near-surface mesh vertices, a conservative friction pyramid, and URDF effort
limits. No added gripper mass, impacts, compliance, or motor thermal model.
"""

from project_paths import URDF
import argparse, json, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import linprog
from plan_knee_entry import create_context
from view_motion import load


def audit(motion, output, mu, phases=None):
    p = create_context()
    m = p.m
    d = p.d
    path, a = load(motion)
    fps = float(a["fps"])
    N = len(a["timestamps"])
    q = np.tile(m.qpos0, (N, 1))
    q[:, :7] = a["qpos"][:, :7]
    q[:, p.ids] = a["joint_pos"]
    vel = np.zeros((N, m.nv))
    for i in range(N):
        lo = max(0, i - 1)
        hi = min(N - 1, i + 1)
        mujoco.mj_differentiatePos(m, vel[i], (hi - lo) / fps, q[lo], q[hi])
    acc = np.gradient(vel, 1 / fps, axis=0, edge_order=2)
    urdf = URDF
    xml = ET.parse(urdf).getroot()
    effort = np.array([float(xml.find(f'joint[@name="{n}"]/limit').get("effort")) for n in p.names])
    dofs = np.array([m.jnt_dofadr[j] for j in p.ji])
    rows = []
    mass = float(m.body_mass.sum())
    tol = 0.002

    def support_points():
        points = []
        for g, v in p.hulls.items():
            b = int(m.geom_bodyid[g])
            verts = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            local = (verts - p.box["center"]) @ p.R
            for z, inside in [
                (p.top, (np.abs(local[:, :2]) <= p.box["half_size"][:2] + 1e-5).all(1)),
                (p.floor, np.ones(len(verts), bool)),
            ]:
                mask = inside & (abs(verts[:, 2] - z) <= tol)
                for pt in verts[mask]:
                    points.append((b, pt.copy(), g))
        # Include solver-detected contact points for meshes whose vertices do not
        # include the contact patch interior. Only upward horizontal surfaces.
        for c in d.contact:
            gs = [int(c.geom1), int(c.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            if 0 not in bs or c.dist > tol or abs(c.frame[2]) < 0.9:
                continue
            k = 0 if bs[0] else 1
            points.append((bs[k], c.pos.copy(), gs[k]))
        unique = {}
        for b, pt, g in points:
            unique[(b, *np.round(pt, 4))] = (b, pt, g)
        return list(unique.values())

    selected_phases = phases or [
        "FOREARM_HOLD",
        "RAISE_AND_TUCK",
        "RIGHT_KNEE",
        "STATIC_SUPPORT_PROBE",
    ]
    selected = np.flatnonzero(np.isin(a["phase"], selected_phases))
    if not len(selected):
        raise ValueError("No frames match requested phases; no acceptance result can be produced")
    for i in selected:
        d.qpos[:] = q[i]
        d.qvel[:] = vel[i]
        mujoco.mj_forward(m, d)
        d.qacc[:] = acc[i]
        mass_matrix = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass_matrix)
        required = mass_matrix @ acc[i] + d.qfrc_bias - d.qfrc_passive
        points = support_points()
        nc = len(points)
        J = []
        for b, pt, g in points:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, pt, b)
            J.append(jac)
        if nc == 0:
            rows.append({"frame": int(i), "status": "FAIL_NO_CONTACTS"})
            continue
        A = np.concatenate([j.T for j in J], axis=1)
        fr = np.zeros((nc * 4, nc * 3))
        for k in range(nc):
            fr[4 * k : 4 * k + 4, 3 * k : 3 * k + 3] = [
                [1, 0, -mu / np.sqrt(2)],
                [-1, 0, -mu / np.sqrt(2)],
                [0, 1, -mu / np.sqrt(2)],
                [0, -1, -mu / np.sqrt(2)],
            ]
        bounds = [bound for k in points for bound in [(None, None), (None, None), (0, None)]]
        cost = np.tile([0, 0, 1.0], nc)
        Aub = np.r_[fr, -A[dofs], A[dofs]]
        bub = np.r_[np.zeros(nc * 4), effort - required[dofs], effort + required[dofs]]
        sol = linprog(
            cost, A_ub=Aub, b_ub=bub, A_eq=A[:6], b_eq=required[:6], bounds=bounds, method="highs"
        )
        relaxed = (
            sol
            if sol.success
            else linprog(
                cost,
                A_ub=fr,
                b_ub=np.zeros(nc * 4),
                A_eq=A[:6],
                b_eq=required[:6],
                bounds=bounds,
                method="highs",
            )
        )
        row = {
            "frame": int(i),
            "time_s": float(a["timestamps"][i]),
            "phase": str(a["phase"][i]),
            "contact_points": nc,
            "force_and_urdf_effort_feasible": bool(sol.success),
            "floating_base_wrench_feasible_without_effort_limits": bool(relaxed.success),
        }
        if sol.success:
            f = sol.x.reshape(-1, 3)
            tau = required[dofs] - A[dofs] @ sol.x
            row.update(
                max_effort_fraction=float(np.max(abs(tau) / effort)),
                total_normal_force_N=float(f[:, 2].sum()),
                normal_force_by_body_N={
                    m.body(b).name: float(
                        sum(f[k, 2] for k, item in enumerate(points) if item[0] == b)
                    )
                    for b in set(x[0] for x in points)
                },
                max_equality_residual=float(np.max(abs(A[:6] @ sol.x - required[:6]))),
            )
        rows.append(row)
    summary = {
        "status": "PASS_NECESSARY_SCREEN_ONLY"
        if all(r.get("force_and_urdf_effort_feasible", False) for r in rows)
        else "FAIL_NECESSARY_SCREEN",
        "frames": len(rows),
        "selected_phases": selected_phases,
        "unique_postures": len(np.unique(np.round(q[selected], 12), axis=0)),
        "force_and_urdf_effort_feasible_frames": sum(
            r.get("force_and_urdf_effort_feasible", False) for r in rows
        ),
        "floating_base_wrench_feasible_frames": sum(
            r.get("floating_base_wrench_feasible_without_effort_limits", False) for r in rows
        ),
        "model_mass_kg": mass,
        "mass_matrix_includes_model_armature": True,
        "model_armature_range": m.dof_armature[
            [int(np.argmin(m.dof_armature)), int(np.argmax(m.dof_armature))]
        ].tolist(),
        "assumed_friction_coefficient": mu,
        "contact_tolerance_m": tol,
        "closed_loop_dynamics": "NOT RUN",
        "scope": "Inverse dynamics feasibility at sampled poses. Finite-difference accelerations and near-contact point-force model; no impact impulses or contact no-slip acceleration constraints. FAIL rejects this sampled candidate under these assumptions, not the robot or climbing method.",
    }
    report = {
        "summary": summary,
        "motion": str(path),
        "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "urdf_effort_limits_Nm": dict(zip(p.names, effort.tolist())),
        "frames": rows,
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--phases", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    audit(args.motion, args.output, args.mu, args.phases)
