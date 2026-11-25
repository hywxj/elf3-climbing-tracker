"""Reference-guided joint targets for the first climbing stage in Isaac."""
import json
from pathlib import Path

import numpy as np
import torch


def example_arm_loads(path, motion_hash, platform_height):
    with np.load(Path(path), allow_pickle=False) as data:
        report = json.loads(str(data["report"]))
        if (not report["passed"] or str(data["motion_sha256"]) != motion_hash
                or not np.isclose(float(data["platform_height_m"]), platform_height)
                or str(data["columns"]) != "time,qpos,qvel,ctrl,loads[8],pelvis_drop"):
            raise ValueError("Preload example must match this motion, platform and contact layout")
        samples = data["samples"]
        loads = samples[samples[:, 0] >= samples[-1, 0] - 0.20, -9:-7]
        if not np.isfinite(loads).all() or np.any(loads < 3.0):
            raise ValueError("Expected two loaded forearms in the final 0.20 s")
        return np.median(loads, axis=0)


class GuidedKneeTargets:
    def __init__(self, env, example):
        self.env = env
        load = example_arm_loads(example, str(env.reference_np["source_sha256"]),
                                 env.reference_manifest["platform_rise_m"])
        self.load = torch.as_tensor(load, dtype=torch.float32, device=env.device)
        self.arms = []
        names = env.reference_np["joint_names"].tolist()
        for side, body in zip(("l", "r"), env.site_ids[2:4]):
            cols = [i for i, name in enumerate(names) if name.startswith(side + "_")
                    and ("shoulder" in name or "elbow" in name)]
            joints = [env.joint_ids[j] for j in cols]
            self.arms.append((body, cols, joints, [j + 6 for j in joints]))
        env.task_report.update(action_prior="v4_joint_residual_with_arm_preload",
                               preload_example=str(Path(example).resolve()), preload_arm_N=load.tolist())

    def desired(self, residual):
        e = self.env
        return self.preloaded_targets(
            residual, e.robot.root_physx_view.get_jacobians(), e.site_positions()
        )

    def preloaded_targets(self, residual, jacobians, sites):
        e = self.env
        frame = (e.frames() + 1).clamp(max=e.segment_end)
        target = e.ref["joint_pos"][frame] + 0.5 * residual.clamp(-2.0, 2.0)
        data = e.robot.data
        # Taper the arm torque only after consecutive measured right-knee loads.
        factor = (1.0 - (e.knee_contacts.knee_load_steps - 1).clamp_min(0) * e.step_dt / 0.12).clamp(0, 1)
        for side, (body, cols, joints, dofs) in enumerate(self.arms):
            offset = sites[:, 2 + side] - data.body_com_pos_w[:, body]
            moment = torch.stack((offset[:, 1], -offset[:, 0], torch.zeros_like(factor)), dim=1)
            jac = jacobians[:, body, :, dofs]
            vertical = jac[:, 2] + (jac[:, 3:] * moment[:, :, None]).sum(1)
            target[:, cols] -= (self.load[side] * factor[:, None] * vertical
                                / data.joint_stiffness[:, joints].clamp_min(1.0))
        target[:, e.wrist_joint] = e.fixed_wrist_target
        return target
