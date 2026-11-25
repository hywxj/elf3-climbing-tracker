"""Batched execution of the bundled ELF3 normal walking actor on the simulator device."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


GAIT_MODEL = (
    Path(__file__).resolve().parents[3]
    / "bxi_controller_ros2/src/bxi_example_py_elf3"
    / "mods/com.bxi.basic_actions/assets/model_normal.onnx"
)


class GroundLocomotion:
    def __init__(self, joint_names, device):
        import onnx
        from onnx.numpy_helper import to_array

        model = onnx.load(GAIT_MODEL)
        metadata = {item.key: item.value for item in model.metadata_props}
        if metadata["joint_names"].split(",") != list(joint_names):
            raise ValueError("Official gait outputs must match all 29 course joints in order")
        nodes = list(model.graph.node)
        if [node.op_type for node in nodes] != [
            "Sub",
            "Div",
            "Gemm",
            "Elu",
            "Gemm",
            "Elu",
            "Gemm",
            "Elu",
            "Gemm",
        ]:
            raise ValueError("Unrecognized official gait network architecture")
        weights = {
            v.name: torch.tensor(to_array(v), device=device) for v in model.graph.initializer
        }
        self.mean = weights[nodes[0].input[1]]
        self.std = weights[nodes[1].input[1]]
        self.layers = [
            (weights[n.input[1]], weights[n.input[2]]) for n in nodes if n.op_type == "Gemm"
        ]
        for dst, src in [
            ("default", "default_joint_pos"),
            ("scale", "action_scale"),
            ("kp", "joint_stiffness"),
            ("kd", "joint_damping"),
        ]:
            values = np.array(metadata[src].split(","), dtype=np.float32)
            if values.shape != (29,) or not np.isfinite(values).all():
                raise ValueError("Invalid official gait metadata: " + src)
            setattr(self, dst, torch.as_tensor(values, device=device))

    def actor(self, obs):
        x = (obs - self.mean) / self.std
        for w, b in self.layers[:-1]:
            x = F.elu(F.linear(x, w, b))
        return F.linear(x, *self.layers[-1])

    def targets(self, env, cmd, active):
        data, joint_ids = env.robot.data, env.joint_ids
        q, qd = data.joint_pos[:, joint_ids], data.joint_vel[:, joint_ids]
        centered_q = q - self.default
        adjusted_vel = qd.clone()
        # The official policy observes a wrist relative to the external flip command.
        # Keep the original motor controller responsible for the physical gripper.
        row = env.course_step
        centered_q[:, env.wrist_joint] -= env.course.poses[row, 7 + env.wrist_joint]
        adjusted_vel[:, env.wrist_joint] -= env.course.velocities[row, 6 + env.wrist_joint]
        obs = torch.cat(
            (
                data.root_ang_vel_b,
                data.projected_gravity_b,
                centered_q,
                adjusted_vel,
                env.gait_actions,
                cmd,
            ),
            1,
        )
        actions = self.actor(obs)
        env.gait_actions.copy_(torch.where(active[:, None], actions, env.gait_actions))
        desired = self.default + self.scale * actions
        # Express vendor stiffness and damping through the existing Isaac position drives.
        stiffness = data.joint_stiffness[:, joint_ids]
        damping = data.joint_damping[:, joint_ids]
        return q + (self.kp * (desired - q) + (damping - self.kd) * qd) / stiffness.clamp_min(1)
