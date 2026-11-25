"""Collect official handoffs, train a contact tracker, or evaluate a checkpoint."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "bxi_controller_ros2" / "tools")]

import elf3_approach as approach
from elf3_climbing.reference import digest


def collect(path, yaw, height=0.70):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    gripper = approach.specification()
    model = approach.build_model("official", True, gripper, height)
    onnx = approach.PACKAGE / "mods/com.bxi.basic_actions/assets/model_normal.onnx"
    policy = approach.POLICIES["model_normal"](str(onnx), backend="onnxruntime")
    options = SimpleNamespace(
        robot="official",
        policy="model_normal",
        mode="forearm",
        initial_yaw_deg=yaw,
        view=False,
        video=None,
        wrist_feedback="relative",
    )
    snapshot = {}

    def record(stage, model, data):
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, state_type))
        mujoco.mj_getState(model, data, state, state_type)
        snapshot[stage] = state

    try:
        report = approach.run(options, model, policy, gripper, on_handoff=record)
    finally:
        policy.close()
    if not report["screening_pass"] or set(snapshot) != {"docked", "supported"}:
        raise RuntimeError(json.dumps(report))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        np.savez_compressed(
            out,
            **snapshot,
            yaw=yaw,
            platform_height_m=height,
            report=json.dumps(report),
            model_signature=model_signature(model),
            policy_sha256=digest(onnx),
        )
    print(path, "four-point hold (s)", report["forearm"]["final_continuous_four_point_hold_s"])


def model_signature(model):
    import hashlib

    h = hashlib.sha256()
    h.update(model.names)
    for value in (
        model.body_pos,
        model.body_mass,
        model.body_inertia,
        model.geom_pos,
        model.geom_quat,
        model.geom_size,
        model.jnt_range,
        model.actuator_ctrlrange,
    ):
        h.update(value.tobytes())
    h.update(str(model.opt.timestep).encode())
    return h.hexdigest()


def main():
    from elf3_climbing.platform_task import DEFAULT_MOTION

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--yaw", type=float, required=True)
    c.add_argument("--platform-height", type=float, default=0.70)
    c.add_argument("--output", type=Path, required=True)
    for name in ("train", "evaluate", "preload", "train-knee", "evaluate-knee"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--handoffs", nargs="+", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
        cmd.add_argument("--seed", type=int, default=42)
        if name != "preload":
            cmd.add_argument("--device", default="cpu", help="PyTorch PPO device, e.g. cpu or cuda:0. MuJoCo physics runs on CPU.")
        if name in ("train", "train-knee"):
            cmd.add_argument("--steps", type=int, default=8192)
            cmd.add_argument("--resume", type=Path)
        elif name in ("evaluate", "evaluate-knee"):
            cmd.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.output, args.yaw, args.platform_height)
        return
    if args.cmd == "preload":
        from elf3_climbing.grounded_preload import export
        export(args, approach, model_signature)
        return
    import torch
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if device.type not in ("cpu", "cuda"):
        parser.error("--device supports cpu or cuda[:index]")
    if device.type == "cuda" and (
        not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        parser.error(f"CUDA device {device} is unavailable in this Python environment")
    if args.cmd in ("train-knee", "evaluate-knee"):
        from elf3_climbing.knee_entry import experiment
        experiment(args, approach, model_signature)
        return
    from elf3_climbing.platform_task import experiment

    experiment(args, approach, model_signature)


if __name__ == "__main__":
    main()
