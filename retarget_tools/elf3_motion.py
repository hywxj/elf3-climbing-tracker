"""Common entry point for the local ELF3 motion tools; no hardware commands."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import project_paths as paths

CATALOG = paths.TOOLS_DIR / "tool_catalog.json"


def main():
    catalog = json.loads(CATALOG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List tools by workflow role without importing them")
    commands.add_parser("paths", help="Show resolved data/model/GMR paths")
    view = commands.add_parser("view", help="Interactive kinematic replay; rear view by default")
    check = commands.add_parser("check", help="Read-only NPZ, joint mapping and endpoint FK check")
    render = commands.add_parser("render", help="Render synchronized rear/side/oblique videos")
    audit = commands.add_parser("audit", help="Run the current ordered-recovery audit")
    for sub in [view, check, render, audit]:
        sub.add_argument(
            "--motion",
            default=paths.DEFAULT_MOTION,
            help="NPZ path or local motion name; defaults to v7 candidate",
        )
    view.add_argument("--view", choices=["default", "rear", "side", "rear-oblique"], default="rear")
    view.add_argument("--speed", type=float, default=1.0)
    view.add_argument("--once", action="store_true")
    render.add_argument("--output", type=Path, required=True, help="New output directory")
    render.add_argument("--size", choices=[640, 960], type=int, default=960)
    audit.add_argument("--work", type=Path, required=True, help="New audit directory")
    run = commands.add_parser("run", help="Run a catalogued script with its original arguments")
    run.add_argument(
        "script", choices=sorted(k for k, v in catalog["tools"].items() if v.get("cli"))
    )
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "list":
        print(f"当前默认演示：{paths.DEFAULT_MOTION}（候选轨迹）")
        print("MuJoCo v4 攀爬参考轨迹", paths.OUTPUT / catalog["mujoco_v4_reference"])
        for group in catalog["groups"]:
            print("\n" + group)
            for name, info in catalog["tools"].items():
                if info["group"] == group:
                    print(f"  {name:36} {info['purpose']}")
        return 0
    if args.command == "paths":
        print("Python:", sys.executable)
        for name in [
            "TOOLS_DIR",
            "ROOT",
            "SOURCE",
            "OUTPUT",
            "PROJECT",
            "GMR",
            "URDF",
            "MJCF",
            "GRIPPER_CONFIG",
        ]:
            value = getattr(paths, name)
            print(f"{name:16} {'OK' if value.exists() else 'MISSING':7} {value}")
        return 0
    if args.command == "run":
        rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        command = [sys.executable, str(paths.TOOLS_DIR / args.script), *rest]
    else:
        motion = Path(args.motion).expanduser()
        resolved = (
            motion.resolve()
            if motion.is_file()
            else paths.OUTPUT
            / (args.motion if args.motion.endswith(".npz") else args.motion + ".npz")
        )
        if not resolved.is_file():
            parser.error(f"Motion not found: {resolved}")
        base = ["--motion", str(resolved)]
        if args.command == "view":
            if args.speed <= 0:
                parser.error("--speed must be positive")
            script = "view_motion.py"
            rest = (
                base
                + ["--view", args.view, "--speed", str(args.speed)]
                + (["--once"] if args.once else [])
            )
        elif args.command == "check":
            script = "view_motion.py"
            rest = base + ["--check-only"]
        elif args.command == "render":
            script = "inspect_multiview_motion.py"
            output = args.output.expanduser().resolve()
            if output.exists():
                parser.error(f"Choose a new output directory: {output}")
            rest = base + ["--output", str(output), "--size", str(args.size)]
        else:
            script = "audit_ordered_recovery.py"
            output = args.work.expanduser().resolve()
            if output.exists():
                parser.error(f"Choose a new audit directory: {output}")
            rest = base + ["--work", str(output)]
        command = [sys.executable, str(paths.TOOLS_DIR / script), *rest]
    # Preserve the caller's working directory for legacy relative output paths.
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
