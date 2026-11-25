"""Train or evaluate whole-platform ascent from a frozen knee-entry policy."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elf3_climbing.launch import result_code


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", type=Path)
    args, _ = p.parse_known_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "third_party/rsl_rl"), env.get("PYTHONPATH", "")]
    )
    cmd = [sys.executable, "-u", "-m", "elf3_climbing.mount_runner", *sys.argv[1:]]
    child = subprocess.run(cmd, env=env)
    return (
        result_code(child.returncode, args.output)
        if "--help" not in sys.argv and "-h" not in sys.argv
        else child.returncode
    )


if __name__ == "__main__":
    sys.exit(main())
