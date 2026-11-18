"""Supervise Isaac's process exit; missing results or error.json mean failure."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def result_code(returncode, output):
    if returncode:
        return returncode
    if output is None:
        return 0
    if (output / "error.json").exists() or not (output / "completion.json").is_file():
        return 1
    result = json.loads((output / "completion.json").read_text())
    return 0 if result.get("status") == "COMPLETED" else 1


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    child = subprocess.run([sys.executable, "-u", "-m", "elf3_climbing.cli", *sys.argv[1:]])
    output = args.output.expanduser().resolve() if args.output is not None else None
    code = result_code(child.returncode, output)
    if code and output is not None:
        print(
            f"Run failed or did not finish. Inspect {output} and the process log.", file=sys.stderr
        )
    sys.exit(code)
