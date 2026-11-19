"""Supervised entry point for the isolated TienKung-Lab walking extension."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    result = subprocess.run([sys.executable, "-u", "-m", "elf3_climbing.walk_cli", *sys.argv[1:]])
    completion = args.output.expanduser().resolve() / "completion.json"
    ok = (
        completion.is_file()
        and not (completion.parent / "error.json").exists()
        and json.loads(completion.read_text()).get("status") == "COMPLETED"
    )
    if result.returncode or not ok:
        print(
            f"Walking run failed or was interrupted; inspect {completion.parent}", file=sys.stderr
        )
    sys.exit(result.returncode or (0 if ok else 1))
