"""Shared local paths. Importing this module never creates or changes files."""

import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def configured_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


WORKSPACE_ROOT = configured_path("ELF3_WORKSPACE_ROOT", Path.home())
ROOT = configured_path(
    "ELF3_DATA_ROOT", WORKSPACE_ROOT / "humanoid_datasheet/datasets/elf3_climbing"
)
SOURCE = ROOT / "source"
OUTPUT = ROOT / "elf3"
PROJECT = configured_path("ELF3_TIENKUNG_ROOT", WORKSPACE_ROOT / "TienKung-Lab")
GMR = configured_path("ELF3_GMR_ROOT", WORKSPACE_ROOT / "GMR")
ASSET = PROJECT / "legged_lab/assets/elf3_lite"
URDF = ASSET / "urdf/elf3.urdf"
MJCF = ASSET / "xml/elf3.xml"
GRIPPER_CONFIG = TOOLS_DIR / "gripper_envelope.json"
DEFAULT_MOTION = "elf3_full_elbow_climb_demo_v7_candidate"
