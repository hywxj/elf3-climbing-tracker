"""Portable runtime layout. Importing never writes or creates directories."""

import os
from pathlib import Path

RUNTIME = (
    Path(
        os.environ.get(
            "ELF3_RUNTIME_ROOT",
            Path.home() / "humanoid_datasheet/datasets/elf3_climbing/runtime_v1",
        )
    )
    .expanduser()
    .resolve()
)
ASSET = RUNTIME / "assets/elf3_lite"
URDF = ASSET / "urdf/elf3.urdf"
MJCF = ASSET / "xml/elf3.xml"
USD = ASSET / "usd/elf3.usd"
GRIPPER_CONFIG = RUNTIME / "gripper_envelope.json"
DEFAULT_SOURCE = RUNTIME / "motion/source.npz"
DEFAULT_PACK = RUNTIME / "reference/reference.npz"
OUTPUT = RUNTIME / "runs"
