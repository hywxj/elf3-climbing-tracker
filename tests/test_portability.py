import json
from pathlib import Path
import shutil
import tempfile
import unittest

from elf3_climbing.launch import result_code
from elf3_climbing.settings import load_settings
from elf3_climbing.cli import parser
from elf3_climbing.paths import DEFAULT_PACK
from elf3_climbing.reference import read_pack


class ConfigurationTests(unittest.TestCase):
    def test_cli_overrides_profile(self):
        args = parser({"run": {"num_envs": 128, "segment": "flip"}}).parse_args(
            ["--num-envs", "2", "--output", "/tmp/unused"]
        )
        self.assertEqual(args.num_envs, 2)
        self.assertEqual(args.segment, "flip")

    def test_unknown_settings_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "bad.toml"
            p.write_text("[run]\nnum_env = 100\n")
            with self.assertRaisesRegex(ValueError, "Unknown run"):
                load_settings(p)

    def test_missing_result_and_error_override_zero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.assertEqual(result_code(0, p), 1)
            (p / "completion.json").write_text(json.dumps({"status": "COMPLETED"}))
            self.assertEqual(result_code(0, p), 0)
            (p / "error.json").write_text("{}")
            self.assertEqual(result_code(0, p), 1)
            self.assertEqual(result_code(7, p), 7)

    @unittest.skipUnless(DEFAULT_PACK.is_file(), "Runtime bundle required")
    def test_reference_survives_directory_relocation_and_detects_changed_source(self):
        meta = json.loads(DEFAULT_PACK.with_name("manifest.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "relocated/reference"
            base.mkdir(parents=True)
            for name in [*meta["hashes"], meta["source"]]:
                old = DEFAULT_PACK.parent / name
                new = base / name
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, new)
            shutil.copy2(DEFAULT_PACK, base / "reference.npz")
            shutil.copy2(DEFAULT_PACK.with_name("manifest.json"), base / "manifest.json")
            data, resolved = read_pack(base / "reference.npz")
            self.assertEqual(len(data["timestamps"]), 796)
            self.assertTrue(Path(resolved["source"]).is_relative_to(folder))
            Path(resolved["source"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Source motion hash mismatch"):
                read_pack(base / "reference.npz")


if __name__ == "__main__":
    unittest.main()
