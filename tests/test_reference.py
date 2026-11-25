"""Offline contract checks; no simulator startup, training, or source writes."""

import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from elf3_climbing.reference import make_contact_commands, prepare, read_pack
from elf3_climbing.paths import DEFAULT_PACK, URDF, DEFAULT_SOURCE, RUNTIME


class ContactEvents(unittest.TestCase):
    def test_no_contacts_have_no_commands(self):
        t = np.arange(10) * 0.1
        p = np.ones((10, 1, 3))
        f = np.zeros((10, 1), bool)
        command, valid = make_contact_commands(t, p, f)
        self.assertFalse(valid.any())
        self.assertFalse(command.any())

    def test_horizon_and_event_anchor(self):
        t = np.arange(10) * 0.1
        p = np.arange(30).reshape(10, 1, 3).astype(float)
        f = np.zeros((10, 1), bool)
        f[4:7] = True
        c, v = make_contact_commands(t, p, f, horizon=0.2)
        np.testing.assert_array_equal(np.flatnonzero(v[:, 0]), [2, 3, 4, 5, 6])
        np.testing.assert_array_equal(c[2:7, 0], np.repeat(p[4], 5, axis=0))

    def test_nearest_event_wins_without_wraparound(self):
        t = np.arange(10) * 0.1
        p = np.arange(30).reshape(10, 1, 3).astype(float)
        f = np.zeros((10, 1), bool)
        f[3:5] = True
        f[7:9] = True
        c, v = make_contact_commands(t, p, f, 1.5)
        np.testing.assert_array_equal(c[0, 0], p[3, 0])
        np.testing.assert_array_equal(c[5, 0], p[7, 0])
        self.assertFalse(v[9, 0])


@unittest.skipUnless(DEFAULT_PACK.is_file(), "Runtime bundle required for model/data checks")
class LocalReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.meta = read_pack(DEFAULT_PACK)

    def test_frames_and_derivatives(self):
        a = self.a
        self.assertEqual(a["joint_pos"].shape, (796, 29))
        np.testing.assert_allclose(np.diff(a["timestamps"]), 0.02, atol=1e-12)
        np.testing.assert_allclose(
            a["joint_vel"], np.gradient(a["joint_pos"], 0.02, axis=0, edge_order=2)
        )
        np.testing.assert_allclose(np.linalg.norm(a["root_quat"], axis=1), 1.0, atol=1e-12)

    def test_segment_boundaries(self):
        a = self.a
        for name, labels in [("flip", ["READY", "GRIPPER_FLIP"]), ("stand", ["STAND", "FINISHED"])]:
            lo, hi = json.loads(str(a["segments_json"]))[name]
            self.assertTrue(np.isin(a["phase"][lo : hi + 1], labels).all())
            self.assertGreater(hi - lo, 50)

    def test_joint_order_and_limits(self):
        xml = ET.parse(URDF).getroot()
        for i, n in enumerate(self.a["joint_names"]):
            limit = xml.find(f'joint[@name="{n}"]/limit')
            self.assertTrue(
                (self.a["joint_pos"][:, i] >= float(limit.get("lower")) - 1e-6).all(), n
            )
            self.assertTrue(
                (self.a["joint_pos"][:, i] <= float(limit.get("upper")) + 1e-6).all(), n
            )
            self.assertLessEqual(
                abs(self.a["joint_vel"][:, i]).max(), float(limit.get("velocity")) + 1e-6, n
            )

    def test_resampling_preserves_shared_times(self):
        with np.load(DEFAULT_SOURCE) as source:
            # 30 Hz and 50 Hz share every 0.1 s. Compare rotations, not quaternion signs.
            np.testing.assert_allclose(
                self.a["joint_pos"][::5], source["joint_pos"][::3], atol=1e-12
            )
            np.testing.assert_allclose(
                Rotation.from_quat(self.a["root_quat"][::5], scalar_first=True).as_matrix(),
                Rotation.from_quat(source["base_quat_wxyz"][::3], scalar_first=True).as_matrix(),
                atol=1e-12,
            )
            ix = (
                np.searchsorted(source["timestamps"], self.a["timestamps"] + 1e-10, side="right")
                - 1
            )
            np.testing.assert_array_equal(self.a["phase"], source["phase"][ix])

    def test_tampered_pack_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reference.npz"
            p.write_bytes(DEFAULT_PACK.read_bytes() + b"tampered")
            p.with_name("manifest.json").write_text(
                DEFAULT_PACK.with_name("manifest.json").read_text()
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_pack(p)

    def test_not_promoted_to_training_acceptance(self):
        self.assertFalse(self.meta["full_motion_training_ready"])
        self.assertIn("NOT measured forces", self.meta["labels"])
        self.assertEqual(self.a["contact_mask"].dtype, np.dtype(bool))


V4_SOURCE = RUNTIME.parent / "elf3/elf3_full_elbow_climb_demo_v4_candidate.npz"


@unittest.skipUnless(V4_SOURCE.is_file(), "Local v4 motion required")
class ForearmHeightAdaptation(unittest.TestCase):
    def test_forearm_faces_and_ground_support_survive_height_change(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            path = Path(folder)
            prepare(V4_SOURCE, path / "native")
            prepare(V4_SOURCE, path / "raised", 0.7)
            native, native_meta = read_pack(path / "native/reference.npz")
            raised, meta = read_pack(path / "raised/reference.npz")
        self.assertEqual(meta["source_sha256"], native_meta["source_sha256"])
        self.assertAlmostEqual(meta["platform_rise_m"], 0.7)
        np.testing.assert_array_equal(raised["phase"], native["phase"])
        flags = native["contact_mask"]
        np.testing.assert_array_equal(raised["contact_mask"], flags)
        dz = 0.7 - native_meta["platform_rise_m"]
        for arm in (2, 3):
            support = flags[:, arm] | flags[:, arm + 2]
            z_delta = raised["site_pos"][support, arm, 2] - native["site_pos"][support, arm, 2]
            np.testing.assert_allclose(z_delta, dz, atol=5e-5)
            before = Rotation.from_quat(native["site_quat"][support, arm], scalar_first=True)
            after = Rotation.from_quat(raised["site_quat"][support, arm], scalar_first=True)
            self.assertLess((after * before.inv()).magnitude().max(), 1e-4)
            slide = raised["site_pos"][support, arm, :2] - native["site_pos"][support, arm, :2]
            self.assertLess(np.linalg.norm(slide, axis=1).max(), 0.025)
        first_arm = np.flatnonzero(flags[:, 2:6].any(1))[0]
        release = np.flatnonzero((np.arange(len(flags)) > first_arm) & ~flags[:, :2].any(1))[0]
        for key in ("site_pos", "site_quat"):
            np.testing.assert_array_equal(raised[key][:release, :2], native[key][:release, :2])
        limits = ET.parse(URDF).getroot()
        for j, name in enumerate(raised["joint_names"]):
            limit = limits.find(f'joint[@name="{name}"]/limit')
            self.assertLessEqual(abs(raised["joint_vel"][:, j]).max(), float(limit.get("velocity")))


if __name__ == "__main__":
    unittest.main()
