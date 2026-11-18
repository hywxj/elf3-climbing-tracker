import unittest
import numpy as np

from elf3_climbing.balance import ground_standing_reference
from elf3_climbing.paths import DEFAULT_PACK
from elf3_climbing.reference import read_pack


@unittest.skipUnless(DEFAULT_PACK.is_file(), "Runtime required")
class StandingReference(unittest.TestCase):
    def test_ground_pose_is_unflipped_stationary_and_does_not_modify_source(self):
        source, _ = read_pack(DEFAULT_PACK)
        original = {key: value.copy() for key, value in source.items()}
        pack, report = ground_standing_reference(source)
        for key in source:
            np.testing.assert_array_equal(source[key], original[key])
        np.testing.assert_array_equal(pack["joint_pos"][0], source["joint_pos"][0])
        for key in ("root_lin_vel", "root_ang_vel", "joint_vel"):
            self.assertFalse(pack[key][0].any())
        self.assertEqual(len(pack["terrain_box_centers_w"]), 1)
        self.assertAlmostEqual(report["wrist_target_rad"], 0)
        self.assertFalse(report["flip_required"])
        self.assertLess(report["ground_top_m"], 0.1)
        self.assertEqual(pack["contact_valid"][0].sum(), 2)
        np.testing.assert_array_equal(pack["contact_targets"][0, :2], pack["site_pos"][0, :2])

    def test_unsupported_or_flipped_start_rejected(self):
        source, _ = read_pack(DEFAULT_PACK)
        source["site_pos"][0, 0, 2] += 0.1
        with self.assertRaisesRegex(ValueError, "horizontal surface"):
            ground_standing_reference(source)
        source, _ = read_pack(DEFAULT_PACK)
        source["joint_pos"][0, source["joint_names"].tolist().index("l_wrist_y_joint")] = -1.309
        with self.assertRaisesRegex(ValueError, "unflipped"):
            ground_standing_reference(source)
