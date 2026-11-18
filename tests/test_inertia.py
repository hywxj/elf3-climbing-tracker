import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from elf3_climbing.inertia import combine
from elf3_climbing.gripper import apply, specification
from elf3_climbing.model import load, model_file
from elf3_climbing.paths import DEFAULT_SOURCE


class CompositeInertia(unittest.TestCase):
    def test_parallel_axis_shift(self):
        mass, center, tensor = combine(1, [0, 0, 0], np.eye(3) * 0.1, 1, [2, 0, 0], np.eye(3) * 0.2)
        self.assertEqual(mass, 2)
        np.testing.assert_allclose(center, [1, 0, 0])
        np.testing.assert_allclose(tensor, np.diag([0.3, 2.3, 2.3]))

    def test_rotated_component_has_products_of_inertia(self):
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        component = rotation @ np.diag([1.0, 2.0, 2.5]) @ rotation.T
        _, _, tensor = combine(1, [0, 0, 0], np.eye(3), 1, [0, 0, 0], component)
        np.testing.assert_allclose(tensor, np.eye(3) + component)
        self.assertNotEqual(tensor[0, 1], 0)

    def test_invalid_mass_rejected(self):
        with self.assertRaises(ValueError):
            combine(1, [0, 0, 0], np.eye(3), -0.5, [0, 0, 0], np.eye(3))


@unittest.skipUnless(DEFAULT_SOURCE.exists(), "Runtime required")
class LoadedGripper(unittest.TestCase):
    def test_zero_payload_preserves_inertials_and_collision_envelope(self):
        cfg = specification()
        self.assertEqual(cfg["mass_kg"], 0.0)
        self.assertEqual(cfg["physical_mass_kg"], 0.5)
        _, source = load(DEFAULT_SOURCE)
        with tempfile.TemporaryDirectory() as folder:
            baseline = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True, False)))
            model = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True, True)))
            np.testing.assert_array_equal(model.body_mass, baseline.body_mass)
            np.testing.assert_array_equal(model.body_ipos, baseline.body_ipos)
            np.testing.assert_array_equal(model.body_inertia, baseline.body_inertia)
            self.assertEqual(model.ngeom, baseline.ngeom + 1)
            self.assertEqual(model.geom(cfg["geom_name"]).contype[0], 1)

    def test_adds_half_kg_only_once_and_preserves_fk(self):
        cfg = specification()
        # Retain regression coverage for the later payload adaptation stage.
        cfg["mass_kg"] = 0.5
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_quat(cfg["mount_quaternion_wxyz"], scalar_first=True).as_matrix()
        squared = (np.array(cfg["dimensions_mm"]) / 1000) ** 2
        cfg["inertia_about_com_in_wrist_kg_m2"] = (
            rotation @ np.diag(0.5 / 12 * (squared.sum() - squared)) @ rotation.T
        ).tolist()
        _, source = load(DEFAULT_SOURCE)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("elf3_climbing.gripper.specification", return_value=cfg),
        ):
            baseline = mujoco.MjModel.from_xml_path(str(model_file(folder, source, True, False)))
            path = model_file(folder, source, True, True)
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertAlmostEqual(model.body_mass.sum() - baseline.body_mass.sum(), 0.5)
            wrist = model.body("l_wrist_z_link").id
            self.assertAlmostEqual(model.body_mass[wrist] - baseline.body_mass[wrist], 0.5)
            d0, d1 = mujoco.MjData(baseline), mujoco.MjData(model)
            mujoco.mj_forward(baseline, d0)
            mujoco.mj_forward(model, d1)
            np.testing.assert_allclose(d0.xpos, d1.xpos, atol=1e-12)
            np.testing.assert_allclose(d0.xquat, d1.xquat, atol=1e-12)
            self.assertTrue((model.body_inertia[wrist] > 0).all())
            root = ET.parse(path).getroot()
            with self.assertRaisesRegex(ValueError, "already applied"):
                apply(root)


if __name__ == "__main__":
    unittest.main()
