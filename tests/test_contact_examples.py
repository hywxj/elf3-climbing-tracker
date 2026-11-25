import unittest
from types import SimpleNamespace

import torch

from elf3_climbing.contact_examples import student_contact_state
from elf3_climbing.contact_dagger import mixed_command, rollin_schedule, wait_for_walking_position


class ContactExamplesTests(unittest.TestCase):
    def test_student_target_uses_actual_previous_drive_and_includes_full_student_rollins(self):
        previous = torch.tensor([[0.6], [0.6], [0.6]])
        teacher = torch.tensor([[0.7], [0.7], [0.7]])
        student = torch.full_like(teacher, -1.2)
        limits = torch.tensor([[[-1.0, 1.0]]])
        result = mixed_command(
            previous,
            teacher,
            student,
            torch.zeros_like(student),
            limits,
            torch.tensor([5.0]),
            0.02,
            torch.tensor([0.0, 0.7, 1.0]),
        )
        torch.testing.assert_close(result[:, 0], torch.tensor([0.7, 0.56, 0.5]))
        env = SimpleNamespace(num_envs=9, device="cpu", targets=previous, step_dt=0.02)
        torch.testing.assert_close(
            rollin_schedule(env, 0, True), torch.tensor([0.70, 0.85, 1.0, 0, 0, 0, 0, 0, 0])
        )
        self.assertTrue((rollin_schedule(env, 180, True) >= 0.70).all())

    def test_lagging_robot_waits_for_actual_dock_position(self):
        env = SimpleNamespace(
            course_time_s=torch.tensor([3.64, 3.64]),
            course_step=torch.tensor([182, 182]),
            step_dt=0.02,
            course=SimpleNamespace(poses=torch.tensor([[0.5, 0.0, 0.0]])),
            walk_dock_step=0,
            advance_direction=torch.tensor([1.0, 0.0]),
            brace_settings=torch.zeros(2, 4),
            ground_stage=torch.zeros(2, dtype=torch.long),
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor([[0.22, 0.0, 1.0], [0.45, 0.0, 1.0]])
                )
            ),
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
        )
        self.assertEqual(wait_for_walking_position(env).tolist(), [True, False])
        self.assertLess(env.course_step[0], env.course_step[1])

    def test_filtered_surfaces_force_centers_and_distinct_wrist_markers(self):
        origin = torch.tensor([[4.0, 3.0, 1.0]])
        local = torch.zeros(1, 8, 3)
        local[:, :, 2] = 0.7
        local[:, 7, 0] = 0.3
        hands = torch.tensor([[[0.2, 0.1, 0.7], [0.2, -0.1, 0.7]]])
        contacts = SimpleNamespace(
            arm_names=["la", "ra"],
            foot_names=["lf", "rf"],
            knee_names=["lk", "rk"],
            lower=0,
            upper=1,
            floor_heights=origin[:, 2],
            top_heights=origin[:, 2] + 0.7,
        )
        # Per-filter contact point tuples are (local x, local z, normal z, force N).
        rows = {
            "lf": [[(0.05, 0, 1, 60)], []],
            "rf": [[], [(0.2, 0.7, 1, 30)]],
            "ra": [[], [(0, 0.7, 0.2, 90)]],  # side scrape
            "l_wrist_y_link": [[], [(0.1, 0.7, 1, 10)]],
            "l_wrist_z_link": [[], [(0.3, 0.7, 1, 30)]],
            "rk": [[], [(0.27, 0.7, 1, 40), (0.52, 0.7, 1, 80)]],  # exclude shin
        }

        def sensor(name):
            lists = rows.get(name, [[], []])
            flat = lists[0] + lists[1]
            pts = torch.tensor([[x, 0.0, z] for x, z, _, _ in flat]).reshape(-1, 3) + origin
            norms = torch.tensor([[0.0, 0.0, nz] for _, _, nz, _ in flat]).reshape(-1, 3)
            f = torch.tensor([n for _, _, _, n in flat], dtype=torch.float32)
            details = (
                f,
                pts,
                norms,
                None,
                torch.tensor([[len(a) for a in lists]]),
                torch.tensor([[0, len(lists[0])]]),
            )
            return SimpleNamespace(
                contact_physx_view=SimpleNamespace(get_contact_data=lambda dt: details)
            )

        bodies = [
            *contacts.arm_names,
            *contacts.foot_names,
            *contacts.knee_names,
            *[f"{side}_wrist_{axis}_link" for side in "lr" for axis in "yz"],
        ]
        env = SimpleNamespace(
            knee_contacts=contacts,
            site_positions=lambda: local + origin[:, None],
            robot=SimpleNamespace(
                data=SimpleNamespace(body_link_pos_w=hands + origin[:, None]),
                find_bodies=lambda *args, **kwargs: ([0, 1], []),
            ),
            scene=SimpleNamespace(env_origins=origin),
            cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.005)),
            support_sensors={b: sensor(b) for b in bodies},
        )
        sites, forces, xyz = student_contact_state(env, with_points=True)
        torch.testing.assert_close(sites[:, 4:6], hands)
        torch.testing.assert_close(
            forces, torch.tensor([[60.0, 30.0, 0.0, 0.0, 40.0, 0.0, 0.0, 40.0]])
        )
        self.assertAlmostEqual(xyz[0, 4, 0].item(), 0.25, places=5)
        self.assertTrue(torch.isnan(xyz[0, 3]).all())
        _, online = student_contact_state(env)
        torch.testing.assert_close(online, forces)
