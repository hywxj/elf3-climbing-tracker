import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from elf3_climbing.ground_support import GroundSupport


class GroundSupportTests(unittest.TestCase):
    def test_table_filter_and_sum_across_opposing_contacts(self):
        support = GroundSupport.__new__(GroundSupport)
        # One existing sensor with the upper platform in column 1 and one
        # additional sensor with that same platform as its only filter.
        hip = torch.zeros(2, 1, 2, 3)
        hip[:, 0, 0, 2] = 300
        hip[:, 0, 1, 0] = 12
        shoulder = torch.zeros(2, 1, 1, 3)
        shoulder[:, 0, 0, 0] = -9
        support.views = [
            (SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix)), col)
            for matrix, col in ((hip, 1), (shoulder, 0))
        ]
        torch.testing.assert_close(
            support.force(torch.tensor([True, False])), torch.tensor([21.0, 0])
        )

    def test_no_extra_queries_without_a_standing_candidate(self):
        class Sensor:
            pass

        support = GroundSupport.__new__(GroundSupport)
        support.views = [(Sensor(), 0)]
        with patch.object(Sensor, "data", create=True, new_callable=PropertyMock) as read:
            torch.testing.assert_close(support.force(torch.tensor([False])), torch.tensor([0.0]))
            read.assert_not_called()
