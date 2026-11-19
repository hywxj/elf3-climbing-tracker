"""Regression checks for wrist/style conflict and data isolation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from elf3_climbing.walk_data import mask_wrist, prepare_walk_manifest, WristMaskedPrior


class WalkPriorTests(unittest.TestCase):
    def test_wrist_cannot_change_discriminator_features(self):
        a = torch.randn(8, 70)
        b = a.clone()
        b[:, 12] = -1.309
        b[:, 41] = 1.5
        torch.testing.assert_close(mask_wrist(a), mask_wrist(b), rtol=0, atol=0)
        torch.testing.assert_close(b[:, 12], torch.full((8,), -1.309))
        other = [i for i in range(70) if i not in (12, 41)]
        torch.testing.assert_close(mask_wrist(b)[:, other], b[:, other], rtol=0, atol=0)
        b[:, 24] += 0.1
        self.assertFalse(torch.equal(mask_wrist(a), mask_wrist(b)))

    def test_every_expert_sampling_path_masks_both_transition_states(self):
        class Source:
            def sample_pairs(self, count):
                return torch.ones(count, 70), torch.ones(count, 70) * 2

        prior = WristMaskedPrior(Source())
        for pair in prior.feed_forward_generator(3, 4):
            for x in pair:
                self.assertEqual(x[:, [12, 41]].count_nonzero().item(), 0)
                self.assertGreater(x[:, 24].sum().item(), 0)
        self.assertEqual(prior.get_full_frame_batch(5)[:, [12, 41]].count_nonzero().item(), 0)

    def test_selection_preserves_split_hash_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, (category, split) in enumerate(
                [("walk_turn", "train"), ("stairs_up", "train"), ("walk_turn", "validation")]
            ):
                path = root / f"clip{i}.npz"
                np.savez(path, data=np.arange(5) + i)
                rows.append(
                    dict(
                        id=str(i),
                        output=path.name,
                        eligible=True,
                        category=category,
                        split=split,
                        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            names = ["unused"] * 29
            names[12] = "l_wrist_y_joint"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        schema_version="test",
                        sample_dt=0.02,
                        contract={"amp_joint_names": names},
                        clips=rows,
                    )
                )
            )
            before = manifest.read_bytes()
            output = prepare_walk_manifest(manifest, root / "selected")
            result = json.loads(output.read_text())
            self.assertEqual([r["id"] for r in result["clips"]], ["0", "2"])
            self.assertEqual(result["selection"]["validation"], 1)
            self.assertFalse((output.parent / "clip1.npz").exists())
            self.assertEqual(manifest.read_bytes(), before)
            for row in result["clips"]:
                self.assertEqual(
                    (output.parent / row["output"]).read_bytes(),
                    (root / row["output"]).read_bytes(),
                )
            (root / "clip0.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare_walk_manifest(manifest, root / "bad")
