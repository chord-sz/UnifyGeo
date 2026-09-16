import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from unifygeo.models import build_model


CHECKPOINTS = {
    "same-area": (
        Path(__file__).resolve().parents[1] / "checkpoints" / "unifygeo_vigor_same_area.pth",
        "6443775bd39a84eb4dca04ed5db206be57a8a8fbac6e36a6c045da00a60025ff",
    ),
    "cross-area": (
        Path(__file__).resolve().parents[1] / "checkpoints" / "unifygeo_vigor_cross_area.pth",
        "71175956671df210c76a830e4962730188bf0148938e597ef2641945d75cd79d",
    ),
}


def config():
    return SimpleNamespace(
        model="convnext_tiny.fb_in22k_ft_in1k_384",
        backbone_norm=None,
        grd_aggregator="soa_gem",
        sat_aggregator="soa_gem",
        aggregator_norm="LN",
        enc_dims=768,
        decoder_norm="BN",
        decoder_match_norm=True,
        train_rerank=False,
        aux_loss=True,
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointTests(unittest.TestCase):
    def test_released_checkpoints_load_strictly(self):
        available = [(name, value) for name, value in CHECKPOINTS.items() if value[0].is_file()]
        if not available:
            self.skipTest("Released checkpoints have not been downloaded.")
        for protocol, (path, expected_hash) in available:
            with self.subTest(protocol=protocol):
                self.assertEqual(sha256(path), expected_hash)
                state_dict = torch.load(path, map_location="cpu")
                self.assertEqual(len(state_dict), 457)
                self.assertEqual(sum(tensor.numel() for tensor in state_dict.values()), 57_700_330)
                model = build_model(config())
                model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    unittest.main()
