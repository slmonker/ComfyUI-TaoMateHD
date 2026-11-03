from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from progressive import (  # noqa: E402
    ProgressiveCanvas,
    coupled_video_noise,
    derive_low_canvas,
    lift_video_latent,
    progressive_stage_sigmas,
)


class ProgressiveTests(unittest.TestCase):
    def test_teacher_schedule(self):
        stages = progressive_stage_sigmas()
        values = (stages[0][0], stages[0][1], stages[1][1], stages[2][1])
        self.assertAlmostEqual(values[0], 1.0, places=6)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(values[3], 0.0)

    def test_derive_low_canvas_portrait(self):
        high = ProgressiveCanvas(width=768, height=1376)
        low = derive_low_canvas(high, 480.0 / 768.0)
        self.assertEqual((low.width, low.height), (480, 864))

    def test_coupled_noise_and_lift_shapes(self):
        low = ProgressiveCanvas(width=256, height=256)
        high = ProgressiveCanvas(width=512, height=512)
        low_noise, high_noise = coupled_video_noise(
            low_canvas=low,
            high_canvas=high,
            temporal_length=2,
            seed=1234,
        )
        self.assertEqual(tuple(low_noise.shape), (1, 24, 2, 16, 16))
        self.assertEqual(tuple(high_noise.shape), (1, 24, 2, 32, 32))
        low_state = torch.randn_like(low_noise)
        lifted = lift_video_latent(
            low_video=low_state,
            low_noise=low_noise,
            high_noise=high_noise,
            sigma=0.4,
            low_canvas=low,
            high_canvas=high,
        )
        self.assertEqual(tuple(lifted.shape), (1, 24, 2, 32, 32))
        self.assertTrue(torch.isfinite(lifted).all())


if __name__ == "__main__":
    unittest.main()
