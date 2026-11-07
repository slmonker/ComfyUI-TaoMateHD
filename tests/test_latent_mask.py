from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMFY_ROOT))

import comfy.nested_tensor
from latent_mask import masks_for_video_shape, normalize_noise_mask, restore_kept

nodes = importlib.import_module("custom_nodes.ComfyUI-TaoMateHD.nodes")


class LatentMaskTests(unittest.TestCase):
    def test_nested_av_mask_normalizes_solid_audio_mask(self):
        video = torch.zeros(1, 24, 2, 8, 12)
        audio = torch.zeros(1, 32, 2, 6)
        raw = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.zeros(1, 1, 64, 64))
        )

        masks = normalize_noise_mask(raw, video, audio)
        video_mask, audio_mask = masks.unbind()

        self.assertEqual(tuple(video_mask.shape), (1, 1, 2, 8, 12))
        self.assertEqual(tuple(audio_mask.shape), (1, 1, 1, 6))
        self.assertTrue(torch.equal(video_mask, torch.ones_like(video_mask)))
        self.assertTrue(torch.equal(audio_mask, torch.zeros_like(audio_mask)))

    def test_partial_audio_time_mask_is_preserved(self):
        video = torch.zeros(1, 24, 2, 8, 12)
        audio = torch.zeros(1, 32, 2, 6)
        audio_mask = torch.ones(1, 32, 6)
        audio_mask[..., :3] = 0
        masks = normalize_noise_mask(
            comfy.nested_tensor.NestedTensor((torch.ones_like(video), audio_mask)),
            video,
            audio,
        )
        _, normalized_audio_mask = masks.unbind()
        self.assertEqual(tuple(normalized_audio_mask.shape), (1, 1, 1, 6))
        self.assertTrue(torch.equal(normalized_audio_mask[..., :3], torch.zeros(1, 1, 1, 3)))
        self.assertTrue(torch.equal(normalized_audio_mask[..., 3:], torch.ones(1, 1, 1, 3)))

    def test_odd_video_latent_reports_resolution_alignment_hint(self):
        video = torch.zeros(1, 24, 2, 7, 8)
        audio = torch.zeros(1, 32, 2, 1)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        with self.assertRaisesRegex(ValueError, "multiple=32"):
            nodes._validate_h3_latent(latent)

    def test_video_mask_and_anchor_scale_to_low_stage(self):
        video = torch.zeros(1, 24, 2, 8, 12)
        audio = torch.zeros(1, 32, 2, 6)
        masks = normalize_noise_mask(
            comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video), torch.zeros(1, 1, 64, 64))
            ),
            video,
            audio,
        )
        low_video = torch.zeros(1, 24, 2, 4, 6)

        low_masks = masks_for_video_shape(masks, low_video)
        low_video_mask, low_audio_mask = low_masks.unbind()

        self.assertEqual(tuple(low_video_mask.shape), (1, 1, 2, 4, 6))
        self.assertEqual(tuple(low_audio_mask.shape), (1, 1, 1, 6))

    def test_exact_keep_regions_restore_anchor(self):
        outputs = (torch.ones(1, 1, 2, 2, 2), torch.ones(1, 1, 1, 4))
        anchors = (torch.zeros(1, 1, 2, 2, 2), torch.full((1, 1, 1, 4), 7.0))
        masks = (torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 4))
        restored_video, restored_audio = restore_kept(outputs, anchors, masks)
        self.assertTrue(torch.equal(restored_video, anchors[0]))
        self.assertTrue(torch.equal(restored_audio, anchors[1]))

    def test_stage_sampler_preserves_zero_mask_and_generates_one_mask(self):
        class FakeSampling:
            def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                return latent_image

        class FakeModel:
            def __init__(self):
                self.inner_model = self
                self.model_sampling = FakeSampling()

            def process_latent_in(self, latent):
                return latent

            def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
                return latent_image

        class FakeGuider:
            def __init__(self):
                self.inner_model = FakeModel()

            def __call__(self, x, sigma, **kwargs):
                return x

        x = torch.ones(1, 1, 8)
        anchor = torch.full_like(x, 7.0)
        mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        sampler = nodes._TaoMateStageSampler(
            start_from_latent=True, stage_name="test", latent_anchor=anchor
        )
        result = sampler.sample(
            FakeGuider(),
            torch.tensor([1.0, 0.0]),
            {},
            None,
            torch.zeros_like(x),
            latent_image=x,
            denoise_mask=mask,
        )
        expected = torch.tensor([[[7.0, 7.0, 7.0, 7.0, 1.0, 1.0, 1.0, 1.0]]])
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
