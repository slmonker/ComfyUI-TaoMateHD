from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(COMFY_ROOT))

nodes = importlib.import_module("custom_nodes.ComfyUI-TaoMateHD.nodes")


class ConditioningValidationTests(unittest.TestCase):
    def test_t2va_conditioning_is_accepted(self):
        nodes._validate_conditioning([[None, {}]], name="positive")

    def test_ref2va_conditioning_is_accepted(self):
        refs = [{"kind": "image", "latent_h": 32, "latent_w": 32, "latent": None}]
        nodes._validate_conditioning([[None, {"minimax_refs": refs}]], name="positive")

    def test_fl2va_keyframes_are_rejected_with_specific_error(self):
        keyframes = [{"resolved_frame_index": 0, "latent": None}]
        with self.assertRaisesRegex(NotImplementedError, "Ref2VA.*FL2VA"):
            nodes._validate_conditioning([[None, {"minimax_keyframes": keyframes}]], name="positive")


if __name__ == "__main__":
    unittest.main()
