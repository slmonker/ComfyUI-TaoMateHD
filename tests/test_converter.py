from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.convert_taomate_lora import convert_adapter  # noqa: E402


class ConverterTests(unittest.TestCase):
    def test_conversion_renames_keys_and_adds_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "adapter"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"rank": 4, "alpha": 4.0}), encoding="utf-8"
            )
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_a": torch.randn(4, 6),
                    "blocks.0.attn.qkv_proj.lora_b": torch.randn(8, 4),
                },
                str(source / "adapter_model.safetensors"),
            )
            model = root / "model.safetensors"
            save_file(
                {"blocks.0.attn.qkv_proj.weight": torch.randn(8, 6)},
                str(model),
            )
            output = root / "converted.safetensors"
            result = convert_adapter(
                source=source,
                output=output,
                dtype="fp32",
                model=model,
            )
            self.assertEqual(result["module_count"], 1)
            self.assertEqual(result["tensor_count"], 3)
            with safe_open(str(output), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                self.assertEqual(
                    keys,
                    {
                        "blocks.0.attn.qkv_proj.lora_A",
                        "blocks.0.attn.qkv_proj.lora_B",
                        "blocks.0.attn.qkv_proj.alpha",
                    },
                )
                self.assertEqual(tuple(handle.get_tensor("blocks.0.attn.qkv_proj.lora_A").shape), (4, 6))
                self.assertEqual(tuple(handle.get_tensor("blocks.0.attn.qkv_proj.lora_B").shape), (8, 4))
                self.assertEqual(float(handle.get_tensor("blocks.0.attn.qkv_proj.alpha")), 4.0)


if __name__ == "__main__":
    unittest.main()
