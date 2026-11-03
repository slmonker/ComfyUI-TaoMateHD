from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _read_adapter(source: Path):
    if source.is_file():
        source = source.parent
    config_path = source / "config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing adapter weights: {weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config["rank"])
    alpha = float(config["alpha"])
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"invalid LoRA rank/alpha: rank={rank!r} alpha={alpha!r}")
    return source, weights_path, rank, alpha


def _read_model_shapes(model_path: Path):
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        return {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}


def _find_model_weight(shapes, base: str):
    for prefix in ("", "diffusion_model."):
        key = f"{prefix}{base}.weight"
        if key in shapes:
            return key, shapes[key]
    return None, None


def convert_adapter(
    *,
    source: Path,
    output: Path,
    dtype: str = "fp32",
    model: Path | None = None,
    overwrite: bool = False,
) -> dict:
    source_dir, weights_path, rank, alpha = _read_adapter(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if dtype not in {"fp32", "bf16"}:
        raise ValueError(f"dtype must be fp32 or bf16, got {dtype!r}")

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
        source_keys = list(handle.keys())
        source_tensors = {key: handle.get_tensor(key) for key in source_keys}

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    unexpected = []
    for key, tensor in source_tensors.items():
        if key.endswith(".lora_a"):
            base = key[: -len(".lora_a")]
            pairs.setdefault(base, {})["a"] = tensor
        elif key.endswith(".lora_b"):
            base = key[: -len(".lora_b")]
            pairs.setdefault(base, {})["b"] = tensor
        else:
            unexpected.append(key)

    if unexpected:
        raise ValueError(
            "adapter contains unexpected tensor keys: " + ", ".join(unexpected[:8])
        )
    if not pairs:
        raise ValueError("adapter contains no lora_a/lora_b tensor pairs")

    incomplete = [base for base, pair in pairs.items() if set(pair) != {"a", "b"}]
    if incomplete:
        raise ValueError("adapter has incomplete LoRA pairs: " + ", ".join(incomplete[:8]))

    converted: dict[str, torch.Tensor] = {}
    model_shapes = _read_model_shapes(model) if model is not None else None
    shape_errors = []
    for base, pair in sorted(pairs.items()):
        a = pair["a"]
        b = pair["b"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{base}: expected rank-2 LoRA tensors, got {a.shape} and {b.shape}")
        if int(a.shape[0]) != rank or int(b.shape[1]) != rank:
            raise ValueError(
                f"{base}: adapter rank mismatch: a={tuple(a.shape)} b={tuple(b.shape)} rank={rank}"
            )
        if model_shapes is not None:
            model_key, model_shape = _find_model_weight(model_shapes, base)
            if model_key is None:
                shape_errors.append(f"{base}: target weight not found")
            else:
                if len(model_shape) != 2:
                    shape_errors.append(f"{base}: target weight is not rank 2 ({model_shape})")
                elif int(a.shape[1]) != int(model_shape[1]) or int(b.shape[0]) != int(model_shape[0]):
                    shape_errors.append(
                        f"{base}: LoRA ({tuple(a.shape)}, {tuple(b.shape)}) "
                        f"does not match target {model_key} {model_shape}"
                    )

        if dtype == "bf16":
            a = a.to(dtype=torch.bfloat16)
            b = b.to(dtype=torch.bfloat16)
        else:
            a = a.to(dtype=torch.float32)
            b = b.to(dtype=torch.float32)
        converted[f"{base}.lora_A"] = a.contiguous()
        converted[f"{base}.lora_B"] = b.contiguous()
        converted[f"{base}.alpha"] = torch.tensor(alpha, dtype=torch.float32)

    if shape_errors:
        raise ValueError("LoRA/model shape validation failed:\n" + "\n".join(shape_errors[:20]))

    metadata = {
        "format": "taomate_hd.comfy_lora.v1",
        "source_adapter": str(source_dir),
        "source_file": weights_path.name,
        "source_rank": str(rank),
        "source_alpha": str(alpha),
        "converted_dtype": dtype,
        "tensor_count": str(len(converted)),
        "module_count": str(len(pairs)),
    }
    if source_metadata:
        metadata["source_metadata"] = json.dumps(source_metadata, sort_keys=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    save_file(converted, str(partial), metadata=metadata)
    os.replace(partial, output)
    return {
        "output": str(output),
        "module_count": len(pairs),
        "tensor_count": len(converted),
        "rank": rank,
        "alpha": alpha,
        "dtype": dtype,
        "size_bytes": output.stat().st_size,
        "validated_against_model": str(model) if model is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert the TaoMate-HD preview adapter to a ComfyUI-native MiniMax H3 LoRA."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing config.json and adapter_model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .safetensors path, normally under ComfyUI/models/loras/",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional ComfyUI MiniMax H3 diffusion model for key/shape validation",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="LoRA tensor dtype. fp32 is closest to the upstream adapter; bf16 halves file size.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    result = convert_adapter(
        source=args.source.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        dtype=args.dtype,
        model=args.model.expanduser().resolve() if args.model is not None else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
