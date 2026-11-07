"""Latent-mask helpers for TaoMate HD's progressive MiniMax H3 sampler."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.nested_tensor


def _description(value):
    if getattr(value, "is_nested", False):
        return f"{type(value).__name__}({[tuple(stream.shape) for stream in value.unbind()]})"
    return f"{type(value).__name__}(shape={getattr(value, 'shape', None)})"


def resize_video_latent(latent, height, width):
    """Spatially resize a video latent or spatial mask without touching time."""
    if tuple(latent.shape[-2:]) == (int(height), int(width)):
        return latent
    if latent.ndim != 5:
        raise ValueError(f"expected a rank-5 video latent, got {tuple(latent.shape)}")
    batch, channels, frames, in_height, in_width = latent.shape
    flattened = latent.float().permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, in_height, in_width
    )
    resized = F.interpolate(
        flattened,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, int(height), int(width)).permute(
        0, 2, 1, 3, 4
    ).to(dtype=latent.dtype).contiguous()


def _normalize_video_mask(mask, shape):
    if mask.ndim == 3:  # B,H,W
        mask = mask[:, None, None]
    elif mask.ndim == 4:  # B,C,H,W, constant over time
        mask = mask[:, :, None]
    if mask.ndim != 5:
        raise ValueError(
            "TaoMate HD video mask must be BHW, BCHW, or BCTHW; "
            f"received {_description(mask)}"
        )
    if mask.shape[0] not in (1, shape[0]):
        raise ValueError(f"video mask batch {mask.shape[0]} does not match latent batch {shape[0]}")
    if mask.shape[2] not in (1, shape[2]):
        raise ValueError(f"video mask time {mask.shape[2]} does not match latent time {shape[2]}")

    # A single binary constraint is sufficient for all latent channels. Taking
    # amax keeps any requested generation region active.
    mask = mask.float().clamp(0.0, 1.0).amax(dim=1, keepdim=True)
    if mask.shape[2] == 1 and shape[2] != 1:
        mask = mask.expand(-1, -1, shape[2], -1, -1)
    if tuple(mask.shape[-2:]) != tuple(shape[-2:]):
        mask = resize_video_latent(mask, shape[-2], shape[-1])
    if mask.shape[0] == 1 and shape[0] != 1:
        mask = mask.expand(shape[0], -1, -1, -1, -1)
    return mask.contiguous()


def _normalize_audio_mask(mask, shape):
    audio_shaped = True
    if mask.ndim == 1:  # T
        mask = mask[None, None, None, :]
    elif mask.ndim == 2:  # B,T
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:  # B,S,T
        mask = mask[:, None, :, :]
    elif mask.ndim == 4:
        audio_shaped = (
            mask.shape[1] in (1, shape[1])
            and mask.shape[2] in (1, shape[2])
            and mask.shape[3] in (1, shape[3])
        )
    else:
        raise ValueError(
            "TaoMate HD audio mask must be T, BT, BST, or BCST; "
            f"received {_description(mask)}"
        )

    # SetLatentNoiseMask stores SolidMask at image resolution. A constant image
    # grid is a whole-audio control, not a time mask, so collapse it without
    # inventing a time mapping.
    if not audio_shaped:
        first = mask[:, :, :1, :1]
        if not torch.equal(mask, first.expand_as(mask)):
            raise ValueError(
                "TaoMate HD audio mask has a non-uniform image grid; use a constant "
                "SolidMask for whole-audio control or a BCST audio mask"
            )
        mask = first

    if mask.shape[3] not in (1, shape[3]):
        raise ValueError(
            f"audio mask time {mask.shape[3]} does not match latent audio time {shape[3]}; "
            "audio is not time-resampled"
        )
    if mask.shape[0] not in (1, shape[0]):
        raise ValueError(f"audio mask batch {mask.shape[0]} does not match latent batch {shape[0]}")

    mask = mask.float().clamp(0.0, 1.0).amax(dim=(1, 2), keepdim=True)
    if mask.shape[3] == 1 and shape[3] != 1:
        mask = mask.expand(-1, -1, -1, shape[3])
    if mask.shape[0] == 1 and shape[0] != 1:
        mask = mask.expand(shape[0], -1, -1, -1)
    return mask.contiguous()


def normalize_noise_mask(noise_mask, video, audio):
    """Normalize a ComfyUI H3 AV noise mask into nested video/audio masks."""
    if noise_mask is None:
        return None
    if getattr(noise_mask, "is_nested", False):
        streams = list(noise_mask.unbind())
        if len(streams) != 2:
            raise ValueError(
                f"TaoMate HD expects a two-stream AV noise mask, received {_description(noise_mask)}"
            )
        video_mask, audio_mask = streams
    elif isinstance(noise_mask, torch.Tensor):
        video_mask, audio_mask = noise_mask, None
    else:
        raise ValueError(f"unsupported TaoMate HD noise mask {_description(noise_mask)}")

    video_mask = _normalize_video_mask(video_mask, tuple(video.shape))
    if audio_mask is None:
        audio_mask = audio.new_ones((1, 1, 1, 1))
    audio_mask = _normalize_audio_mask(audio_mask, tuple(audio.shape))
    return comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def masks_for_video_shape(masks, video):
    if masks is None:
        return None
    video_mask, audio_mask = masks.unbind()
    return comfy.nested_tensor.NestedTensor(
        (resize_video_latent(video_mask, video.shape[-2], video.shape[-1]), audio_mask)
    )


def restore_kept(outputs, anchors, masks):
    """Restore exact mask=0 regions while leaving soft mask values to sampling."""
    restored = []
    for output, anchor, mask in zip(outputs, anchors, masks):
        keep = mask.to(output.device) == 0
        restored.append(torch.where(keep, anchor.to(output), output))
    return restored


__all__ = [
    "masks_for_video_shape",
    "normalize_noise_mask",
    "resize_video_latent",
    "restore_kept",
]
