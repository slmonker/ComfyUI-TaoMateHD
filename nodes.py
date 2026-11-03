from __future__ import annotations

import logging
import math

import torch

import comfy.model_base
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from .progressive import (
    DEFAULT_LOW_SCALE,
    SWITCH_STATE_INDEX,
    ProgressiveCanvas,
    audio_noise,
    coupled_video_noise,
    derive_low_canvas,
    latent_t_to_pixel_frames,
    lift_video_latent,
    progressive_stage_sigmas,
)

LOGGER = logging.getLogger(__name__)


def _as_nested_streams(samples):
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError("TaoMate HD expects the native MiniMax H3 nested AV latent")
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError(f"expected 2 AV streams, got {len(streams)}")
    return streams[0], streams[1]


def _validate_h3_latent(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a ComfyUI latent dictionary")
    if "noise_mask" in latent:
        raise ValueError("TaoMate HD progressive sampler does not support latent noise masks")
    video, audio = _as_nested_streams(latent["samples"])
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(f"expected H3 video latent [B,24,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
        raise ValueError(f"expected H3 audio latent [B,32,2,T], got {tuple(audio.shape)}")
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("TaoMate HD currently supports batch size 1")
    if int(video.shape[-2]) % 2 or int(video.shape[-1]) % 2:
        raise ValueError("H3 video latent spatial dimensions must be even for 2x2 patchification")
    video_latent_t = int(video.shape[2])
    frame_count = latent_t_to_pixel_frames(video_latent_t)
    expected_audio_t = round(frame_count * 40.0 / 24.0)
    if int(audio.shape[-1]) != expected_audio_t:
        raise ValueError(
            f"audio latent length {int(audio.shape[-1])} does not match video duration "
            f"{frame_count} frames ({expected_audio_t} audio latents)"
        )
    high = ProgressiveCanvas.from_video_latent(video)
    return video, audio, high


def _validate_conditioning(conditioning, *, name):
    if not conditioning:
        raise ValueError(f"{name} conditioning is required")
    for item in conditioning:
        if len(item) < 2:
            raise ValueError(f"invalid ComfyUI {name} conditioning entry")
        values = item[1]
        if "minimax_keyframes" in values:
            raise NotImplementedError(
                "TaoMate HD supports T2VA and Ref2VA in this node, but not FL2VA "
                "first/last-frame keyframes yet; disconnect first_frame/last_frame "
                "or use the native MiniMax H3 sampler for FL2VA"
            )


class _TaoMateStageSampler:
    """One Euler state transition with optional continuation from the input state."""

    def __init__(self, *, start_from_latent: bool, stage_name: str):
        self.start_from_latent = bool(start_from_latent)
        self.stage_name = stage_name
        self.last_denoised = None

    @torch.no_grad()
    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if denoise_mask is not None:
            raise ValueError("TaoMate HD progressive stages do not support denoise masks")
        if len(sigmas) < 2:
            raise ValueError(f"{self.stage_name}: expected at least two sigma boundaries")
        model_sampling = model_wrap.inner_model.model_sampling
        if self.start_from_latent:
            x = latent_image
            if x is None:
                raise ValueError(f"{self.stage_name}: continuation state is missing")
        else:
            x = model_sampling.noise_scaling(
                sigmas[0], noise, latent_image, max_denoise=False
            )

        total_steps = len(sigmas) - 1
        for step in range(total_steps):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            s_in = x.new_ones([x.shape[0]])
            denoised = model_wrap(x, sigma * s_in, **extra_args)
            self.last_denoised = denoised
            if callback is not None:
                callback(step, denoised, x, total_steps)
            if float(sigma_next) == 0.0:
                x = denoised
            else:
                ratio = sigma_next / sigma
                x = x + (denoised - x) * (1.0 - ratio)
        return x


def _allocate_stage_steps(total_steps: int) -> tuple[int, int, int]:
    total_steps = int(total_steps)
    if total_steps < 3:
        raise ValueError("steps must be at least 3 for the three TaoMate stages")
    first = math.ceil(total_steps / 3)
    remaining = total_steps - first
    second = math.ceil(remaining / 2)
    third = remaining - second
    if third < 1:
        raise ValueError(f"cannot allocate {total_steps} steps across three stages")
    return first, second, third


def _split_sigma_interval(start: float, end: float, steps: int) -> list[float]:
    steps = int(steps)
    if steps < 1:
        raise ValueError("each TaoMate stage needs at least one step")
    return [float(start + (end - start) * index / steps) for index in range(steps + 1)]


def _run_stage(guider, *, noise, latent_image, sigmas, start_from_latent, stage_name, seed):
    sampler = _TaoMateStageSampler(
        start_from_latent=start_from_latent, stage_name=stage_name
    )
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
    LOGGER.info(
        "[TaoMateHD] %s sigma %.6f -> %.6f steps=%d start_from_latent=%s",
        stage_name,
        float(sigmas[0]),
        float(sigmas[-1]),
        len(sigmas) - 1,
        start_from_latent,
    )
    return guider.sample(
        noise,
        latent_image,
        sampler,
        sigma_tensor,
        denoise_mask=None,
        callback=None,
        disable_pbar=True,
        seed=seed,
    )


class TaoMateHDProgressiveSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TaoMateHDProgressiveSampler",
            display_name="TaoMate HD Progressive Sampler",
            category="model/sampling/taomate",
            description=(
                "TaoMate-HD three-forward progressive sampler for MiniMax H3. "
                "Load the converted TaoMate LoRA with LoraLoaderModelOnly first."
            ),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model patched with a converted TaoMate LoRA."),
                io.Conditioning.Input(
                    "positive",
                    tooltip="MiniMax H3 conditioning. Supports T2VA and Ref2VA.",
                ),
                io.Conditioning.Input(
                    "negative",
                    optional=True,
                    tooltip="Optional negative conditioning. Required only when cfg > 1.0.",
                ),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Empty MiniMax H3 AV latent. Marked optional only so the UI keeps negative below positive; it is required at execution time.",
                ),
                io.Int.Input(
                    "seed",
                    default=8301,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for coupled progressive video noise and audio noise.",
                ),
                io.Int.Input(
                    "steps",
                    default=3,
                    min=3,
                    max=24,
                    step=1,
                    advanced=True,
                    tooltip="Total model evaluations. 3 is the official TaoMate count; 6 is an experimental 2/2/2 split.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Classifier-free guidance scale. TaoMate is distilled for cfg=1.0; values above 1.0 are experimental.",
                ),
                io.Float.Input(
                    "low_scale",
                    default=DEFAULT_LOW_SCALE,
                    min=0.3,
                    max=0.9,
                    step=0.001,
                    tooltip="Low-stage resolution scale. Official default is 480/768 = 0.625.",
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    advanced=True,
                    tooltip="Log stage shapes, sigma boundaries, step allocation, and lift details.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, model, positive, negative=None, latent=None, seed=8301, steps=3, cfg=1.0, low_scale=DEFAULT_LOW_SCALE, debug=False):
        if not isinstance(getattr(model, "model", None), comfy.model_base.MiniMaxH3):
            raise ValueError("TaoMate HD Progressive Sampler requires a native MiniMax H3 model")
        if latent is None:
            raise ValueError("latent input is required")
        _validate_conditioning(positive, name="positive")
        if float(cfg) != 1.0 and negative is None:
            raise ValueError("cfg > 1.0 requires the optional negative conditioning input")
        video, audio, high_canvas = _validate_h3_latent(latent)
        low_canvas = derive_low_canvas(high_canvas, float(low_scale))
        stage_sigmas = progressive_stage_sigmas()
        sigma_0, sigma_16, sigma_33, sigma_49 = (
            stage_sigmas[0][0],
            stage_sigmas[0][1],
            stage_sigmas[1][1],
            stage_sigmas[2][1],
        )
        stage_counts = _allocate_stage_steps(int(steps))
        stage1_sigmas = _split_sigma_interval(sigma_0, sigma_16, stage_counts[0])
        stage2_sigmas = _split_sigma_interval(sigma_16, sigma_33, stage_counts[1])
        stage3_sigmas = _split_sigma_interval(sigma_33, sigma_49, stage_counts[2])
        if debug:
            LOGGER.info(
                "[TaoMateHD] high=%s low=%s sigma=%s steps=%s stage_steps=%s cfg=%s",
                high_canvas.as_dict(),
                low_canvas.as_dict(),
                (sigma_0, sigma_16, sigma_33, sigma_49),
                int(steps),
                stage_counts,
                float(cfg),
            )

        video_latent_t = int(video.shape[2])
        low_video = torch.zeros(
            (
                1,
                24,
                video_latent_t,
                low_canvas.latent_height,
                low_canvas.latent_width,
            ),
            dtype=video.dtype,
            device=comfy.model_management.intermediate_device(),
        )
        low_audio = torch.zeros(
            tuple(int(v) for v in audio.shape),
            dtype=audio.dtype,
            device=comfy.model_management.intermediate_device(),
        )

        low_video_noise, high_video_noise = coupled_video_noise(
            low_canvas=low_canvas,
            high_canvas=high_canvas,
            temporal_length=video_latent_t,
            seed=int(seed),
        )
        low_audio_noise = audio_noise(
            audio_shape=tuple(int(v) for v in audio.shape),
            high_canvas=high_canvas,
            temporal_length=video_latent_t,
            seed=int(seed),
        )
        low_noise = comfy.nested_tensor.NestedTensor((low_video_noise, low_audio_noise))
        zero_low_noise = comfy.nested_tensor.NestedTensor(
            (torch.zeros_like(low_video), torch.zeros_like(low_audio))
        )
        low_empty = comfy.nested_tensor.NestedTensor((low_video, low_audio))

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative if negative is not None else [])
        guider.set_cfg(float(cfg))

        pbar = comfy.utils.ProgressBar(3)
        stage1 = _run_stage(
            guider,
            noise=low_noise,
            latent_image=low_empty,
            sigmas=stage1_sigmas,
            start_from_latent=False,
            stage_name="low-0-to-16",
            seed=int(seed),
        ).to(comfy.model_management.intermediate_device())
        pbar.update(1)

        stage2 = _run_stage(
            guider,
            noise=zero_low_noise,
            latent_image=stage1,
            sigmas=stage2_sigmas,
            start_from_latent=True,
            stage_name="low-16-to-33",
            seed=int(seed),
        ).to(comfy.model_management.intermediate_device())
        pbar.update(1)

        stage2_video, stage2_audio = _as_nested_streams(stage2)
        target_device = stage2_video.device
        lifted_video = lift_video_latent(
            low_video=stage2_video,
            low_noise=low_video_noise.to(device=target_device),
            high_noise=high_video_noise.to(device=target_device),
            sigma=sigma_33,
            low_canvas=low_canvas,
            high_canvas=high_canvas,
        )
        lifted = comfy.nested_tensor.NestedTensor((lifted_video, stage2_audio))
        zero_high_noise = comfy.nested_tensor.NestedTensor(
            (torch.zeros_like(lifted_video), torch.zeros_like(stage2_audio))
        )

        stage3 = _run_stage(
            guider,
            noise=zero_high_noise,
            latent_image=lifted,
            sigmas=stage3_sigmas,
            start_from_latent=True,
            stage_name="high-33-to-49",
            seed=int(seed),
        ).to(comfy.model_management.intermediate_device())
        pbar.update(1)
        out = latent.copy()
        out["samples"] = stage3
        out.pop("noise_mask", None)
        if debug:
            LOGGER.info("[TaoMateHD] completed final latent shape %s", tuple(stage3.shape))
        return io.NodeOutput(out)


__all__ = ["TaoMateHDProgressiveSampler"]



