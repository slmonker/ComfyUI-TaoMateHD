# TaoMate HD examples

The directory contains UI workflows and API twins:

- `taomate_hd_t2va_5s_480p.json`
  - 480 x 864, 124 frames, 24 FPS
  - Recommended for a single-GPU smoke test
- `taomate_hd_t2va_5s_768p.json`
  - 768 x 1376, 124 frames, 24 FPS
  - Higher VRAM pressure
- `taomate_hd_dmad_6step_1024_experimental.json`
  - Experimental combined DMAD and TaoMate setup

The workflows expect these ComfyUI model names:

- `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors`
- `TaoMate-HD-preview-comfy.safetensors`
- `MiniMax-H3\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
- `MiniMax-H3\\minimax_h3_video_vae_fp16.safetensors`
- `MiniMax-H3\\minimax_h3_audio_vae_fp32.safetensors`

Select the corresponding models available in your installation. The converted LoRA is available at:

https://huggingface.co/slmonker/TaoMate-HD-preview-comfy

For Ref2VA, replace the T2VA conditioning node with `MiniMax H3 Reference to Video` and keep the same single `positive` and `latent` connections.

Regenerate the workflows after editing model names or prompt content:

```bash
python examples/build_workflow.py
python examples/validate_workflows.py
```
