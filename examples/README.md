# TaoMate HD examples

This directory contains two UI workflows:

- `taomateHD_t2va.json`
  - 480 x 864, 124 frames, native MiniMax H3 T2VA conditioning
- `taomateHD_ref2va.json`
  - 480 x 864, 124 frames, native MiniMax H3 Ref2VA conditioning
  - Replace the `Load reference image` placeholder `example.png` with your own image

Both workflows expect:

- `TaoMate-HD-preview-comfy.safetensors`
- A MiniMax H3 model with T2VA or Ref2VA capability
- The corresponding MiniMax H3 text encoder, video VAE, and audio VAE

The converted LoRA is available at:

https://huggingface.co/slmonker/TaoMate-HD-preview-comfy
