# TaoMate HD examples

This directory contains four UI workflows:

- `taomateHD_t2va.json`
  - 480 x 864, 124 frames, native MiniMax H3 T2VA conditioning
- `taomateHD_ref2va.json`
  - 480 x 864, 124 frames, native MiniMax H3 Ref2VA conditioning
  - Choose your own reference image in the `LoadImage` node before running
- `TaoMateHD_digital_human.json`
  - Digital-human workflow with reference-image conditioning and source-audio preservation
  - Uses `SolidMask=0` on the audio latent so TaoMate HD keeps the source voice
  - Uses `ResolutionSelector.multiple=32` so H3 latent spatial axes stay even
  - Replace both the reference image and source audio before running
- `TaoMateHD_digital_human_SOL.json`
  - SOL block-sparse accelerated digital-human workflow
  - Adds `BlockSparseAttention` (`sol-attn`), local prompt enhancement, model preview and preview output nodes
  - Requires the corresponding custom-node packs in addition to the TaoMate HD requirements
  - Replace both the reference image and source audio before running

All workflows expect:

- `TaoMate-HD-preview-comfy.safetensors`
- A MiniMax H3 model with T2VA or Ref2VA capability
- The corresponding MiniMax H3 text encoder, video VAE, and audio VAE

The converted LoRA is available at:

https://huggingface.co/slmonker/TaoMate-HD-preview-comfy

## Demo

- [Animated preview](../demo/taomate_hd_digital_human_sol.gif)
- [MP4 with audio](../demo/taomate_hd_digital_human_sol.mp4)

