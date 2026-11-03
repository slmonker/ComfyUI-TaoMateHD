from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNET = "minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors"
LORA = "TaoMate-HD-preview-comfy.safetensors"
CLIP = r"MiniMax-H3\qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = r"MiniMax-H3\minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = r"MiniMax-H3\minimax_h3_audio_vae_fp32.safetensors"
PROMPT = """Summary: A young East Asian woman sits at a wooden table in a warm sunlit room and speaks directly to the camera in one continuous shot.
Narration: Medium close-up, static eye-level camera, realistic skin texture, soft window light, warm neutral colors, shallow depth of field. One person only, no cuts, no camera movement, no text, no subtitles, no logos.
Speaker_1's Appearance: A young East Asian woman with natural makeup, dark hair tied back, wearing a simple cream blouse.
Speaker_1's Actions: She sits upright, looks into the camera, and speaks naturally with small hand gestures. Hands remain anatomically natural.
Speaker 1 (S1) says: <d>[Chinese] 大家好，这是 TaoMate HD 在 ComfyUI 中的第一次测试。今天我们来验证渐进分辨率采样是否正常。</d>
Speaker_1's Voice Description: Young adult female, warm, clear Mandarin.
Sound-Visual Alignment: Clear dry voice, no music, natural lip sync, no overlapping speech."""


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.api = {}

    def node(self, type_, pos, size, widgets, inputs=(), outputs=(), api_inputs=None, title=None, color=None, bgcolor=None, api=True):
        node_id = len(self.nodes) + 1
        node = {
            "id": node_id, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": node_id, "mode": 0,
            "inputs": [{"name": name, "type": type_name, "link": None} for name, type_name in inputs],
            "outputs": [{"name": name, "type": type_name, "links": []} for name, type_name in outputs],
            "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets),
        }
        if title:
            node["title"] = title
        if color:
            node["color"] = color
        if bgcolor:
            node["bgcolor"] = bgcolor
        self.nodes.append(node)
        if api:
            self.api[str(node_id)] = {"class_type": type_, "inputs": dict(api_inputs or {})}
        return node_id

    def link(self, source, source_slot, target, input_name):
        source_node = self.nodes[source - 1]
        target_node = self.nodes[target - 1]
        target_index = next(i for i, item in enumerate(target_node["inputs"]) if item["name"] == input_name)
        link_id = len(self.links) + 1
        source_node["outputs"][source_slot]["links"].append(link_id)
        target_node["inputs"][target_index]["link"] = link_id
        self.links.append([link_id, source, source_slot, target, target_index, target_node["inputs"][target_index]["type"]])
        self.api[str(target)]["inputs"][input_name] = [str(source), source_slot]

    def ui(self):
        return {
            "last_node_id": len(self.nodes), "last_link_id": len(self.links),
            "nodes": self.nodes, "links": self.links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4,
        }


def build(width: int, height: int, output_prefix: str):
    g = Graph()
    unet = g.node("UNETLoader", (-940, 0), (360, 130), [UNET, "default"], outputs=[("MODEL", "MODEL")], api_inputs={"unet_name": UNET, "weight_dtype": "default"})
    lora = g.node("LoraLoaderModelOnly", (-940, 170), (360, 130), [LORA, 1.0], inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")], api_inputs={"lora_name": LORA, "strength_model": 1.0}, title="TaoMate LoRA (converted)")
    sampling = g.node("MiniMaxH3SigmaShift", (-940, 340), (360, 130), [12.0, 3.0], inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")], api_inputs={"shift_video": 12.0, "shift_audio": 3.0}, title="ModelSamplingMiniMaxH3")
    clip = g.node("CLIPLoader", (-940, 510), (360, 160), [CLIP, "minimax", "default"], outputs=[("CLIP", "CLIP")], api_inputs={"clip_name": CLIP, "type": "minimax", "device": "default"})
    video_vae = g.node("VAELoader", (-940, 710), (360, 100), [VIDEO_VAE], outputs=[("VAE", "VAE")], api_inputs={"vae_name": VIDEO_VAE}, title="Video VAE")
    audio_vae = g.node("VAELoader", (-940, 850), (360, 100), [AUDIO_VAE], outputs=[("VAE", "VAE")], api_inputs={"vae_name": AUDIO_VAE}, title="Audio VAE")
    condition = g.node("MiniMaxH3ImageToVideo", (-500, 0), (430, 500), [PROMPT, width, height, 124], inputs=[("clip", "CLIP"), ("vae", "VAE")], outputs=[("positive", "CONDITIONING"), ("latent", "LATENT")], api_inputs={"prompt": PROMPT, "width": width, "height": height, "length": 124}, title="MiniMax H3 T2VA condition + empty AV latent")
    sampler = g.node("TaoMateHDProgressiveSampler", (10, 0), (400, 220), [8301, "fixed", 3, 1.0, 0.625, True], inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("latent", "LATENT")], outputs=[("latent", "LATENT")], api_inputs={"seed": 8301, "steps": 3, "cfg": 1.0, "low_scale": 0.625, "debug": True}, title="TaoMate HD Progressive Sampler")
    separate = g.node("LTXVSeparateAVLatent", (460, 0), (300, 130), [], inputs=[("av_latent", "LATENT")], outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")], title="Separate AV Latent")
    video_decode = g.node("VAEDecode", (800, -80), (280, 120), [], inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    audio_decode = g.node("VAEDecodeAudio", (800, 90), (280, 120), [], inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("AUDIO", "AUDIO")])
    create_video = g.node("CreateVideo", (1120, 0), (300, 190), [24.0, "auto", "sRGB", "none"], inputs=[("images", "IMAGE"), ("audio", "AUDIO")], outputs=[("VIDEO", "VIDEO")], api_inputs={"fps": 24.0, "bit_depth": "auto", "color_space": "sRGB", "codec": "none"})
    save_video = g.node("SaveVideo", (1460, 0), (340, 220), [output_prefix, "auto", "auto", "auto"], inputs=[("video", "VIDEO")], outputs=[("video", "VIDEO")], api_inputs={"filename_prefix": output_prefix, "format": "auto", "codec": "auto"}, title="Save TaoMate HD video")
    g.node("MarkdownNote", (-940, 1010), (520, 260), [f"TaoMate HD 5s T2VA example\n\nResolution: {width}x{height}, 124 frames, 24 FPS.\nLoRA: {LORA}\n\nThis workflow uses T2VA. Ref2VA can use the same sampler through native H3 minimax_refs conditioning. First/last-frame FL2VA keyframes are not supported.\n\nUse LoraLoaderModelOnly strength 1.0 with the converted LoRA. The source TaoMate adapter must not be loaded directly."], api=False, color="#223", bgcolor="#335")
    for source, slot, target, name in (
        (unet, 0, lora, "model"), (lora, 0, sampling, "model"), (sampling, 0, sampler, "model"),
        (clip, 0, condition, "clip"), (video_vae, 0, condition, "vae"),
        (condition, 0, sampler, "positive"), (condition, 1, sampler, "latent"),
        (sampler, 0, separate, "av_latent"),
        (separate, 0, video_decode, "samples"), (video_vae, 0, video_decode, "vae"),
        (separate, 1, audio_decode, "samples"), (audio_vae, 0, audio_decode, "vae"),
        (video_decode, 0, create_video, "images"), (audio_decode, 0, create_video, "audio"),
        (create_video, 0, save_video, "video"),
    ):
        g.link(source, slot, target, name)
    return g


def main():
    for width, height, stem in ((480, 864, "taomate_hd_t2va_5s_480p"), (768, 1376, "taomate_hd_t2va_5s_768p")):
        g = build(width, height, f"TaoMateHD/{stem}")
        (HERE / f"{stem}.json").write_text(json.dumps(g.ui(), indent=2, ensure_ascii=False), encoding="utf-8")
        (HERE / f"{stem}_api.json").write_text(json.dumps(g.api, indent=2, ensure_ascii=False), encoding="utf-8")
        print(stem)


if __name__ == "__main__":
    main()




