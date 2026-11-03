# ComfyUI-TaoMateHD 中文说明

[English README](README.md)

这是面向 ComfyUI 原生 MiniMax H3 音视频模型的 TaoMate-HD 三段式渐进采样器非官方适配。

[![GitHub](https://img.shields.io/badge/GitHub-slmonker%2FComfyUI--TaoMateHD-181717)](https://github.com/slmonker/ComfyUI-TaoMateHD)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-TaoMate--HD--preview--comfy-yellow)](https://huggingface.co/slmonker/TaoMate-HD-preview-comfy)

## 节点功能

`TaoMate HD Progressive Sampler` 在一个 ComfyUI 节点内部执行 TaoMate-HD 的三次模型前向：

```text
低分辨率：sigma 1.0  -> sigma(state 16)
低分辨率：sigma 16   -> sigma(state 33)
耦合提升：低分辨率 latent -> 高分辨率 latent
高分辨率：sigma 33   -> sigma 0
```

固定的四个 teacher 状态为：

```text
0, 16, 33, 49
```

默认 `steps=3` 表示：

```text
2 次低分辨率模型前向 + 1 次高分辨率模型前向
```

## 功能

- 支持 ComfyUI 原生 MiniMax H3 AV latent 的三段式渐进采样
- 支持 T2VA
- 通过原生 H3 `minimax_refs` 条件支持 Ref2VA
- 使用耦合噪声完成低分辨率到高分辨率 latent 的提升
- 支持原生 ComfyUI 音视频 latent
- 支持实验性的 4/6/8 步阶段细分

Ref2VA 在当前 ComfyUI 适配中属于实验功能。它使用 MiniMax H3 原生参考条件路径，而不是 TaoMate-HD 官方公开脚本中的纯 T2VA 推理路径。

## 环境要求

- 支持原生 MiniMax H3 的 ComfyUI
- 具备 FL2VA 或 Ref2VA 能力的 MiniMax H3 模型
- MiniMax H3 视频 VAE、音频 VAE 和文本编码器
- 已转换的 ComfyUI 格式 TaoMate-HD LoRA：

  [slmonker/TaoMate-HD-preview-comfy](https://huggingface.co/slmonker/TaoMate-HD-preview-comfy)

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/slmonker/ComfyUI-TaoMateHD.git
```

安装完成后重启 ComfyUI。

## 下载转换后的 LoRA

```bash
hf download slmonker/TaoMate-HD-preview-comfy \
  TaoMate-HD-preview-comfy.safetensors \
  --local-dir ComfyUI/models/loras
```

在节点中这样加载：

```text
LoraLoaderModelOnly
  lora_name = TaoMate-HD-preview-comfy.safetensors
  strength_model = 1.0
```

上传的 LoRA 已经转换为 ComfyUI 使用的键格式，并写入了 TaoMate 的逐模块 alpha。不要把上游原始的 `adapter_model.safetensors` 直接加载到 ComfyUI。

## 自行转换上游 LoRA

```bash
python tools/convert_taomate_lora.py \
  --source /path/to/TaoMate-HD \
  --output /path/to/ComfyUI/models/loras/TaoMate-HD-preview-comfy.safetensors \
  --model /path/to/minimax_h3_hybrid_fl2va_ref2va.safetensors
```

默认输出 FP32 权重，约 2.48 GB。`--dtype bf16` 可以生成更小的实验版本。

## 工作流结构

```text
UNETLoader
  -> LoraLoaderModelOnly
  -> MiniMaxH3SigmaShift (shift_video=12, shift_audio=3)
  -> TaoMate HD Progressive Sampler
  -> Separate AV Latent
       -> VAEDecode（视频 VAE）
       -> VAEDecodeAudio（音频 VAE）
```

### T2VA

```text
MiniMax H3 Image to Video（不接首尾帧）
  positive -> TaoMateHDProgressiveSampler.positive
  latent   -> TaoMateHDProgressiveSampler.latent
```

### Ref2VA

```text
MiniMax H3 Reference to Video
  positive -> TaoMateHDProgressiveSampler.positive
  latent   -> TaoMateHDProgressiveSampler.latent
```

Ref2VA 需要使用具备 Ref2VA 能力的模型或混合模型。参考条件会通过同一个原生 `positive` 输入贯穿三个采样阶段。

## 节点输入

| 输入 | 说明 |
|---|---|
| `model` | 已加载 TaoMate LoRA 的 MiniMax H3 模型 |
| `positive` | MiniMax H3 的 T2VA 或 Ref2VA 条件 |
| `negative` | 可选负条件；只有 `cfg > 1.0` 时需要 |
| `latent` | 空 MiniMax H3 AV latent，或 H3 条件节点输出的 latent |
| `seed` | 视频与音频耦合噪声种子 |
| `steps` | 总模型前向次数；官方默认值为 `3` |
| `cfg` | CFG 强度；TaoMate 蒸馏目标为 `1.0` |
| `low_scale` | 低阶段空间比例；默认 `0.625`，即 480/768 |
| `debug` | 输出阶段尺寸、sigma、步数分配和提升信息 |

## 步数分配

- `steps=3`：`1/1/1`，TaoMate 官方路径
- `steps=4`：`2/1/1`
- `steps=5`：`2/2/1`
- `steps=6`：`2/2/2`
- `steps=8`：`3/3/2`

大于 `3` 的设置只是进一步细分 TaoMate 三个阶段，并不属于官方蒸馏 schedule，属于实验路径。

## 示例工作流

目录 [`examples/`](examples/) 只包含两个 UI 工作流：

- `taomateHD_t2va.json`
  - 480 × 864，124 帧，原生 T2VA 条件
- `taomateHD_ref2va.json`
  - 480 × 864，124 帧，原生 Ref2VA 条件
  - 导入后请在 `LoadImage` 节点中选择自己的参考图

模型文件名使用 ComfyUI 中实际安装的模型名。请按本机环境选择对应的 H3 模型、文本编码器、视频 VAE 和音频 VAE。

## 当前限制

当前版本是单窗口概念验证，不包含上游完整流式长视频实现：

- 单个 5 秒窗口，约 124 帧 / 24 FPS
- batch size 为 1
- 支持 T2VA
- Ref2VA 通过原生 H3 条件支持，属于实验功能
- 尚未实现 FL2VA 首尾帧 keyframe 条件
- 不支持 denoise mask
- 不包含 5 秒窗口循环
- 不包含跨窗口 stage-local KV cache
- 不包含 8 卡 1/1/4/2 流水线

## 硬件说明

上游参考实现使用 8 张 H20 96 GB GPU 验证。本 ComfyUI 适配复用原生 H3 加载器和量化模型，目标是支持单卡单窗口实验，但不保证达到上游速度或显存表现。

对于 32 GB 显存 GPU，建议：

- 使用 int8/fp8 H3 模型
- 保持文本编码器可 offload
- 先在 480p 验证，再尝试 768p
- 高分辨率最后阶段仍需要足够的单帧显存

## 实现说明

- 四个 teacher 状态固定为 `0, 16, 33, 49`
- 低阶段状态通过耦合噪声提升到高分辨率网格
- 音频 latent 几何在整个过程中保持不变
- 采样器要求原生 MiniMax H3 嵌套 AV latent
- Ref2VA 通过原生 `minimax_refs` 条件元数据传递

## 致谢与许可

这是非官方 ComfyUI 适配，与阿里巴巴淘Live AIGC 或 MiniMax 无隶属关系。

- TaoMate-HD：https://github.com/TaoLiveAIGC/TaoMate-HD
- MiniMax H3：https://huggingface.co/MiniMaxAI/MiniMax-H3

适配代码遵循 MiniMax H3 Community License Agreement。请参阅 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。基础模型和上游适配器仍适用其原始许可。
