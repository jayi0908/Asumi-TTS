# asumi_tts — 亜澄（あすみ）日语语音合成调用包

面向应用调用的轻量封装：加载一次微调好的 **Style-Bert-VITS2 JP-Extra** 模型，
提供 Python API / 命令行 / HTTP 微服务三种调用方式。

- 模型：`asu_e40_s9200.safetensors`（JP-Extra，40 epoch / 9200 步，最佳 checkpoint）
- 设备：Apple M4 上走 **MPS**；Windows 默认 **ONNX Runtime / CPU**（`device="onnx"`）；有 CUDA 用 `cuda`
- 语言：**仅日语**（JP-Extra）。输入日文文本，输出 44.1 kHz 单声道 wav
- 专有名词：内置用户词典（`亜澄 → アスミ`、`錦亜澄 → ニシキアスミ`）

> 必须使用 Style-Bert-VITS2 的 conda 环境（`sbv2`），因为引擎依赖其
> `style_bert_vits2` 代码、日语 BERT 与 pyopenjtalk 词典。

---

## 1. Python API（应用是 Python 时首选，同进程、无网络开销）

```python
from asumi_tts import AsumiTTSEngine

tts = AsumiTTSEngine(device="onnx")          # Windows；macOS 用 mps，CUDA 用 cuda
sr, wav = tts.speak("おはよう、亜澄だよ。")   # sr=44100, wav=float32 numpy
tts.speak_to_file("また明日も会えるといいな。", "hello.wav")
```

- `speak(text, *, style="Neutral", intonation_scale=1.0, length=1.0)`
- 线程安全（内部加锁），可多线程调用
- `get_engine()` 返回进程级单例，避免重复加载

## 2. 命令行

```bash
cd /Users/jayi0908/Desktop/Something/asumi/asumi_voice
conda run -n sbv2 python -m asumi_tts.cli \
  --text "おはよう、亜澄だよ。" --out hello.wav

# 多句 -> 目录（001.wav, 002.wav, ...）
conda run -n sbv2 python -m asumi_tts.cli --text-file lines.txt --out outdir/
```

## 3. HTTP 微服务（应用是非 Python / 跨进程时首选）

启动：

```bash
cd /Users/jayi0908/Desktop/Something/asumi/asumi_voice
conda run -n sbv2 python -m uvicorn asumi_tts.server:app --host 127.0.0.1 --port 8077
```

接口：

| 方法 | 路径 | 请求 | 返回 |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","device":"dml"}` |
| POST | `/tts` | `{"text":"...", "style":"Neutral", "intonation_scale":1.0, "length":1.0}` | `audio/wav` 二进制 |
| POST | `/tts/base64` | 同上 | `{"sample_rate":44100,"format":"pcm_s16le","audio_base64":"..."}` |

调用：

```bash
curl -X POST http://127.0.0.1:8077/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"こんにちは、亜澄です。"}' --output hello.wav
```

Python 客户端示例见 `examples/http_client.py`。

---

## 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `text` | — | 日文台词 |
| `style` | `Neutral` | 风格名（当前模型仅 Neutral） |
| `intonation_scale` | `1.0` | 音高起伏缩放。降低可减少“颤音”，但会令声音发虚，**建议保持 1.0** |
| `length` | `1.0` | 语速，>1 更慢 |
| `sdp_ratio` / `noise` / `noise_w` | 0.2 / 0.667 / 0.8 | 生成随机性，一般无需改动 |

## 目录

```
asumi_tts/
  engine.py            核心引擎（加载模型 + speak）
  export_onnx.py       把 safetensors 导出为 ONNX（DirectML 路径用，免 PyTorch BERT）
  cli.py               命令行
  server.py            FastAPI 服务
  examples/            python_api.py, http_client.py
  requirements.txt
```

## 常见问题

- **端口占用**：改 `--port`。
- **想用 CPU**：`device="cpu"`（或设环境变量 `ASUMI_DEVICE=cpu`），走 PyTorch，更慢但更省内存（需 1.3 GB PyTorch BERT）。
- **Windows**：默认 `device="onnx"`，走 ONNX Runtime / CPU，首次会自动下载 JP ONNX BERT（约 650 MB）并把权重导出成 ONNX。需要 `onnxruntime-directml`（见 `requirements.txt`）。
- **DirectML（`device="dml"`）**：能用核显，但每个新输入长度都要重编译内核（约 13 s），自由文本每句都会等，实测比 CPU 慢约 9 倍，除非输入长度固定，否则不建议。
- **模型路径**：可用环境变量覆盖 `SBV2_ROOT` / `ASUMI_MODEL_DIR` / `ASUMI_MODEL` / `ASUMI_DEVICE`。
- **JP BERT 精度**：ONNX 路径默认用框架的 fp16 BERT（约 650 MB）。若觉得比 Mac/MPS 情绪淡，设 `ASUMI_ONNX_BERT=fp32`（或 CLI `--onnx-bert fp32`）切到 fp32（约 1.3 GB，首次自动下载）；合成器两条路径都是 fp32。
- **情感强度**：服务端默认值可用环境变量覆盖（Agent 只发 `text`，所以以这里为准）：`ASUMI_SDP_RATIO`（默认 0.2，调高更有起伏）、`ASUMI_NOISE`、`ASUMI_NOISE_W`、`ASUMI_INTONATION_SCALE`、`ASUMI_LENGTH`、`ASUMI_STYLE`。设完重启服务生效。
- **首次加载**：服务启动时预热（加载 JP BERT + 词典，跑一次合成）；之后按设备单句亚秒到数秒。设 `ASUMI_WARMUP=0` 可跳过预热（服务秒起，但首句变慢）。
