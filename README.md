# Asumi-TTS

亜澄（あすみ）的日语语音，用 **Style-Bert-VITS2 JP-Extra** 微调得到，并以一个本地
HTTP 服务的形式提供给 [Asumi Agent](https://github.com/jayi0908/Asumi-Agent) 使用。

- **模型**：JP-Extra，40 epoch / 9200 步。权重（240MB）托管在
  **Hugging Face**（私有）：`jayi0908/style-bert-vits2-asumi`；仓库里只放
  `config.json` 与 `style_vectors.npy`，权重首次运行自动下载
- **输出**：44.1kHz 单声道 WAV
- **语言**：**仅日语**。前端是日语专用的：中文会被按汉字音读、拉丁字母会被逐个拼读，
  所以调用方必须只把日语送进来（Asumi Agent 用「日语/中文」两个字段来保证这一点）。
- **速度**：Apple M4 上走 MPS，热身后约 4× 实时（2.8s 语音约 0.66s），常驻内存约 3.5GB

## 依赖

| 依赖 | 说明 |
|---|---|
| `Style-Bert-VITS2` 仓库 | 框架本体（BERT 权重、词典编译器）。**不随本仓库分发**，需单独 clone |
| conda 环境 `sbv2` | torch 2.3.1 + MPS，框架与推理共用 |

框架默认放在**本仓库同级目录**（`../Style-Bert-VITS2`），也可以用 `SBV2_ROOT` 指定：

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git ../Style-Bert-VITS2
```

## 运行

```bash
# 1) 取权重（首次；Hub 仓库是私有的，需要能读它的 token）
hf auth login                       # 或 export HF_TOKEN=hf_xxx
python fetch_model.py

# 2) 起服务
./serve.sh            # 默认 127.0.0.1:8077
./serve.sh 9000       # 换端口
```

权重不下载也没关系：第一次合成时 `asumi_tts` 会自己去 Hub 取（同样需要 token）。

> 如果 `resolve/` 卡住或超时，说明 Hub 的 CDN 在当前网络不可达（上传走的是 API，
> 通常是通的；下载会 302 到 CDN）。挂镜像即可：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> python fetch_model.py
> ```

等价的原始命令（应用就是用这条拉起的），需要时用 `SBV2_ROOT` 指向框架：

```bash
SBV2_ROOT=../Style-Bert-VITS2 \
  conda run -n sbv2 python -m uvicorn asumi_tts.server:app --host 127.0.0.1 --port 8077
```

### HTTP 接口

| 方法 | 路径 | 请求 | 返回 |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","device":"mps"}` |
| POST | `/tts` | `{"text":"…","style":"Neutral","length":1.0}` | `audio/wav` |
| POST | `/tts/base64` | 同上 | `{"sample_rate":44100,"format":"pcm_s16le","audio_base64":"…"}` |

```bash
curl -X POST http://127.0.0.1:8077/tts -H 'Content-Type: application/json' \
     -d '{"text":"こんにちは、亜澄です。"}' --output hello.wav
```

首次调用会加载模型（约 5s），之后每句话约 0.7s。参数 `length` >1 更慢。

### Python API

```python
from asumi_tts import AsumiTTSEngine

tts = AsumiTTSEngine(device="mps")          # 只加载一次
tts.speak_to_file("おはよう、亜澄だよ。", "hello.wav")
```

`speak()` 返回 `(sample_rate, float32 waveform)`，波形在 **[-1, 1]**。

## 在 Asumi Agent 中使用

应用的设置里指定三项，之后它会在第一次需要发声时自动拉起本服务：

| 设置 | 值 |
|---|---|
| 语音仓库 | 本仓库的路径 |
| 服务地址 | `http://127.0.0.1:8077` |
| Python 路径 | `sbv2` 环境里的 python |

应用只会把日语那半句送进来，服务没应答时它保持静默，不会改用别的嗓音。

## 权重放在哪、为什么

| 文件 | 位置 | 原因 |
|---|---|---|
| `asu_e40_s9200.safetensors`（240MB） | Hugging Face（私有） | 超过 GitHub 单文件 100MB 硬限；Hub 无单文件限制、免费，也不占用 GitHub LFS 的配额与流量 |
| `config.json`、`style_vectors.npy` | 本仓库 | 只有几 KB，是模型的一部分，放在这里便于直接查看 |

下载由 `asumi_tts` 内的 `ensure_weights()` 完成（`huggingface_hub`），或直接跑
`python fetch_model.py`。目标仓库可用 `ASUMI_HF_REPO` 覆盖，例如指向你自己 fork 的权重。

## 模型是怎么来的

训练流水线保留在 `scripts/`（数据清洗、VAD 切分、响度归一、转写、SBV2 预处理、评测）。
数据与中间产物（`processed/`、`build/`）不在仓库里：可从原始素材重跑，且原始素材本身
不随仓库分发。要点：

- 数据清洗后 **1999 段 / 3.63h / 44.1kHz**，在 GPU 服务器上微调 44 分钟（batch 8、bf16）。
- 对比过 e40 与 e100：**e40 最优**（e100 过拟合、CER 变差），故部署 e40。
- 专有名词靠 `dict_data/default.csv`（`亜澄 → アスミ`、`錦亜澄 → ニシキアスミ`）。

## 两个坑（改代码前先读）

**1. 输出标度。** `TTSModel.infer()` 按设计返回 **int16 量级**的数组，而 `soundfile`
写 `PCM_16` 时期望 float 在 ±1 —— 直接把 int16 量级的 float 交给它会**铲平 91% 的采样**
（听感是"炸麦 + 沙沙声"）。`engine.py` 因此按模型 config 里的 `max_wav_value` 缩到 ±1。
判断依据：`rms≈0.95 / 削波≈90%` 就是踩了这个坑，正常应 `rms≈0.13 / 削波≈0%`。

**2. 只喂日语。** 见上文——模型不会因为输入是中文而报错，只会读错。

## 版权

音色克隆自商业作品《ハミダシクリエイティブ》的角色 **錦 あすみ**，角色与音声权利归
Madosoft 所有。本仓库仅供个人研究使用，**请勿再分发模型权重或用于商业用途**。
