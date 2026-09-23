# Asumi-TTS

[锦亚澄 / 錦 あすみ](https://madosoft.net/hamidashi/character#asumi) 的日语语音模型，用 **Style-Bert-VITS2 JP-Extra** 微调得到，通过本地 HTTP 服务的形式供 [Asumi Agent](https://github.com/jayi0908/Asumi-Agent) 调用。

本仓库只包含 Asumi-TTS 的服务端代码，[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 框架本体（BERT 权重、词典编译器）需要单独 clone：

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git ../Style-Bert-VITS2
```

训练后权重放在 Hugging Face 仓库（`jayi0908/style-bert-vits2-asumi`）中，出于版权问题考虑仓库为私有。

## 运行

```bash
# 1) 取权重
hf auth login         # 或 export HF_TOKEN=hf_xxx
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

首次调用会加载模型（约 5s），之后每句话约 0.7s。

### Python API

```python
from asumi_tts import AsumiTTSEngine

tts = AsumiTTSEngine(device="mps")          # 只加载一次
tts.speak_to_file("おはよう、亜澄だよ。", "hello.wav")
```

## 在 Asumi Agent 中使用

应用的设置里指定三项，之后它会在第一次需要发声时自动拉起本服务：

| 设置 | 值 |
|---|---|
| 语音仓库 | 本仓库的路径 |
| 服务地址 | `http://127.0.0.1:8077` |
| Python 路径 | `sbv2` 环境里的 python |

应用只会把日语那半句送进来，服务没应答时它保持静默，不会改用别的嗓音。

## 版权

音色克隆自商业作品《ハミダシクリエイティブ》的角色 **錦 あすみ**，角色与音声权利归
Madosoft 所有。本仓库仅供个人研究使用，**请勿再分发模型权重或用于商业用途**。
