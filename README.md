# Asumi-TTS

[锦亚澄 / 錦 あすみ](https://madosoft.net/hamidashi/character#asumi) 的日语语音模型，用 **Style-Bert-VITS2 JP-Extra** 微调得到，通过本地 HTTP 服务的形式供 [Asumi Agent](https://github.com/jayi0908/Asumi-Agent) 调用。

本仓库只包含 Asumi-TTS 的服务端代码，[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 框架本体（BERT 权重、词典编译器）需要单独 clone：

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git ../Style-Bert-VITS2
```

训练后权重放在 Hugging Face 仓库（`jayi0908/style-bert-vits2-asumi`）中，出于版权问题考虑仓库为私有。

## 运行

```bash
# 0) 安装依赖
conda create --name sbv2 python=3.12 -y
conda activate sbv2     # 使用 miniconda / Anaconda 创建虚拟环境 sbv2
pip install -r asumi_tts/requirements.txt   # 安装必要的依赖

# 1) 取权重
hf auth login         # 或 export HF_TOKEN=hf_xxx
python fetch_model.py

# 2) 起服务
./serve.sh            # 默认 127.0.0.1:8077
./serve.sh 9000       # 换端口
```

权重不下载也没关系：第一次合成时 `asumi_tts` 会自己去 Hub 取（同样需要 token）。

> 如果 `resolve/` 卡住或超时，说明 Hub 的 CDN 在当前网络不可达（上传走的是 API，通常是通的；下载会 302 到 CDN）。可以挂镜像：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> python fetch_model.py
> ```
>
> 但镜像只在**出口 IP 属于中国大陆**时才有用。如果挂了代理 / VPN 且出口在境外，hf-mirror 会在跨域跳转时丢掉 `Authorization` 头导致私有仓库被当成匿名访问，这种网络下**别挂镜像**，直接走原站：
>
> ```bash
> export HF_ENDPOINT=https://huggingface.co
> python fetch_model.py
> ```
>
> 另外本仓库启用了 Hub 的 **Xet** 存储，而 `huggingface_hub >= 0.32` 默认走 Xet，在部分网络里会一直卡住。下载困难时设 `HF_HUB_DISABLE_XET=1` 绕开它
>
> ```bash
> export HF_HUB_DISABLE_XET=1        # 绕开会卡住的 Xet 路径
> python fetch_model.py
> ```

### 按架构设置

`asumi_tts` 会按系统自动选择后端，通常不用手动指定；各系统只需在初次配置时补一步。

先分清后端名称（`ASUMI_DEVICE` / `--device`，**留空即自动**）：

| 名称 | 适用范围 | 需要什么 |
|---|---|---|
| `mps` | macOS（Apple Silicon） | PyTorch 自带 |
| `cuda` | 有 NVIDIA 显卡 | CUDA 版 PyTorch；框架的 PyTorch BERT |
| `cpu` | 想用 PyTorch 跑 CPU（任意系统） | 框架的 PyTorch BERT（约 1.3 GB），较慢 |
| `onnx` | Windows 默认；无 GPU 的 CPU 推理 | `asumi_tts/requirements.txt`；首次自动下载 ONNX BERT |
| `dml` | Windows 核显想试 GPU | 同 `onnx`；但每个新句长要重编译，通常更慢 |

自动规则：macOS → `mps`，Windows → `onnx`，其余系统有 CUDA 用 `cuda`、否则 `cpu`。

> `onnx` 和 `cpu` 都会用 CPU，区别在于推理框架：`onnx` 用 ONNX Runtime（Windows 默认、无需 PyTorch BERT），`cpu` 用 PyTorch（需要框架 BERT）。有 NVIDIA 就直接 `cuda`。

#### macOS

PyTorch 自带 MPS，无额外依赖。框架的 BERT 尚未下载时，先在框架目录执行一次：

```bash
(cd ../Style-Bert-VITS2 && python initialize.py --only_infer --skip_default_models)
```

之后按上面的「运行」取权重并 `./serve.sh` 即可。

#### Windows

```powershell
conda activate sbv2
python fetch_model.py
```

默认走 ONNX Runtime（CPU），启动见下方命令。首次启动会自动下载 JP ONNX BERT 并把本模型导出为 ONNX（各一次，之后不再重复）。

#### Linux + NVIDIA

有 NVIDIA 显卡会自动用 CUDA，同样需要框架的 BERT：

```bash
(cd ../Style-Bert-VITS2 && python initialize.py --only_infer --skip_default_models)
```

> 可选环境变量：`ASUMI_DEVICE` 强制后端（`onnx` / `dml` / `cpu` / `cuda` / `mps`）；
> `ASUMI_SDP_RATIO` 调情感（默认 0.2，调大更明显）；`ASUMI_WARMUP=0` 跳过预热。

等价的原始命令（应用就是用这条拉起的），需要时用 `SBV2_ROOT` 指向框架：

```bash
SBV2_ROOT=../Style-Bert-VITS2 \
  conda run -n sbv2 python -m uvicorn asumi_tts.server:app --host 127.0.0.1 --port 8077
```

### HTTP 接口

| 方法 | 路径 | 请求 | 返回 |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","device":"onnx"}` |
| POST | `/tts` | `{"text":"…","style":"Neutral","length":1.0}` | `audio/wav` |
| POST | `/tts/base64` | 同上 | `{"sample_rate":44100,"format":"pcm_s16le","audio_base64":"…"}` |

```bash
curl -X POST http://127.0.0.1:8077/tts -H 'Content-Type: application/json' \
     -d '{"text":"こんにちは、亜澄です。"}' --output hello.wav
```

启动时会预热（加载 JP BERT 并跑一次合成），**预热完成后**每句话通常亚秒到数秒。若不需要预热（例如希望服务秒起、接受首次请求变慢），设 `ASUMI_WARMUP=0`。

### Python API

```python
from asumi_tts import AsumiTTSEngine

tts = AsumiTTSEngine()           # macOS 用 mps，CUDA 用 cuda，CPU 用 cpu，ONNX Runtime 用 onnx
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
