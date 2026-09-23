# scripts — 数据处理与评测流水线

`ROOT = asumi_voice/`。

- **sbv2 环境**（已保留）：推理、音色指标、评测、调参 —— 部署与常用脚本。
- **asumi 环境**（**已删除**）：仅 `03_vad_trim_resample.py`（silero-vad）、
  `06_transcribe.py` 与 `eval_cer.py`（faster-whisper）需要；如需重跑数据流水线，
  按根目录 `README.md` 的说明重建该环境。

## 数据准备（asumi 环境，已删除，需要时重建）

原始素材放在 `data/source_ogg/`（2020 个 .ogg，游戏解包干声）。

| 脚本 | 作用 | 产出 |
|---|---|---|
| `01_scan_durations.py` | 扫描全部 .ogg 时长 | `processed/manifests/durations.tsv` |
| `02_filter_short.py` | 剔除 <1s 片段（软移动到 excluded） | `kept_files.txt`, `review_short.txt` |
| `03_vad_trim_resample.py` | Silero VAD 去首尾静音 + 重采样 24kHz | `processed/wav24k/*.wav`, `vad_segments.json` |
| `04_split_long.py` | >12s 片段按静音点切分为 3–12s | `processed/dataset/*.wav`, `dataset.tsv` |
| `05_loudnorm.py` | 两遍 EBU R128 响度归一到 -18 LUFS | `processed/dataset_norm/*.wav` |
| `06_transcribe.py` | faster-whisper(ja) 转写（可断点续跑） | `transcripts.tsv`, `train_list.txt` |
| `09_render_sbv2_raw.py` | 由 dataset.tsv 渲染 44.1kHz raw + esd.list | `processed/sbv2/asu/{raw,esd.list}` |

## 评测与调参（sbv2 环境）

| 脚本 | 作用 |
|---|---|
| `eval_sbv2_synth.py` | 用数据集文本配音并计算音色相似度（pyannote wespeaker） |
| `eval_cer.py`（asumi 环境） | ASR 回读计算字符错误率 CER |
| `detect_tremolo.py` | 局部 F0 抖动（颤音）检测与排序 |
| `tune_stability.py` / `sweep_tremolo.py` | 推理参数扫描（noise/sdp/intonation…） |
| `sbv2_tts.py` | 早期 CLI（已被 `../asumi_tts/` 取代，保留备用） |

## 训练

训练在 `Style-Bert-VITS2/` 内使用官方脚本（GPU 服务器上执行）：
`resample.py → preprocess_text.py → bert_gen.py → style_gen.py → train_ms_jp_extra.py`，
详见根目录 `README.md`。

> CuteTTS 中文零样本实验（`07/08`）及其环境、目录均已删除，不再保留。
