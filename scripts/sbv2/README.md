# SBV2-side setup scripts

Kept here because the framework is a plain clone of
[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) and these were
written for this model's run rather than upstream:

| File | What it does |
|---|---|
| `setup_asu.py` | One-shot preparation of `Data/asu` (mirrors the WebUI's Step 1) |
| `test_asu_infer.py` | Measures the inference memory footprint |
| `requirements-mac.txt` | Dependencies for the environment it ran in |
| `eval_manifest*.tsv` | The utterances behind the similarity and CER numbers in the model card |

They ran against that clone's `Data/` and config, so they are records of how the
model was produced rather than part of the service's runtime.
