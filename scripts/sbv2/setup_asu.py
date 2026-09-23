"""One-shot setup for the 'asu' JP-Extra fine-tuning run (mirrors the WebUI Step 1)."""
import json
import shutil
from pathlib import Path

import yaml

MODEL_NAME = "asu"
DATASET = Path("Data") / MODEL_NAME
USE_JP_EXTRA = True
BATCH_SIZE = 2
EPOCHS = 10
SAVE_EVERY_STEPS = 500
LOG_INTERVAL = 50

def main() -> None:
    template = "configs/config_jp_extra.json" if USE_JP_EXTRA else "configs/config.json"
    config = json.loads(Path(template).read_text(encoding="utf-8"))
    config["model_name"] = MODEL_NAME
    config["data"]["training_files"] = str(DATASET / "train.list")
    config["data"]["validation_files"] = str(DATASET / "val.list")
    config["data"]["use_jp_extra"] = USE_JP_EXTRA
    config["train"]["batch_size"] = BATCH_SIZE
    config["train"]["epochs"] = EPOCHS
    config["train"]["eval_interval"] = SAVE_EVERY_STEPS
    config["train"]["log_interval"] = LOG_INTERVAL
    config["train"]["bf16_run"] = False
    (DATASET / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pretrained = Path("pretrained_jp_extra" if USE_JP_EXTRA else "pretrained")
    model_dir = DATASET / "models"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(pretrained, model_dir)
    print("pretrained ->", model_dir, sorted(p.name for p in model_dir.iterdir()))

    cfg_yml = Path("config.yml")
    yml = yaml.safe_load(Path("default_config.yml").read_text(encoding="utf-8"))
    yml["model_name"] = MODEL_NAME
    yml["dataset_path"] = str(DATASET)
    cfg_yml.write_text(yaml.dump(yml, allow_unicode=True), encoding="utf-8")
    print("wrote", DATASET / "config.json", "and", cfg_yml)


if __name__ == "__main__":
    main()
