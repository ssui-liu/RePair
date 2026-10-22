"""Run naming, logging setup and JSON persistence for training entry points."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.training.config import TrainingJobConfig

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def generate_run_id(config: "TrainingJobConfig") -> str:
    """e.g. `clip_ep10_bs64_lr2e-6_1022-0130` (+ `_sub<N>` when subsampling)."""
    training = config.training
    parts = [
        "clip",
        f"ep{training.num_train_epochs}",
        f"bs{training.per_device_train_batch_size}",
        f"lr{training.learning_rate:.0e}".replace("e-0", "e-"),
    ]
    if config.subsample_train:
        parts.append(f"sub{config.subsample_train}")
    parts.append(datetime.now().strftime("%m%d-%H%M"))
    return "_".join(parts)


def hard_pair_tag(config: "TrainingJobConfig") -> str:
    """Filesystem-safe name of the enabled hard-pair types, or `vanilla`."""
    raw = "+".join(config.hard_pairs.enabled) or "vanilla"
    return "".join(ch if ch.isalnum() or ch in "+-_" else "-" for ch in raw)


def _short_model_name(config: "TrainingJobConfig") -> str:
    return config.model.model_name.rstrip("/").split("/")[-1].replace("clip-", "")


def default_output_dir(project_root: Path, config: "TrainingJobConfig") -> Path:
    """results/<dataset>/<model>/train/<hard pairs>/<run_id>"""
    return (project_root / "results" / config.dataset.key / _short_model_name(config) / "train"
            / hard_pair_tag(config) / generate_run_id(config))


def default_wandb_run_name(config: "TrainingJobConfig") -> str:
    base = (f"{config.dataset.key}_{_short_model_name(config)}_{hard_pair_tag(config)}"
            f"_lambda{config.loss.grid_lambda:g}_{generate_run_id(config)}")
    tag = config.wandb.strategy_tag
    return f"{tag}_{base}" if tag else base


def setup_logging(output_dir: Optional[Path] = None, log_filename: str = "train.log") -> Optional[Path]:
    """Configure console logging once; optionally also write all records to output_dir/log_filename."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    root.setLevel(logging.INFO)
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (output_dir / log_filename).resolve()
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in root.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    return log_path


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
