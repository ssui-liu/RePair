"""
Training configuration: typed YAML sections, explicit CLI overrides, defaults.

Unknown keys are rejected, so a misspelled or stale hyperparameter fails loudly
instead of being silently ignored.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.hard_pairs import HARD_PAIR_TYPES
from src.data.karpathy import DATASET_KEYS
from src.training.runs import default_output_dir, default_wandb_run_name, hard_pair_tag


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    model_name: str = Field(..., description="HF model id or local path, e.g. openai/clip-vit-base-patch32")
    device: Optional[str] = Field(default=None, description="cuda/cpu; None selects automatically.")


class DatasetConfig(_Section):
    key: str = Field(default="flickr30k", description="flickr30k or coco30k")
    cache_dir: str = Field(default="data_cache", description="Root of the prepared caches.")

    @field_validator("key")
    @classmethod
    def _known_dataset(cls, key: str) -> str:
        if key not in DATASET_KEYS:
            raise ValueError(f"dataset.key must be one of {DATASET_KEYS}, got {key!r}")
        return key


class TrainingConfig(_Section):
    train_split: str = "train"
    eval_split: str = "val"
    output_dir: str = Field(..., description="Trainer output directory.")

    num_train_epochs: int = Field(default=1, ge=0)
    # Anchors per microbatch: the global InfoNCE candidate set. Gradient
    # accumulation averages microbatches but does not enlarge that set.
    per_device_train_batch_size: int = Field(default=32, ge=1)
    per_device_eval_batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    dataloader_num_workers: int = Field(default=0, ge=0)

    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_ratio: float = Field(default=0.0, ge=0.0)
    lr_scheduler_type: str = "linear"
    fp16: bool = False

    logging_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1, description="Validation retrieval every N optimizer steps.")
    save_total_limit: int = Field(default=2, ge=1)

    # Validation retrieval metrics (see src/evaluation/metrics.py).
    eval_metrics: Optional[List[str]] = None
    eval_top_k: Optional[List[int]] = None
    eval_tasks: Optional[List[str]] = None

    log_batch_stats: bool = Field(default=False, description="Log grid composition of every batch.")


class LossConfig(_Section):
    grid_lambda: float = Field(default=0.5, ge=0.0, description="Grid-loss weight lambda (paper Eq. 6: 0.5).")


def _check_hard_pair_types(names: Iterable[str]) -> None:
    unknown = sorted(set(names) - set(HARD_PAIR_TYPES))
    if unknown:
        raise ValueError(f"Unknown hard-pair types {unknown}; expected {list(HARD_PAIR_TYPES)}")


class HardPairsConfig(_Section):
    enabled: List[str] = Field(default_factory=list, description="repair-i2t / repair-t2i; empty = vanilla")
    paths: Dict[str, str] = Field(default_factory=dict, description="Optional records path per type.")

    @field_validator("enabled")
    @classmethod
    def _known_types(cls, names: List[str]) -> List[str]:
        _check_hard_pair_types(names)
        return names

    @field_validator("paths")
    @classmethod
    def _known_path_types(cls, paths: Dict[str, str]) -> Dict[str, str]:
        _check_hard_pair_types(paths)
        return paths

    def active(self) -> Dict[str, Optional[str]]:
        """Enabled types in order -> configured path (None: default cache path)."""
        return {name: self.paths.get(name) for name in self.enabled}


class SamplingConfig(_Section):
    pairs_per_direction: int = Field(default=1, ge=1, description="Hard pairs per direction and anchor (1: 3x3).")


class WandbConfig(_Section):
    enable: bool = False
    project: Optional[str] = None
    run_name: Optional[str] = None
    strategy_tag: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class TrainingJobConfig(_Section):
    """Everything `train_repair` needs."""

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    hard_pairs: HardPairsConfig = Field(default_factory=HardPairsConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    subsample_train: Optional[int] = None
    shuffle_subsample: bool = True
    seed: int = 42
    verbose: bool = True


# ---------------------------------------------------------------------------
# YAML + CLI
# ---------------------------------------------------------------------------

_SECTIONS = ("model", "dataset", "training", "loss", "hard_pairs", "sampling", "wandb")
_RUNTIME = ("subsample_train", "shuffle_subsample", "seed", "verbose")

# (argparse attribute, YAML section, YAML key); None CLI values keep the YAML value.
_OVERRIDES = (
    ("model_name", "model", "model_name"),
    ("device", "model", "device"),
    ("dataset_key", "dataset", "key"),
    ("data_cache", "dataset", "cache_dir"),
    ("output_dir", "training", "output_dir"),
    ("epochs", "training", "num_train_epochs"),
    ("batch_size", "training", "per_device_train_batch_size"),
    ("learning_rate", "training", "learning_rate"),
    ("weight_decay", "training", "weight_decay"),
    ("warmup_ratio", "training", "warmup_ratio"),
    ("lr_scheduler_type", "training", "lr_scheduler_type"),
    ("gradient_accumulation_steps", "training", "gradient_accumulation_steps"),
    ("dataloader_num_workers", "training", "dataloader_num_workers"),
    ("eval_steps", "training", "eval_steps"),
    ("save_steps", "training", "save_steps"),
    ("fp16", "training", "fp16"),
    ("log_batch_stats", "training", "log_batch_stats"),
    ("grid_lambda", "loss", "grid_lambda"),
    ("use_wandb", "wandb", "enable"),
    ("wandb_project", "wandb", "project"),
    ("wandb_run_name", "wandb", "run_name"),
    ("wandb_strategy_tag", "wandb", "strategy_tag"),
)


def _arg(args: Any, name: str) -> Any:
    return getattr(args, name, None)


def _apply_cli_overrides(raw: Dict[str, Any], args: Any) -> Dict[str, Any]:
    merged = copy.deepcopy(raw)
    unknown = sorted(set(merged) - set(_SECTIONS) - set(_RUNTIME))
    if unknown:
        raise ValueError(f"Unknown top-level training config keys: {unknown}")
    for section in _SECTIONS:
        merged[section] = dict(merged.get(section) or {})

    for attr, section, key in _OVERRIDES:
        if _arg(args, attr) is not None:
            merged[section][key] = _arg(args, attr)
    if _arg(args, "wandb_tags") is not None:
        merged["wandb"]["tags"] = [t for t in str(args.wandb_tags).split(",") if t]
    if _arg(args, "hard_pairs") is not None:
        merged["hard_pairs"]["enabled"] = [t.strip() for t in args.hard_pairs.split(",") if t.strip()]
    elif _arg(args, "vanilla"):
        merged["hard_pairs"]["enabled"] = []
    for name in _RUNTIME:
        if _arg(args, name) is not None:
            merged[name] = _arg(args, name)
    return merged


def build_job_config(*, raw_config: Dict[str, Any], args: Any, project_root: Path) -> TrainingJobConfig:
    """Validate YAML + CLI; resolve relative paths against `project_root` and fill defaults."""
    merged = _apply_cli_overrides(raw_config, args)
    if not merged["training"].get("output_dir"):
        merged["training"]["output_dir"] = "PENDING"  # filled below, once validated
    config = TrainingJobConfig.model_validate(merged)
    training, dataset = config.training, config.dataset
    if not Path(dataset.cache_dir).is_absolute():
        dataset.cache_dir = str(project_root / dataset.cache_dir)
    if training.output_dir == "PENDING":
        training.output_dir = str(default_output_dir(project_root, config))
    elif not Path(training.output_dir).is_absolute():
        training.output_dir = str(project_root / training.output_dir)

    wandb = config.wandb
    if wandb.enable:
        if not wandb.project or wandb.project.lower() == "auto":
            wandb.project = f"RePair-{dataset.key}"
        if not wandb.run_name:
            wandb.run_name = default_wandb_run_name(config)
        tags = [f"dataset:{dataset.key}", f"hard_pairs:{hard_pair_tag(config)}"]
        wandb.tags = list(dict.fromkeys([*wandb.tags, *tags]))
    return config


__all__ = ["TrainingJobConfig", "build_job_config"]
