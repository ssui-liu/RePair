"""
Train CLIP with RePair local 3x3 grids.

    python scripts/train.py --config config/training/repair_full.yaml \
        --seed 42 --output-dir checkpoints/repair-full-seed42

CLI values override the YAML; omitted flags keep it. The run directory receives
train.log, effective_config.json, cli_args.json, results.json and final/.
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import load_config  # noqa: E402
from src.training.config import build_job_config  # noqa: E402
from src.training.runs import save_json, setup_logging  # noqa: E402
from src.training.trainer import train_repair  # noqa: E402

logger = logging.getLogger(__name__)


def run_training(config, args):
    job_cfg = build_job_config(raw_config=config, args=args, project_root=project_root)
    output_dir = Path(job_cfg.training.output_dir)
    log_path = setup_logging(output_dir=output_dir)
    logger.info("Output directory: %s (log: %s)", output_dir, log_path)
    logger.info("Effective configuration:\n%s", job_cfg.model_dump_json(indent=2))
    save_json(output_dir / "effective_config.json", job_cfg.model_dump())
    save_json(output_dir / "cli_args.json", vars(args))

    results = train_repair(job_cfg)
    save_json(output_dir / "results.json", results)
    logger.info("Training complete: %s", output_dir / "final")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RePair local-grid CLIP training")
    bool_flag = argparse.BooleanOptionalAction
    parser.add_argument("--config", default=str(project_root / "config/training/repair_full.yaml"),
                        help="Training YAML")

    group = parser.add_argument_group("model and data")
    group.add_argument("--model-name", help="HF model id or local checkpoint")
    group.add_argument("--device", help="cuda or cpu (default: auto)")
    group.add_argument("--dataset-key", choices=["flickr30k", "coco30k"])
    group.add_argument("--data-cache", help="Root containing the dataset and RePair caches")
    hard_pairs = group.add_mutually_exclusive_group()
    hard_pairs.add_argument("--hard-pairs", help="Comma-separated hard-pair types, e.g. 'repair-i2t,repair-t2i'")
    hard_pairs.add_argument("--vanilla", action="store_true", default=None,
                            help="Original pairs only (no RePair grids)")
    group.add_argument("--subsample-train", type=int, help="Use this many training images")
    group.add_argument("--shuffle-subsample", action=bool_flag, help="Shuffle before subsampling (YAML default: on)")

    group = parser.add_argument_group("optimization")
    group.add_argument("--output-dir", help="Run directory")
    group.add_argument("--seed", type=int)
    group.add_argument("--epochs", type=int)
    group.add_argument("--batch-size", type=int, help="Anchors per microbatch (the InfoNCE candidate set)")
    group.add_argument("--gradient-accumulation-steps", type=int)
    group.add_argument("--learning-rate", type=float)
    group.add_argument("--weight-decay", type=float)
    group.add_argument("--warmup-ratio", type=float)
    group.add_argument("--lr-scheduler-type", help="e.g. linear, cosine")
    group.add_argument("--fp16", action=bool_flag)
    group.add_argument("--grid-lambda", type=float, help="Grid-loss weight lambda (paper: 0.5)")
    group.add_argument("--dataloader-num-workers", type=int, help="Worker processes for batch building")
    group.add_argument("--eval-steps", type=int, help="Validation retrieval every N optimizer steps")
    group.add_argument("--save-steps", type=int)

    group = parser.add_argument_group("logging")
    group.add_argument("--verbose", action=bool_flag, help="Progress bars while loading data")
    group.add_argument("--log-batch-stats", action=bool_flag, help="Log grid composition of every batch")
    group.add_argument("--use-wandb", action=bool_flag, help="Report to Weights & Biases")
    group.add_argument("--wandb-project")
    group.add_argument("--wandb-run-name")
    group.add_argument("--wandb-tags", help="Comma-separated tags")
    group.add_argument("--wandb-strategy-tag", help="Run-name prefix")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    setup_logging()
    logger.info("Loaded config: %s", args.config)
    run_training(load_config(args.config), args)


if __name__ == "__main__":
    main()
