"""
Full-ranking retrieval evaluation and FP-pool export (paper Eq. 1).

    # Test-split metrics of a checkpoint (written to <checkpoint>/../retrieval/test/)
    python scripts/retrieve.py --model-name checkpoints/run/final --split test

    # FP pools for synthesis: top-K non-GT candidates of every training query
    python scripts/retrieve.py --split train --save-fp-pools --mining-top-k 100
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import load_config  # noqa: E402
from src.data.feature_extractor import CLIPFeatureExtractor  # noqa: E402
from src.data.karpathy import DATASET_KEYS, KarpathyCache  # noqa: E402
from src.evaluation.evaluator import RetrievalEvaluator  # noqa: E402
from src.evaluation.fp_pool import fp_pool_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _from_root(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else project_root / path


def _existing_or_name(name: str) -> str:
    """Resolve a repo-relative local path; leave Hugging Face model ids unchanged."""
    candidate = project_root / name
    return str(candidate) if not Path(name).is_absolute() and candidate.exists() else name


def run_retrieval(config, args):
    dataset_cfg, model_cfg, eval_cfg = config["dataset"], config["model"], config["evaluation"]
    dataset_key = args.dataset_key or dataset_cfg.get("key")
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"Set dataset.key to one of {DATASET_KEYS} or pass --dataset-key (got {dataset_key!r}).")
    mining_top_k = args.mining_top_k if args.mining_top_k is not None else eval_cfg.get("mining_top_k", 100)
    if int(mining_top_k) < 1:
        raise ValueError("mining_top_k must be positive")
    cache_dir = _from_root(args.data_cache or dataset_cfg.get("cache_dir") or "data_cache")
    if not (cache_dir / dataset_key).exists():
        raise FileNotFoundError(f"Local cache not found: {cache_dir / dataset_key} (see scripts/prepare_dataset.py)")

    split = args.split or dataset_cfg["split"]
    cache = KarpathyCache(dataset_key, str(cache_dir))
    dataset = cache.load_split(split)
    if args.subsample and 0 < args.subsample < len(dataset):
        if args.shuffle:
            dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(args.subsample))
    all_captions, caption_to_image, id_mapping = cache.caption_mapping(dataset)
    logger.info("%s/%s: %d images, %d captions", dataset_key, split, len(dataset), len(all_captions))

    model_name = _existing_or_name(args.model_name or model_cfg["model_name"])
    processor_name = args.processor_name or model_cfg.get("processor_name")
    extractor = CLIPFeatureExtractor(
        model_name=model_name,
        processor_name=_existing_or_name(processor_name) if processor_name else None,
        device=args.device or model_cfg.get("device"),
        batch_size=args.batch_size or model_cfg["batch_size"],
    )
    image_features = extractor.encode_images(dataset)
    caption_features = extractor.encode_texts(all_captions)

    evaluator = RetrievalEvaluator(
        metrics_list=eval_cfg.get("metrics"),
        top_k=eval_cfg.get("top_k"),
        tasks=args.eval_tasks or eval_cfg.get("eval_tasks"),
        use_tqdm=not args.no_tqdm,
    )

    # A local checkpoint writes next to itself; a hub model under results/<dataset>/<model>/.
    if Path(model_name).is_dir():
        output_dir = Path(model_name).parent / "retrieval" / split
    else:
        retriever = model_name.split("/")[-1].replace("clip-", "")
        output_dir = project_root / "results" / dataset_key / retriever / "retrieval" / split
    override_dir = args.output_dir or eval_cfg.get("output_dir")
    if override_dir:
        output_dir = _from_root(override_dir)
        if output_dir.name != split and split not in output_dir.parts:
            output_dir = output_dir / split

    fp_pools_only = bool(args.fp_pools_only)
    save_fp_pools = fp_pools_only or (
        args.save_fp_pools if args.save_fp_pools is not None else eval_cfg.get("save_fp_pools", False)
    )
    save_tail = eval_cfg.get("save_tail", False)
    tail_dir = output_dir / "tail_t2i"
    if eval_cfg.get("tail_output_dir"):
        tail_dir = _from_root(eval_cfg["tail_output_dir"])
    if save_fp_pools:
        logger.info("FP pools: %s", output_dir)

    results = evaluator.evaluate(
        image_features=image_features,
        caption_features=caption_features,
        caption_to_image=caption_to_image,
        model_name=model_name,
        dataset=dataset,
        all_captions=all_captions,
        compute_metrics=not fp_pools_only,
        save_tail=save_tail,
        tail_k=eval_cfg.get("tail_k", 10),
        tail_output_dir=str(tail_dir),
        save_fp_pools=save_fp_pools,
        i2t_pool_path=str(fp_pool_path(output_dir, "i2t")),
        t2i_pool_path=str(fp_pool_path(output_dir, "t2i")),
        dataset_name=dataset_key,
        split=split,
        id_mapping=id_mapping,
        mining_top_k=int(mining_top_k),
    )

    if not fp_pools_only and not args.no_save and eval_cfg.get("save_results", True):
        results_path = args.results_path or eval_cfg.get("results_path")
        results_path = _from_root(results_path) if results_path else output_dir / "evaluation_results.json"
        evaluator.save_results(results, str(results_path))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full-ranking retrieval evaluation and FP-pool export")
    parser.add_argument("--config", default=str(project_root / "config/retrieval.yaml"))
    parser.add_argument("--split", help="train | val | test (default: dataset.split)")
    parser.add_argument("--dataset-key", choices=["flickr30k", "coco30k"])
    parser.add_argument("--data-cache", help="Root containing the dataset caches")
    parser.add_argument("--subsample", type=int, default=0, help="Use only this many images (0 = all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before subsampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", help="HF model id or local checkpoint directory")
    parser.add_argument("--processor-name", help="Processor source (default: the model)")
    parser.add_argument("--device", help="cuda | cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, help="Encoding batch size")
    parser.add_argument("--eval-tasks", nargs="+", help="t2i and/or i2t (default: config)")
    parser.add_argument("--mining-top-k", type=int, help="FP-pool size per query (paper K_mine=100)")
    parser.add_argument("--save-fp-pools", action=argparse.BooleanOptionalAction, default=None,
                        help="Write both FP pools next to the metrics")
    parser.add_argument("--fp-pools-only", action="store_true", default=None,
                        help="Write the FP pools only; skip metrics and evaluation_results.json")
    parser.add_argument("--results-path", help="Metrics JSON path (default: <output dir>/evaluation_results.json)")
    parser.add_argument("--output-dir", help="Output directory (the split is appended)")
    parser.add_argument("--no-save", action="store_true", help="Do not write evaluation_results.json")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger.info("Arguments: %s", vars(args))
    run_retrieval(load_config(args.config), args)


if __name__ == "__main__":
    main()
