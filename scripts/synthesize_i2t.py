#!/usr/bin/env python
"""
RePair I2T synthesis (paper Algorithm 1, caption side).

For every (query image, FP caption) pair of the I2T FP pool: one joint QC call,
one edit instruction G(q, p_fp), M caption edits and CLIP-cosine selection (PQC).

    # Resolved, redacted config only: no API calls, model loading or output writes
    python scripts/synthesize_i2t.py --config config/synthesis/i2t.yaml --dry-run

    # Paid API calls; the key comes from GEMINI_API_KEY (or OPENAI_API_KEY)
    python scripts/synthesize_i2t.py --config config/synthesis/i2t.yaml --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

_REQUIRED = ("dataset.key", "dataset.split", "dataset.cache_dir", "paths.results_dir", "paths.retriever",
             "api.text_model", "api.vision_model")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RePair I2T synthesis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default="config/synthesis/i2t.yaml", help="Synthesis YAML")

    group = parser.add_argument_group("work items")
    group.add_argument("--max-queries", "-n", type=int, help="Queries to process (worst GT rank first)")
    group.add_argument("--fps-per-query", "-f", type=int, help="FPs retained per query (paper d_I2T=5)")
    group.add_argument("--fp-top-pct", type=float, help="Retain ceil(pool * pct) FPs per query instead (0 < pct <= 1)")
    group.add_argument("--gt-caption-mode", choices=["first", "last", "all"], help="GT caption(s) shown to QC")
    group.add_argument("--num-candidates", "-m", type=int, help="Edit candidates M per pair (paper: 3)")
    group.add_argument("--max-pairs", type=int, help="Global cap on work items")

    group = parser.add_argument_group("quality control")
    group.add_argument("--no-qc", action="store_true", help="Skip QC (ablation)")
    group.add_argument("--no-gt-qc", action="store_true", help="Skip criterion 1 (GT matches the image)")
    group.add_argument("--no-fp-qc", action="store_true", help="Skip criterion 2 (FP is not already valid)")
    group.add_argument("--no-pqc", action="store_true", help="Take the first edit instead of cosine selection")

    group = parser.add_argument_group("API")
    group.add_argument("--vision-model", help="Model for QC and instructions (image input)")
    group.add_argument("--text-model", help="Model for caption edits")
    group.add_argument("--base-url", help="OpenAI-compatible endpoint (default: config or OPENAI_BASE_URL)")
    group.add_argument("--max-retries", type=int, help="Retries of transient API failures")
    group.add_argument("--parallel", "-p", action="store_true", help="Process items in a thread pool")
    group.add_argument("--workers", "-w", type=int, help="Threads in parallel mode")

    group = parser.add_argument_group("paths and outputs")
    group.add_argument("--results-dir", help="Root of FP pools and synthesis outputs")
    group.add_argument("--data-cache", help="Root of the dataset caches")
    group.add_argument("--split", help="Dataset split")
    group.add_argument("--dataset-key", choices=["flickr30k", "coco30k"])
    group.add_argument("--retriever", help="Retriever directory under results/<dataset>/ (e.g. vit-base-patch32)")
    group.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=None,
                       help="Skip items that already have an output directory")
    group.add_argument("--resume-processed-statuses",
                       help="Statuses treated as done when resuming, e.g. 'success,qc_rejected' to re-run failures")
    group.add_argument("--save-query-images", action=argparse.BooleanOptionalAction, default=None,
                       help="Also save the query image of successful items")
    group.add_argument("--dry-run", action="store_true",
                       help="Print the resolved config; no API calls, model downloads or output writes")
    group.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    group.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def merge_args_with_config(config: dict, args) -> dict:
    """A copy of the YAML config with the explicitly given CLI values applied."""
    config = copy.deepcopy(config)
    for section in ("api", "synthesis", "quality_control", "parallel", "retry", "paths", "dataset", "logging"):
        config.setdefault(section, {})
    overrides = {
        ("synthesis", "max_queries"): args.max_queries,
        ("synthesis", "fps_per_query"): args.fps_per_query,
        ("synthesis", "fp_top_pct"): args.fp_top_pct,
        ("synthesis", "gt_caption_mode"): args.gt_caption_mode,
        ("synthesis", "num_candidates"): args.num_candidates,
        ("synthesis", "max_pairs"): args.max_pairs,
        ("synthesis", "resume"): args.resume,
        ("synthesis", "save_query_images"): args.save_query_images,
        ("api", "vision_model"): args.vision_model,
        ("api", "text_model"): args.text_model,
        ("api", "base_url"): args.base_url or os.environ.get("OPENAI_BASE_URL") or None,
        ("parallel", "max_workers"): args.workers,
        ("retry", "max_retries"): args.max_retries,
        ("paths", "results_dir"): args.results_dir,
        ("paths", "retriever"): args.retriever,
        ("dataset", "cache_dir"): args.data_cache,
        ("dataset", "split"): args.split,
        ("dataset", "key"): args.dataset_key,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.resume_processed_statuses is not None:
        config["synthesis"]["resume_processed_statuses"] = [
            s.strip() for s in args.resume_processed_statuses.split(",") if s.strip()
        ]
    flags = {
        ("quality_control", "enable_qc"): (args.no_qc, False),
        ("quality_control", "enable_gt_check"): (args.no_gt_qc, False),
        ("quality_control", "enable_fp_check"): (args.no_fp_qc, False),
        ("quality_control", "enable_pqc"): (args.no_pqc, False),
        ("parallel", "enable"): (args.parallel, True),
        ("logging", "show_progress"): (args.no_progress, False),
    }
    for (section, key), (given, value) in flags.items():
        if given:
            config[section][key] = value
    return config


def _require(cfg: dict, dotted_key: str) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing required config: {dotted_key}")
        value = value[part]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise KeyError(f"Missing required config: {dotted_key}")
    return value


def _get(section: dict, key: str, default: Any) -> Any:
    value = section.get(key)
    return default if value is None else value


def _from_root(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else project_root / path


def build_api_client(config: dict):
    """Text and vision chat clients behind one retry layer."""
    from src.synthesis.api import APIClient, ChatClient, ChatConfig
    from src.synthesis.retry import RetryConfig

    api_cfg, retry_cfg = config["api"], config["retry"]
    api_key = api_cfg.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Set GEMINI_API_KEY (or OPENAI_API_KEY for a compatible provider).")
    common = dict(api_key=api_key, base_url=api_cfg.get("base_url"),
                  temperature=_get(api_cfg, "temperature", 0.2),
                  timeout_seconds=_get(api_cfg, "timeout_seconds", 120.0))
    retry = RetryConfig(max_retries=_get(retry_cfg, "max_retries", 3),
                        initial_delay=_get(retry_cfg, "initial_delay", 2.0),
                        backoff_factor=_get(retry_cfg, "backoff_factor", 2.0),
                        max_delay=_get(retry_cfg, "max_delay", 60.0))
    logger.info("Retry: max_retries=%d, initial_delay=%.1fs", retry.max_retries, retry.initial_delay)
    return APIClient(
        text_client=ChatClient(ChatConfig(model=str(api_cfg["text_model"]), **common)),
        vision_client=ChatClient(ChatConfig(model=str(api_cfg["vision_model"]),
                                            max_tokens=_get(api_cfg, "max_tokens", 1024), **common)),
        retry_config=retry,
    )


def run(config: dict, api_client=None, cosine_selector=None) -> Dict[str, int]:
    """Run synthesis on a merged config; `api_client` / `cosine_selector` can be injected (tests)."""
    from src.data.karpathy import KarpathyCache, metadata_column
    from src.evaluation.fp_pool import find_fp_pool, load_fp_pool
    from src.synthesis.api import TokenTracker
    from src.synthesis.i2t import CosineSelector, I2TWorkflow
    from src.synthesis.runner import BatchRunner
    from src.synthesis.store import ResultStore
    from src.synthesis.work_items import WorkItemBuilder

    synthesis, qc = config["synthesis"], config["quality_control"]
    max_queries = int(_get(synthesis, "max_queries", 100))
    fps_per_query = int(_get(synthesis, "fps_per_query", 1))
    gt_caption_mode = str(_get(synthesis, "gt_caption_mode", "first"))
    num_candidates = int(_get(synthesis, "num_candidates", 1))
    enable_qc = bool(_get(qc, "enable_qc", True))
    enable_gt_check = bool(_get(qc, "enable_gt_check", True))
    enable_fp_check = bool(_get(qc, "enable_fp_check", True))
    enable_pqc = bool(_get(qc, "enable_pqc", True))
    statuses = synthesis.get("resume_processed_statuses")
    if isinstance(statuses, str):
        statuses = [s.strip() for s in statuses.split(",") if s.strip()]
    workers = int(_get(config["parallel"], "max_workers", 2)) if config["parallel"].get("enable") else 1
    logger.info("I2T synthesis: queries<=%d, FPs/query=%s, M=%d, QC=%s (gt=%s, fp=%s), PQC=%s, workers=%d",
                max_queries, synthesis.get("fp_top_pct") or fps_per_query, num_candidates,
                enable_qc, enable_gt_check, enable_fp_check, enable_pqc, workers)
    api_client = api_client or build_api_client(config)  # validates the key before any output is written

    dataset_key, split = str(_require(config, "dataset.key")), str(_require(config, "dataset.split"))
    dataset = KarpathyCache(dataset_key, str(_from_root(_require(config, "dataset.cache_dir")))).load_split(split)
    imgids = [int(i) for i in metadata_column(dataset, "imgid")]
    if len(set(imgids)) != len(imgids):
        raise ValueError("Duplicate imgid in the dataset split")
    imgid_to_idx = {imgid: row for row, imgid in enumerate(imgids)}
    sentid_to_text = {
        int(sentid): str(caption)
        for captions, sentids in zip(metadata_column(dataset, "caption"), metadata_column(dataset, "sentids"),
                                      strict=True)
        for sentid, caption in zip(sentids, captions, strict=True)
    }

    results_dir = _from_root(_require(config, "paths.results_dir"))
    retriever = str(_require(config, "paths.retriever"))
    pool = load_fp_pool(find_fp_pool(results_dir, dataset_key, retriever, split, "i2t"))
    metadata = pool.get("metadata", {})
    if metadata.get("mining_strategy") != "top_k_non_gt":
        raise ValueError(f"FP pool has mining_strategy={metadata.get('mining_strategy')!r}, expected "
                         "'top_k_non_gt': regenerate it with scripts/retrieve.py --save-fp-pools")
    if fps_per_query > int(metadata.get("mining_top_k", 0)):
        raise ValueError("fps_per_query exceeds the mining_top_k of the FP pool")

    work_items = WorkItemBuilder(
        max_queries=max_queries,
        fps_per_query=fps_per_query,
        fp_top_pct=synthesis.get("fp_top_pct"),
        max_pairs=synthesis.get("max_pairs"),
        gt_caption_mode=gt_caption_mode,
    ).build(pool["samples"], sentid_to_text)
    if not work_items:
        logger.warning("No work items found")
        return {}

    store = ResultStore(
        output_dir=results_dir / dataset_key / retriever / "synthesis" / split,
        task_type="i2t",
        max_queries=max_queries,
        fps_per_query=fps_per_query,
        gt_caption_mode=gt_caption_mode,
        num_candidates=num_candidates,
        enable_qc=enable_qc,
        enable_gt_check=enable_gt_check,
        enable_fp_check=enable_fp_check,
        enable_pqc=enable_pqc,
        model_name=str(config["api"]["text_model"]),  # one model name in the path (text == vision here)
        config=config,
        resume=bool(_get(synthesis, "resume", True)),
        save_query_images=bool(_get(synthesis, "save_query_images", False)),
        resume_processed_statuses=statuses,
    )
    if cosine_selector is None and enable_pqc and num_candidates > 1:
        cosine_selector = CosineSelector()
    workflow = I2TWorkflow(
        api_client=api_client,
        sentid_to_text=sentid_to_text,
        imgid_to_idx=imgid_to_idx,
        enable_qc=enable_qc,
        enable_gt_check=enable_gt_check,
        enable_fp_check=enable_fp_check,
        enable_pqc=enable_pqc,
        num_candidates=num_candidates,
        cosine_selector=cosine_selector,
    )
    tracker = getattr(api_client, "token_tracker", None) or TokenTracker()
    runner = BatchRunner(workflow, store, dataset, token_tracker=tracker)
    counts = runner.run(work_items, max_workers=workers,
                        show_progress=bool(_get(config["logging"], "show_progress", True)))

    tokens = tracker.totals()
    store.save_summary(token_usage=tokens)
    logger.info("This invocation: %s | run totals: %s | tokens: %s", counts, store.statistics(), tokens)
    logger.info("Output directory: %s", store.run_dir)
    return counts


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    config = merge_args_with_config(load_config(args.config), args)
    for key in _REQUIRED:
        _require(config, key)
    if int(_get(config["synthesis"], "num_candidates", 1)) < 1:
        raise ValueError("num_candidates must be positive")
    if args.dry_run:
        from src.synthesis.store import redact_secrets

        print(json.dumps(redact_secrets(config), indent=2))
        return 0
    # httpx logs one INFO line per request.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
