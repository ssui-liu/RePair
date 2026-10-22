"""
Collect successful RePair I2T synthesis items into the training parquet.

    python scripts/prepare_i2t.py --datasets flickr30k \
        --run-name gtfirst_m3_qc1_gtqc1_fpqc1_pqc1_gemini-2.5-flash

Reads `<input-root>/<dataset>/<retriever>/synthesis/train/i2t/<run-name>/<status>/*/metadata.json`
and writes `<output-root>/<dataset>/I2T/synthesis_success.parquet` with columns
query_imgid, item_id, fp_index, gt_rank, gt_sentid, fp_sentid, edited_caption.
Each record is checked against the FP pool that produced it (FP sentid and text);
an existing parquet file is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.fp_pool import find_fp_pool, load_fp_pool  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Work-item ids: `<query_imgid>_fp<fp_index>`.
_ITEM_ID = re.compile(r"(\d+)_fp(\d+)$")
SCHEMA = pa.schema([
    ("query_imgid", pa.int64()), ("item_id", pa.string()), ("fp_index", pa.int64()), ("gt_rank", pa.int64()),
    ("gt_sentid", pa.int64()), ("fp_sentid", pa.int64()), ("edited_caption", pa.string()),
])


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _pool_index(pool: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(sample["query_imgid"]): {
            "gt_sentid": sample["ground_truth"]["best_sentid"],
            "gt_rank": sample["ground_truth"]["best_rank"],
            "fp_sentids": sample["false_positives"]["sentids"],
            "fp_texts": sample["false_positives"]["texts"],
        }
        for sample in pool["samples"]
    }


def _record(item_dir: Path, pool: Dict[int, Dict[str, Any]], verify_text: bool) -> Optional[Dict[str, Any]]:
    """Training record of one item directory, or None (with a warning) if it fails a check."""
    match = _ITEM_ID.match(item_dir.name)
    if not match:
        logger.warning("Skipping %s: not a work-item id", item_dir.name)
        return None
    query_imgid, fp_index = int(match.group(1)), int(match.group(2))
    entry = pool.get(query_imgid)
    if entry is None or fp_index >= len(entry["fp_sentids"]):
        logger.warning("Skipping %s: query/FP not in the FP pool", item_dir.name)
        return None
    metadata_path = item_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning("Skipping %s: no metadata.json", item_dir.name)
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fp_sentid = int(entry["fp_sentids"][fp_index])
    if metadata.get("fp_sentid") is not None and int(metadata["fp_sentid"]) != fp_sentid:
        logger.warning("Skipping %s: fp_sentid %s does not match the FP pool (%s)",
                       item_dir.name, metadata["fp_sentid"], fp_sentid)
        return None
    if verify_text and _normalize(metadata.get("fp_caption") or "") != _normalize(entry["fp_texts"][fp_index]):
        logger.warning("Skipping %s: FP text differs from the FP pool", item_dir.name)
        return None
    edited_caption = str(metadata.get("edited_caption") or "").strip()
    if not edited_caption:
        logger.warning("Skipping %s: no edited_caption", item_dir.name)
        return None
    return {
        "query_imgid": query_imgid,
        "item_id": item_dir.name,
        "fp_index": fp_index,
        "gt_rank": int(entry["gt_rank"]),
        "gt_sentid": int(entry["gt_sentid"]),
        "fp_sentid": fp_sentid,
        "edited_caption": edited_caption,
    }


def convert_dataset(
    input_root: Path,
    output_root: Path,
    dataset: str,
    run_name: str,
    fp_pool: Optional[str] = None,
    retriever: str = "vit-base-patch32",
    status: str = "success",
    batch_size: int = 1000,
    compression: str = "zstd",
    text_verify_enabled: bool = True,
) -> Dict[str, Any]:
    """Convert one dataset's run; returns its manifest.

    `fp_pool` is relative to `<input_root>/<dataset>/<retriever>/` (default: the
    training-split I2T pool, `retrieval/train/i2t_fp_pool.json`).
    """
    input_root, output_root = Path(input_root), Path(output_root)
    run_dir = input_root / dataset / retriever / "synthesis" / "train" / "i2t" / run_name / status
    pool_path = (input_root / dataset / retriever / fp_pool if fp_pool
                 else find_fp_pool(input_root, dataset, retriever, "train", "i2t"))
    out_dir = output_root / dataset / "I2T"
    parquet_file = out_dir / f"synthesis_{status}.parquet"
    if parquet_file.exists():
        raise FileExistsError(f"Refusing to overwrite: {parquet_file}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Synthesis items not found: {run_dir}")
    pool = _pool_index(load_fp_pool(pool_path))
    item_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    if not item_dirs:
        raise ValueError(f"No items under {run_dir}")
    logger.info("Converting %d %s items of %s/%s", len(item_dirs), status, dataset, run_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = parquet_file.with_suffix(".parquet.tmp")
    rows: List[Dict[str, Any]] = []
    num_records = num_skipped = 0
    try:
        with pq.ParquetWriter(temporary.as_posix(), schema=SCHEMA, use_dictionary=True,
                              compression=None if compression == "none" else compression) as writer:
            for item_dir in tqdm(item_dirs, desc=f"Converting {dataset}", unit="item"):
                record = _record(item_dir, pool, text_verify_enabled)
                if record is None:
                    num_skipped += 1
                    continue
                rows.append(record)
                num_records += 1
                if len(rows) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                    rows = []
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        if not num_records:
            raise ValueError("No valid synthesis records; check IDs and FP-pool provenance")
        temporary.replace(parquet_file)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "dataset": dataset,
        "run_name": run_name,
        "status": status,
        "num_records": num_records,
        "num_skipped": num_skipped,
        "parquet_file": parquet_file.name,
        "input_path": str(run_dir.relative_to(input_root)),
        "fp_pool": str(pool_path),
        "schema": {field.name: str(field.type) for field in SCHEMA},
        "text_verify_enabled": text_verify_enabled,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("%s: %d records written to %s (%d skipped)", dataset, num_records, parquet_file, num_skipped)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RePair I2T synthesis items into the training parquet")
    parser.add_argument("--input-root", default="results", help="Root of FP pools and synthesis outputs")
    parser.add_argument("--output-root", default="data_cache/RePair", help="Root of the hard-pair caches")
    parser.add_argument("--datasets", nargs="+", default=["flickr30k"])
    parser.add_argument("--run-name", required=True,
                        help="Run directory printed by synthesis, e.g. gtfirst_m3_qc1_gtqc1_fpqc1_pqc1_<model>")
    parser.add_argument("--retriever", default="vit-base-patch32", help="Retriever directory under results/<dataset>/")
    parser.add_argument("--fp-pool", help="FP pool relative to <input-root>/<dataset>/<retriever>/ "
                                          "(default: retrieval/train/i2t_fp_pool.json)")
    parser.add_argument("--status", default="success", help="Item status directory to convert")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per parquet write")
    parser.add_argument("--compression", default="zstd", choices=["snappy", "gzip", "brotli", "zstd", "lz4", "none"])
    parser.add_argument("--skip-text-verify", action="store_true", help="Skip the FP text check (debugging only)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    failed = []
    for dataset in args.datasets:
        try:
            convert_dataset(input_root=Path(args.input_root), output_root=Path(args.output_root), dataset=dataset,
                            run_name=args.run_name, fp_pool=args.fp_pool, retriever=args.retriever,
                            status=args.status, batch_size=args.batch_size, compression=args.compression,
                            text_verify_enabled=not args.skip_text_verify)
        except Exception:
            logger.exception("Conversion failed for %s", dataset)
            failed.append(dataset)
    if failed:
        raise RuntimeError(f"Conversion failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
