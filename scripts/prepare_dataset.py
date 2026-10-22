"""Build local Karpathy parquet caches from user-supplied images/annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path


def prepare_dataset(annotations, images_root, output_root, dataset_key,
                    train_size=None, seed=42, train_image_ids=None, shard_size=256,
                    keep_annotation_splits=False):
    """Write <output_root>/<dataset_key>/ and return its manifest.

    The training images are, in order of precedence: the annotation's own train
    split (`keep_annotation_splits`), an explicit ID list (`train_image_ids`), a
    seeded sample of `train_size` images (COCO30K default: 29,000 COCO 2014 train
    images), or the whole train split.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if keep_annotation_splits and (train_size is not None or train_image_ids is not None):
        raise ValueError("keep_annotation_splits cannot be combined with train_size or train_image_ids")
    annotations, images_root, output_root = map(Path, (annotations, images_root, output_root))
    images_root = images_root.resolve()
    source = json.loads(annotations.read_text(encoding="utf-8"))
    records = source["images"]
    if dataset_key not in {"flickr30k", "coco30k"}:
        raise ValueError("dataset_key must be flickr30k or coco30k")
    if len({int(r["imgid"]) for r in records}) != len(records):
        raise ValueError("Karpathy imgid values must be unique")
    if len({r["filename"] for r in records}) != len(records):
        raise ValueError("Image filenames must be unique")
    all_sentids = [int(s["sentid"]) for r in records for s in r["sentences"]]
    if len(set(all_sentids)) != len(all_sentids):
        raise ValueError("Karpathy sentid values must be unique across splits")
    split_rows = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    # Appendix A: COCO30K comes from COCO 2014 train images, never val/test/restval.
    if dataset_key == "coco30k" and not keep_annotation_splits:
        split_rows["train"] = [r for r in split_rows["train"]
                               if "train2014" in r.get("filepath", "")
                               or "train2014" in r["filename"]]
        train_size = 29000 if train_size is None else train_size
    eligible = split_rows["train"]
    if train_image_ids is not None:
        requested = [int(i) for i in json.loads(Path(train_image_ids).read_text(encoding="utf-8"))]
        if len(set(requested)) != len(requested):
            raise ValueError("Training ID manifest contains duplicates")
        eligible_ids = {int(r["imgid"]) for r in eligible}
        if not set(requested) <= eligible_ids:
            raise ValueError("Training ID manifest contains IDs outside the eligible training pool")
        keep = set(requested)
    elif train_size is not None:
        if not 0 < train_size <= len(eligible):
            raise ValueError(f"train_size must be between 1 and {len(eligible)}")
        keep = set(random.Random(seed).sample([int(r["imgid"]) for r in eligible], train_size))
    else:
        keep = {int(r["imgid"]) for r in eligible}
    split_rows["train"] = [r for r in eligible if int(r["imgid"]) in keep]
    if any(not rows for rows in split_rows.values()):
        raise ValueError("Annotations must provide nonempty train, val and test splits")
    destination = output_root / dataset_key
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing cache: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_key}-", dir=output_root))
    schema = pa.schema([("filename", pa.string()), ("image_bytes", pa.binary())])
    try:
        kept_records = []
        manifest = {"dataset": dataset_key, "sampling_seed": seed,
                    "sampling_algorithm": "python.random.Random.sample; source JSON order",
                    "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
                    "annotation_splits_kept": bool(keep_annotation_splits),
                    "training_ids_supplied": train_image_ids is not None, "splits": {}}
        for split, rows in split_rows.items():
            part_dir = staging / split / "images_parquet"
            part_dir.mkdir(parents=True)
            sentids = [int(s["sentid"]) for r in rows for s in r["sentences"]]
            if len(set(sentids)) != len(sentids) or any(not r["sentences"] for r in rows):
                raise ValueError(f"Invalid caption IDs or empty caption list in {split}")
            manifest["splits"][split] = {"images": len(rows), "captions": len(sentids),
                                          "image_ids": [int(r["imgid"]) for r in rows]}
            for offset in range(0, len(rows), shard_size):
                packed = []
                for r in rows[offset:offset + shard_size]:
                    candidates = [images_root / r.get("filepath", "") / r["filename"],
                                  images_root / r["filename"]]
                    image_path = next((p.resolve() for p in candidates if p.is_file()), None)
                    if image_path is None:
                        raise FileNotFoundError(f"Missing image: {r['filename']}")
                    image_path.relative_to(images_root)
                    packed.append({"filename": r["filename"], "image_bytes": image_path.read_bytes()})
                pq.write_table(pa.Table.from_pylist(packed, schema=schema),
                               part_dir / f"part-{offset // shard_size:05d}.parquet")
            kept_records.extend(rows)
        output_json = dict(source, images=kept_records)
        (staging / f"dataset_{dataset_key}.json").write_text(json.dumps(output_json, ensure_ascii=False),
                                                             encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / "train_image_ids.json").write_text(json.dumps(manifest["splits"]["train"]["image_ids"]),
                                                      encoding="utf-8")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data_cache"))
    parser.add_argument("--dataset-key", choices=["flickr30k", "coco30k"], required=True)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--train-image-ids", type=Path, help="JSON list of original Karpathy imgids")
    parser.add_argument("--keep-annotation-splits", action="store_true",
                        help="Use the train/val/test membership of --annotations as-is (no sampling or "
                             "COCO train2014 filter)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    manifest = prepare_dataset(args.annotations, args.images_root, args.output_root,
                               args.dataset_key, args.train_size, args.seed,
                               args.train_image_ids, args.shard_size, args.keep_annotation_splits)
    print(json.dumps({k: {"images": v["images"], "captions": v["captions"]}
                      for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
