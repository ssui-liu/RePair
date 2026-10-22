"""Import already selected RePair T2I edits (JSONL + image files) into the training parquet.

Each JSONL record names a text query, its FP and GT images and the selected edited
image: {"query_sentid", "fp_imgid", "gt_imgid", "edited_image_path"}. Output columns:
query_sentid, fp_imgid, gt_imgid, edited_image. This is an artifact importer, not a
Flux image editor or QC/PQC implementation.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

RECORD_KEYS = ("query_sentid", "fp_imgid", "gt_imgid", "edited_image_path")


def prepare_t2i(records_path, annotations, output, shard_size=256):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    records_path, annotations, output = map(Path, (records_path, annotations, output))
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing T2I cache: {output}")
    source = json.loads(annotations.read_text(encoding="utf-8"))
    train = [row for row in source["images"] if row["split"] == "train"]
    image_ids = {int(row["imgid"]) for row in train}
    caption_owner = {int(s["sentid"]): int(row["imgid"]) for row in train for s in row["sentences"]}
    schema = pa.schema([("query_sentid", pa.int64()), ("fp_imgid", pa.int64()),
                        ("gt_imgid", pa.int64()), ("edited_image", pa.binary())])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".t2i-", dir=output.parent))
    count, shard = 0, 0
    batch, seen = [], set()
    try:
        with records_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = [key for key in RECORD_KEYS if row.get(key) is None]
                if missing:
                    raise ValueError(f"Line {line_number}: missing {missing}")
                query, fp, gt = (int(row[key]) for key in RECORD_KEYS[:3])
                if caption_owner.get(query) != gt or fp not in image_ids or fp == gt:
                    raise ValueError(f"Line {line_number}: invalid training query/GT/FP IDs")
                if (query, fp) in seen:
                    raise ValueError(f"Line {line_number}: duplicate selected query/FP pair")
                seen.add((query, fp))
                image_path = Path(row["edited_image_path"])
                if not image_path.is_absolute():
                    image_path = records_path.parent / image_path
                with Image.open(image_path) as image:
                    image.verify()
                batch.append({"query_sentid": query, "fp_imgid": fp, "gt_imgid": gt,
                              "edited_image": image_path.read_bytes()})
                count += 1
                if len(batch) == shard_size:
                    pq.write_table(pa.Table.from_pylist(batch, schema=schema),
                                   staging / f"part-{shard:05d}.parquet")
                    batch, shard = [], shard + 1
        if not count:
            raise ValueError("No selected T2I records to import")
        if batch:
            pq.write_table(pa.Table.from_pylist(batch, schema=schema),
                           staging / f"part-{shard:05d}.parquet")
        manifest = {"records": count, "source_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                    "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
                    "qc_pqc": "external; input records must already be selected and verified"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="JSONL of selected edits")
    parser.add_argument("--annotations", type=Path, required=True, help="Prepared Karpathy JSON")
    parser.add_argument("--output", type=Path, required=True, help="New RePair/<dataset>/T2I directory")
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare_t2i(args.records, args.annotations, args.output, args.shard_size), indent=2))


if __name__ == "__main__":
    main()
