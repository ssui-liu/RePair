"""
Local Karpathy-split image caches (Flickr30K / COCO30K) stored as parquet shards.

Cache layout (written by `scripts/prepare_dataset.py`):

    <cache_dir>/<dataset_key>/dataset_<dataset_key>.json
    <cache_dir>/<dataset_key>/<split>/images_parquet/part-*.parquet   # filename, image_bytes

`load_split` returns a `datasets.Dataset` with one row per image. The metadata
columns `filename`, `imgid`, `caption` and `sentids` come from the Karpathy JSON,
which is the single source of truth for IDs; `image` is decoded lazily on row
access. Use `metadata_column` to read metadata without decoding any image.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATASET_KEYS = ("flickr30k", "coco30k")
_METADATA_COLUMNS = ("filename", "imgid", "caption", "sentids")


def _basename(name: Any) -> str:
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def _decode_image(value: Any):
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    return Image.open(io.BytesIO(value)).convert("RGB")


def decode_images(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """`datasets` format transform that adds a decoded `image` column.

    Metadata-only queries (e.g. `dataset["caption"]`) pass through untouched.
    Kept at module level so `datasets` can hash and copy it cheaply.
    """
    if "image_bytes" not in batch:
        return batch
    batch = dict(batch)
    batch["image"] = [_decode_image(value) for value in batch["image_bytes"]]
    return batch


def metadata_column(dataset, name: str) -> List[Any]:
    """Return one metadata column of a split without decoding images.

    Accepts a `datasets.Dataset` (shuffles/selections are respected) or a plain
    list of row dicts.
    """
    if hasattr(dataset, "with_format"):
        return dataset.with_format(None)[name]
    return [row[name] for row in dataset]


class KarpathyCache:
    """One dataset's local cache: splits with official Karpathy image/caption IDs."""

    def __init__(self, dataset_key: str, cache_dir: str = "data_cache") -> None:
        if dataset_key not in DATASET_KEYS:
            raise ValueError(f"dataset_key must be one of {DATASET_KEYS}, got {dataset_key!r}")
        self.dataset_key = dataset_key
        self.cache_dir = str(cache_dir)
        self._images_by_filename: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = str(split).strip().lower()
        return "val" if split in ("validation", "valid") else split

    def _karpathy_json_path(self) -> Path:
        root = Path(self.cache_dir)
        candidates = [
            root / self.dataset_key / f"dataset_{self.dataset_key}.json",
            root / f"dataset_{self.dataset_key}.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Karpathy annotation file not found. Tried:\n" + "\n".join(f"- {p}" for p in candidates)
        )

    def _karpathy_images(self) -> Dict[str, Dict[str, Any]]:
        """basename -> {imgid, split, captions, sentids}, loaded once."""
        if self._images_by_filename:
            return self._images_by_filename
        with self._karpathy_json_path().open("r", encoding="utf-8") as f:
            images = json.load(f).get("images") or []
        if not images:
            raise ValueError("Karpathy JSON has no 'images' entries.")
        for image in images:
            sentences = image.get("sentences") or []
            if not sentences:
                raise ValueError(f"Image {image.get('filename')} has no captions.")
            self._images_by_filename[_basename(image["filename"])] = {
                "imgid": int(image["imgid"]),
                "split": str(image.get("split", "")).lower(),
                "captions": [str(s["raw"]) for s in sentences],
                "sentids": [int(s["sentid"]) for s in sentences],
            }
        return self._images_by_filename

    def _parquet_files(self, split: str) -> List[Path]:
        directory = Path(self.cache_dir) / self.dataset_key / split / "images_parquet"
        if not directory.exists():
            raise FileNotFoundError(f"Parquet directory not found: {directory}")
        parts = sorted(directory.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet parts found under: {directory}")
        return parts

    def load_split(self, split: str):
        """Load one split with Karpathy metadata columns and lazy image decoding."""
        from datasets import Dataset, concatenate_datasets

        split = self._normalize_split(split)
        images = Dataset.from_parquet([str(p) for p in self._parquet_files(split)])
        name_column = next((c for c in ("file_name", "filename", "file") if c in images.column_names), None)
        if name_column is None or "image_bytes" not in images.column_names:
            raise ValueError("Parquet shards need a file name column and an `image_bytes` column.")

        karpathy = self._karpathy_images()
        names = [_basename(name) for name in images[name_column]]
        in_split = [i for i, name in enumerate(names) if karpathy.get(name, {}).get("split") == split]
        if not in_split:
            raise ValueError(f"No images of split '{split}' found in {self._parquet_files(split)[0].parent}.")

        # Attach metadata as extra Arrow columns (zero-copy; image bytes are not rewritten).
        rows = [karpathy.get(name) for name in names]
        metadata = Dataset.from_dict({
            "filename": names,
            "imgid": [row["imgid"] if row else -1 for row in rows],
            "caption": [row["captions"] if row else [] for row in rows],
            "sentids": [row["sentids"] if row else [] for row in rows],
        })
        stale = [c for c in _METADATA_COLUMNS if c in images.column_names]
        dataset = concatenate_datasets([images.remove_columns(stale), metadata], axis=1)
        if len(in_split) != len(dataset):
            # Parquet shards may contain images of other splits; keep this split only.
            dataset = dataset.select(in_split)
        return dataset.with_transform(decode_images)

    @staticmethod
    def pairs_and_mapping(dataset) -> Tuple[List[Tuple[int, int, int, int, str]], Dict[str, Dict]]:
        """Flatten a split into (imgid, sentid, row_idx, local_idx, caption) pairs.

        Returns `(pairs, {"imgid_to_idx", "idx_to_imgid", "sentid_to_info"})`.
        """
        imgid_to_idx: Dict[int, int] = {}
        idx_to_imgid: Dict[int, int] = {}
        sentid_to_info: Dict[int, Dict[str, Any]] = {}
        pairs: List[Tuple[int, int, int, int, str]] = []
        imgids, captions_by_row, sentids_by_row = (
            metadata_column(dataset, name) for name in ("imgid", "caption", "sentids")
        )
        for idx, (imgid, captions, sentids) in enumerate(
            zip(imgids, captions_by_row, sentids_by_row, strict=True)
        ):
            imgid = int(imgid)
            imgid_to_idx[imgid] = idx
            idx_to_imgid[idx] = imgid
            for local_idx, (sentid, caption) in enumerate(zip(sentids, captions, strict=True)):
                pairs.append((imgid, int(sentid), idx, local_idx, str(caption)))
                sentid_to_info[int(sentid)] = {
                    "imgid": imgid, "text": str(caption), "idx": idx, "local_idx": local_idx,
                }
        return pairs, {
            "imgid_to_idx": imgid_to_idx,
            "idx_to_imgid": idx_to_imgid,
            "sentid_to_info": sentid_to_info,
        }

    @staticmethod
    def caption_mapping(dataset) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Build the retrieval view of a split.

        Returns:
            all_captions: every caption of the split, image by image.
            caption_to_image: dataset row index of each caption's image.
            id_mapping: `sentid_list` (caption -> Karpathy sentid) and
                `idx_to_imgid` (row index -> Karpathy imgid).
        """
        all_captions: List[str] = []
        caption_to_image: List[int] = []
        sentid_list: List[int] = []
        idx_to_imgid: Dict[int, int] = {}
        imgids, captions_by_row, sentids_by_row = (
            metadata_column(dataset, name) for name in ("imgid", "caption", "sentids")
        )
        for idx, (imgid, captions, sentids) in enumerate(
            zip(imgids, captions_by_row, sentids_by_row, strict=True)
        ):
            idx_to_imgid[idx] = int(imgid)
            for sentid, caption in zip(sentids, captions, strict=True):
                all_captions.append(str(caption))
                caption_to_image.append(idx)
                sentid_list.append(int(sentid))
        return all_captions, caption_to_image, {"sentid_list": sentid_list, "idx_to_imgid": idx_to_imgid}
