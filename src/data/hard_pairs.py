"""
RePair hard pairs: synthesis records and how they enter the element graph.

- `repair-i2t` record (query image I_q, FP caption T_fp, edited caption T_edit):
  the local-only pair (I_q, T_edit) is linked from the anchor element of T_fp.
- `repair-t2i` record (query caption T_q, FP image I_fp, edited image I_edit):
  the local-only pair (I_edit, T_q) is linked from the first anchor element of I_fp.

In the grid of that anchor the linked pair is a row of paper Eq. 4: the edit is
the minimal change of the anchor caption/image toward the record's query.

Records live under `<cache_dir>/RePair/<dataset>/{I2T,T2I}`.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

if TYPE_CHECKING:
    from .graph import ElementCollection

logger = logging.getLogger(__name__)

HARD_PAIR_TYPES = ("repair-i2t", "repair-t2i")
CACHE_DIR = "RePair"
_SUBDIR = {"repair-i2t": Path("I2T") / "synthesis_success.parquet", "repair-t2i": Path("T2I")}

# ---------------------------------------------------------------------------
# Synthetic image ids
# ---------------------------------------------------------------------------

MAX_EDITS_PER_QUERY = 1000


def encode_t2i_image_id(query_sentid: int, aug_offset: int) -> int:
    """Negative id of edited image `aug_offset` of text query `query_sentid` (Karpathy imgids are >= 0)."""
    query_sentid, aug_offset = int(query_sentid), int(aug_offset)
    if query_sentid < 0 or not 0 <= aug_offset < MAX_EDITS_PER_QUERY:
        raise ValueError(f"Invalid T2I image key: query_sentid={query_sentid}, aug_offset={aug_offset}")
    return -(query_sentid * MAX_EDITS_PER_QUERY + aug_offset + 1)


def decode_t2i_image_id(image_id: int) -> Tuple[int, int]:
    """(query_sentid, aug_offset) of an encoded synthetic image id."""
    if int(image_id) >= 0:
        raise ValueError(f"Not a synthetic image id: {image_id}")
    return divmod(-int(image_id) - 1, MAX_EDITS_PER_QUERY)


# ---------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------


def default_path(cache_dir: str, dataset_key: str, name: str) -> Path:
    """Default records path of a hard-pair type under the cache root."""
    if name not in _SUBDIR:
        raise ValueError(f"Unknown hard-pair type: {name}; expected one of {HARD_PAIR_TYPES}")
    return Path(cache_dir) / CACHE_DIR / dataset_key / _SUBDIR[name]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _require_columns(names: List[str], required: Tuple[str, ...]) -> None:
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"Missing columns {missing} (available: {names})")


@dataclass(frozen=True)
class I2TRecord:
    query_imgid: int
    fp_sentid: int
    edited_caption: str


class I2TRecords:
    """Selected caption edits, written by `scripts/prepare_i2t.py`.

    Columns used: query_imgid, fp_sentid, edited_caption. `path` is the parquet
    file or its directory.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records: List[I2TRecord] = []

    def load(self) -> List[I2TRecord]:
        import pyarrow.parquet as pq

        if self.records:
            return self.records
        path = self.path / "synthesis_success.parquet" if self.path.is_dir() else self.path
        if not path.is_file():
            raise FileNotFoundError(f"RePair I2T records not found: {path}")
        columns = ("query_imgid", "fp_sentid", "edited_caption")
        _require_columns(pq.read_schema(path).names, columns)
        rows = pq.read_table(path, columns=list(columns)).to_pylist()
        for row in rows:
            text = str(row["edited_caption"] or "").strip()
            if row["query_imgid"] is not None and row["fp_sentid"] is not None and text:
                self.records.append(I2TRecord(int(row["query_imgid"]), int(row["fp_sentid"]), text))
        if len(self.records) != len(rows):
            logger.warning("RePair I2T: skipped %d incomplete rows", len(rows) - len(self.records))
        logger.info("RePair I2T: %d records from %s", len(self.records), path)
        return self.records


@dataclass(frozen=True)
class T2IRecord:
    query_sentid: int
    fp_imgid: int
    gt_imgid: int
    aug_offset: int  # position among the edits of the same query


class T2IRecords:
    """Selected image edits, written by `scripts/prepare_t2i.py`; also serves the images.

    Columns: query_sentid, fp_imgid, gt_imgid, edited_image. Images stay in the
    memory-mapped Arrow cache and are decoded on request.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records: List[T2IRecord] = []
        self._dataset = None
        self._rows_by_query: Dict[int, List[int]] = {}

    def _parquet_files(self) -> List[Path]:
        if self.path.is_file():
            return [self.path]
        files = sorted(self.path.glob("part-*.parquet")) or sorted(self.path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"RePair T2I records not found under: {self.path}")
        return files

    def load(self) -> List[T2IRecord]:
        from datasets import Dataset

        if self._dataset is not None:
            return self.records
        dataset = Dataset.from_parquet([str(p) for p in self._parquet_files()])
        _require_columns(dataset.column_names, ("query_sentid", "fp_imgid", "gt_imgid", "edited_image"))
        query_ids, fp_ids, gt_ids = (dataset[c] for c in ("query_sentid", "fp_imgid", "gt_imgid"))
        has_image = dataset.data.column("edited_image").is_valid().to_pylist()
        for row, (query, fp, gt, valid) in enumerate(zip(query_ids, fp_ids, gt_ids, has_image, strict=True)):
            if None not in (query, fp, gt) and valid:
                self._rows_by_query.setdefault(int(query), []).append(row)
        skipped = len(query_ids) - sum(len(rows) for rows in self._rows_by_query.values())
        if skipped:
            logger.warning("RePair T2I: skipped %d incomplete rows", skipped)
        for query in sorted(self._rows_by_query):
            for offset, row in enumerate(self._rows_by_query[query]):
                self.records.append(T2IRecord(query, int(fp_ids[row]), int(gt_ids[row]), offset))
        self._dataset = dataset
        logger.info("RePair T2I: %d records for %d queries from %s",
                    len(self.records), len(self._rows_by_query), self.path)
        return self.records

    def get_image(self, query_sentid: int, aug_offset: int = 0) -> Any:
        """Edited image `aug_offset` of text query `query_sentid`, as RGB."""
        from PIL import Image

        self.load()
        rows = self._rows_by_query.get(int(query_sentid), [])
        if not 0 <= aug_offset < len(rows):
            raise KeyError(f"RePair T2I: no edited image {aug_offset} for query_sentid {query_sentid}")
        value = self._dataset[rows[aug_offset]]["edited_image"]
        if hasattr(value, "convert"):  # already decoded by an Image feature
            return value.convert("RGB")
        if isinstance(value, dict):
            value = value["bytes"]
        return Image.open(io.BytesIO(value)).convert("RGB")


# ---------------------------------------------------------------------------
# Linking records into the element graph
# ---------------------------------------------------------------------------


def link_i2t_pairs(collection: "ElementCollection", records: List[I2TRecord],
                   keep_image_ids: Optional[Set[int]] = None, verbose: bool = True) -> None:
    anchors_by_caption = collection.anchors_by_caption()
    pair_eids: Dict[Tuple[int, int], int] = {}
    skipped_query = skipped_fp = links = 0
    for rec in tqdm(records, desc="Linking RePair I2T pairs", disable=not verbose):
        if keep_image_ids is not None and rec.query_imgid not in keep_image_ids:
            skipped_query += 1
            continue
        anchor = anchors_by_caption.get(rec.fp_sentid)
        if anchor is None:
            skipped_fp += 1
            continue
        caption_id = collection.caption_id_for(rec.edited_caption)
        key = (rec.query_imgid, caption_id)
        if key not in pair_eids:
            pair_eids[key] = collection.add_element(image_id=rec.query_imgid, caption_id=caption_id,
                                                    is_anchor=False, caption_text=rec.edited_caption).eid
        if pair_eids[key] not in anchor.i2t_pair_eids:
            anchor.i2t_pair_eids.append(pair_eids[key])
            links += 1
    logger.info("RePair I2T: %d pairs, %d links; skipped %d records (query image outside split) and "
                "%d (FP caption outside split)", len(pair_eids), links, skipped_query, skipped_fp)


def link_t2i_pairs(collection: "ElementCollection", records: List[T2IRecord],
                   keep_sentence_ids: Optional[Set[int]] = None, verbose: bool = True) -> None:
    anchors_by_image = collection.anchors_by_image()
    pair_eids: Dict[Tuple[int, int], int] = {}
    skipped_query = skipped_fp = links = 0
    for rec in tqdm(records, desc="Linking RePair T2I pairs", disable=not verbose):
        if keep_sentence_ids is not None and rec.query_sentid not in keep_sentence_ids:
            skipped_query += 1
            continue
        anchor = anchors_by_image.get(rec.fp_imgid)
        if anchor is None:
            skipped_fp += 1
            continue
        key = (rec.query_sentid, rec.aug_offset)
        if key not in pair_eids:
            pair_eids[key] = collection.add_element(image_id=encode_t2i_image_id(*key),
                                                    caption_id=rec.query_sentid, is_anchor=False).eid
        if pair_eids[key] not in anchor.t2i_pair_eids:
            anchor.t2i_pair_eids.append(pair_eids[key])
            links += 1
    logger.info("RePair T2I: %d pairs, %d links; skipped %d records (query caption outside split) and "
                "%d (FP image outside split)", len(pair_eids), links, skipped_query, skipped_fp)
