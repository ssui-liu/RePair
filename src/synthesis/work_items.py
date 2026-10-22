"""
Work items: one per (query image, FP caption) pair of an I2T FP pool.

Every query has a pool of K_mine non-GT captions (paper Eq. 1); the first
`fps_per_query` are retained (Eq. 2, d_I2T = 5). Queries are processed worst GT
rank first, so a `max_queries` budget below the number of queries keeps the
hardest ones. The M edit candidates of a pair are generated inside `I2TWorkflow`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

logger = logging.getLogger(__name__)

# Ground-truth caption(s) shown to QC: best ranked ("first"), last annotated ("last") or all.
GTCaptionMode = Literal["first", "last", "all"]


@dataclass
class WorkItem:
    """One (query image, FP caption) pair of an FP-pool sample."""

    sample: Dict[str, Any]
    fp_index: int          # position in the query's FP pool
    fp_sentid: int         # Karpathy sentid of the FP caption
    fp_caption: str = ""
    gt_captions: List[str] = field(default_factory=list)

    @property
    def query_imgid(self) -> int:
        return int(self.sample["query_imgid"])

    @property
    def query_image_filename(self) -> str:
        return str(self.sample.get("query_image_filename", ""))

    @property
    def gt_rank(self) -> int:
        return int(self.sample["ground_truth"].get("best_rank", 0))

    @property
    def best_gt_caption(self) -> str:
        return str(self.sample["ground_truth"].get("best_text", ""))

    @property
    def item_id(self) -> str:
        """`{query_imgid}_fp{fp_index}`; also the item's output directory name."""
        return f"{self.query_imgid}_fp{self.fp_index}"


class WorkItemBuilder:
    def __init__(
        self,
        max_queries: int = 10,
        fps_per_query: int = 1,
        fp_top_pct: Optional[float] = None,
        max_pairs: Optional[int] = None,
        gt_caption_mode: GTCaptionMode = "first",
    ):
        """
        Args:
            max_queries: Number of queries to process (worst GT rank first).
            fps_per_query: FPs retained per query (paper d_I2T).
            fp_top_pct: Alternatively retain ceil(pool_size * pct) FPs (at least 1).
            max_pairs: Global cap on the number of work items.
            gt_caption_mode: Ground-truth caption(s) given to QC.
        """
        if fp_top_pct is not None and not 0.0 < float(fp_top_pct) <= 1.0:
            raise ValueError(f"fp_top_pct must be in (0, 1], got: {fp_top_pct}")
        if max_pairs is not None and int(max_pairs) < 0:
            raise ValueError(f"max_pairs must be >= 0 or null, got: {max_pairs}")
        self.max_queries = max_queries
        self.fps_per_query = fps_per_query
        self.fp_top_pct = float(fp_top_pct) if fp_top_pct is not None else None
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.gt_caption_mode = gt_caption_mode

    def _num_fps(self, pool_size: int) -> int:
        if pool_size <= 0:
            return 0
        if self.fp_top_pct is not None:
            return max(1, min(math.ceil(pool_size * self.fp_top_pct), pool_size))
        return min(max(int(self.fps_per_query), 0), pool_size)

    def select_queries(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Samples with a non-empty FP pool, worst GT rank first, capped at max_queries."""
        with_pool = [s for s in samples if (s.get("false_positives") or {}).get("sentids")]
        with_pool.sort(key=lambda s: int((s.get("ground_truth") or {}).get("best_rank") or 0), reverse=True)
        selected = with_pool[:self.max_queries]
        logger.info("Selected %d of %d queries with FP pools", len(selected), len(with_pool))
        return selected

    def _gt_captions(self, sample: Dict[str, Any], sentid_to_text: Optional[Mapping[int, str]]) -> List[str]:
        gt = sample.get("ground_truth") or {}
        sentids = gt.get("sentids") or []
        if self.gt_caption_mode == "first" or not sentids:
            return [gt["best_text"]] if gt.get("best_text") else []
        if sentid_to_text is None:
            return []
        chosen = sentids[-1:] if self.gt_caption_mode == "last" else sentids
        return [str(sentid_to_text[int(s)]) for s in chosen if sentid_to_text.get(int(s))]

    def expand(self, samples: List[Dict[str, Any]],
               sentid_to_text: Optional[Mapping[int, str]] = None) -> List[WorkItem]:
        items: List[WorkItem] = []
        for sample in samples:
            pool = sample.get("false_positives") or {}
            fp_ids = [int(s) for s in pool.get("sentids") or []]
            fp_texts = pool.get("texts") or []
            gt_ids = {int(s) for s in (sample.get("ground_truth") or {}).get("sentids") or []}
            gt_captions = self._gt_captions(sample, sentid_to_text)
            for fp_index, fp_id in enumerate(fp_ids[:self._num_fps(len(fp_ids))]):
                if self.max_pairs is not None and len(items) >= self.max_pairs:
                    break
                if fp_id in gt_ids:  # defensive: FP pools already exclude positives
                    continue
                fp_caption = fp_texts[fp_index] if fp_index < len(fp_texts) else ""
                if not fp_caption and sentid_to_text is not None:
                    fp_caption = str(sentid_to_text.get(fp_id, ""))
                items.append(WorkItem(sample, fp_index, fp_id, fp_caption, list(gt_captions)))
        strategy = f"top_pct={self.fp_top_pct}" if self.fp_top_pct is not None else f"fixed_k={self.fps_per_query}"
        logger.info("Expanded %d queries to %d work items (%s, max_pairs=%s)",
                    len(samples), len(items), strategy, self.max_pairs)
        return items

    def build(self, samples: List[Dict[str, Any]],
              sentid_to_text: Optional[Mapping[int, str]] = None) -> List[WorkItem]:
        return self.expand(self.select_queries(samples), sentid_to_text)
