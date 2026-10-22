"""
I2T synthesis of one (query image, FP caption) pair (paper Algorithm 1):

1. QC: one call checks that the GT caption matches the image (criterion 1) and
   that the FP caption is not already a valid description (criterion 2).
2. Edit instruction G(q, p_fp) from the image and the FP caption only.
3. M caption edits E(p_fp, tau_edit) and selection of the edit with maximal
   CLIP cosine to p_fp (post-edit quality control, PQC, Eq. 3).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..work_items import WorkItem
from .cosine_selector import CosineSelector
from .steps import I2TSteps

logger = logging.getLogger(__name__)


class I2TWorkflow:
    def __init__(
        self,
        api_client: Any,
        sentid_to_text: Dict[int, str],
        imgid_to_idx: Dict[int, int],
        enable_qc: bool = True,
        enable_gt_check: bool = True,
        enable_fp_check: bool = True,
        enable_pqc: bool = True,
        num_candidates: int = 1,
        cosine_selector: Optional[CosineSelector] = None,
    ):
        """
        Args:
            api_client: `APIClient` (or a compatible mock).
            sentid_to_text: Karpathy sentid -> caption, for FP captions missing from a work item.
            imgid_to_idx: Karpathy imgid -> dataset row of the query image.
            enable_qc / enable_gt_check / enable_fp_check: QC and its two criteria.
            enable_pqc: Select among the M edits by CLIP cosine (else take the first).
            num_candidates: M, edits generated per pair (paper: 3).
            cosine_selector: Required when PQC is on and M > 1.
        """
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if enable_pqc and num_candidates > 1 and cosine_selector is None:
            raise ValueError("PQC with M>1 requires a cosine selector")
        self.steps = I2TSteps(api_client)
        self.sentid_to_text = sentid_to_text
        self.imgid_to_idx = imgid_to_idx
        self.enable_qc = enable_qc
        self.enable_gt_check = enable_gt_check
        self.enable_fp_check = enable_fp_check
        self.enable_pqc = enable_pqc
        self.num_candidates = num_candidates
        self.cosine_selector = cosine_selector

    def process(self, work_item: WorkItem, dataset: Any) -> Dict[str, Any]:
        """Run one work item; API/parse errors propagate and are recorded as failures by the runner."""
        item = {
            "item_id": work_item.item_id,
            "query_imgid": work_item.query_imgid,
            "fp_sentid": work_item.fp_sentid,
            "fp_index": work_item.fp_index,
            "query_image_filename": work_item.query_image_filename,
            "gt_caption": " | ".join(work_item.gt_captions) or work_item.best_gt_caption,
        }
        fp_caption = work_item.fp_caption or self.sentid_to_text.get(work_item.fp_sentid, "")
        if not fp_caption:
            return _result(item, "failed", error="No FP caption text available")
        row = self.imgid_to_idx.get(work_item.query_imgid)
        if row is None:
            return _result(item, "failed", error=f"query_imgid not in dataset: {work_item.query_imgid}")
        query_img = dataset[row]["image"]
        item["fp_caption"] = fp_caption

        qc_checked = self.enable_qc and (self.enable_gt_check or self.enable_fp_check)
        if qc_checked:
            gt_valid, fp_is_benign, reason = self.steps.quality_check(query_img, item["gt_caption"], fp_caption)
            if self.enable_gt_check and not gt_valid:
                return self._qc_reject(item, f"GT inconsistent: {reason}")
            if self.enable_fp_check and fp_is_benign:
                return self._qc_reject(item, f"FP is benign mismatch: {reason}")
        item["qc_status"] = "passed" if qc_checked else "not_checked"

        instruction = self.steps.edit_instruction(query_img, fp_caption)
        item["edit_instruction"] = instruction
        candidates = self._edit_candidates(item["item_id"], fp_caption, instruction)
        if len(candidates) != self.num_candidates:
            return _result(item, "failed",
                           error=f"Expected {self.num_candidates} candidates, received {len(candidates)}")
        if self.enable_pqc and len(candidates) > 1:
            edited, cosine = self.cosine_selector.select_closest(fp_caption, candidates)
        else:
            edited, cosine = candidates[0], None
        item.update(edited_caption=edited, num_candidates=self.num_candidates, selection_cosine=cosine,
                    gt_rank=work_item.gt_rank)
        return _result(item, "success", query_img=query_img)

    def _edit_candidates(self, item_id: str, fp_caption: str, instruction: str) -> List[str]:
        """M sampled edits; with M=1 an editing error propagates, with M>1 it drops that candidate."""
        if self.num_candidates == 1:
            return [self.steps.edit_caption(fp_caption, instruction)]
        candidates = []
        for m in range(self.num_candidates):
            try:
                candidates.append(self.steps.edit_caption(fp_caption, instruction))
            except Exception as e:
                logger.warning("[I2T] %s: edit candidate %d failed: %s", item_id, m, e)
        return candidates

    @staticmethod
    def _qc_reject(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
        logger.warning("[I2T QC REJECT] %s: %s", item["item_id"], reason)
        return _result(item, "qc_rejected", qc_status="rejected", error=f"QC Rejected: {reason}")


def _result(item: Dict[str, Any], status: str, error: Optional[str] = None, query_img=None,
            **fields) -> Dict[str, Any]:
    """Result record of one work item (`_query_img` is consumed by the result store)."""
    result = {"task_type": "i2t", "qc_status": "not_checked", **item, **fields, "status": status, "error": error}
    if query_img is not None:
        result["_query_img"] = query_img
    return result
