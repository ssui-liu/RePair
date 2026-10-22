"""
Synthesis outputs: one directory per work item, grouped by status.

    {output_dir}/{task}/{signature}/
      config.json, summary.json
      success/<item_id>/      metadata.json, captions_compare.txt, directive.json[, query_image.jpg]
      qc_rejected/<item_id>/  analysis.json
      failed/<item_id>/       error.json

The signature encodes the settings that change an item's meaning (see
config/synthesis/OUTPUT_LAYOUT.md), so resuming a run never mixes settings.
Item directories double as the resume record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Union

logger = logging.getLogger(__name__)

STATUSES = ("success", "qc_rejected", "failed")
_SECRET_KEYS = {"api_key", "token", "password", "secret"}


def redact_secrets(value):
    """Keep credentials out of persisted configs and CLI previews."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if v and k.lower() in _SECRET_KEYS else redact_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class SynthesisResult:
    """Persisted record of one work item."""

    item_id: str
    task_type: str = "i2t"
    status: str = "failed"           # success | qc_rejected | failed
    error: Optional[str] = None
    qc_status: str = "not_checked"   # passed | rejected | not_checked
    query_imgid: Optional[int] = None
    fp_sentid: Optional[int] = None
    fp_index: Optional[int] = None
    query_image_filename: Optional[str] = None
    gt_rank: Optional[int] = None
    gt_caption: Optional[str] = None
    fp_caption: Optional[str] = None
    edited_caption: Optional[str] = None
    edit_instruction: Optional[str] = None
    num_candidates: Optional[int] = None
    selection_cosine: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisResult":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ResultStore:
    def __init__(
        self,
        output_dir: Union[str, Path],
        task_type: str,
        max_queries: int,
        fps_per_query: int = 1,
        gt_caption_mode: str = "first",
        num_candidates: int = 1,
        enable_qc: bool = True,
        enable_gt_check: bool = True,
        enable_fp_check: bool = True,
        enable_pqc: bool = True,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume: bool = True,
        save_query_images: bool = False,
        resume_processed_statuses: Optional[Iterable[str]] = None,
    ):
        """
        Args:
            output_dir: `<results>/<dataset>/<retriever>/synthesis/<split>`.
            max_queries / fps_per_query: Run budget; recorded, not part of the signature.
            gt_caption_mode, num_candidates, enable_*, model_name: Encoded in the signature.
            config: Full run config, saved with secrets redacted.
            resume: Skip items whose directory exists under a status in `resume_processed_statuses`
                (default: all statuses; e.g. ["success", "qc_rejected"] re-runs failures).
            save_query_images: Also save the query image of successful items.
        """
        self.task_type = task_type
        self.gt_caption_mode = gt_caption_mode
        self.num_candidates = num_candidates
        self.enable_qc = enable_qc
        self.enable_gt_check = enable_gt_check
        self.enable_fp_check = enable_fp_check
        self.enable_pqc = enable_pqc
        self.model_name = model_name
        self.config = redact_secrets(config or {})
        self.resume = resume
        self.save_query_images = save_query_images
        self.resume_processed_statuses = self._normalize_resume_statuses(resume_processed_statuses)

        self.run_dir = Path(output_dir) / task_type / self._signature()
        self.status_dirs = {status: self.run_dir / status for status in STATUSES}
        for directory in self.status_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.run_metadata = {
            "task_type": task_type,
            "max_queries": max_queries,
            "fps_per_query": fps_per_query,
            "gt_caption_mode": gt_caption_mode,
            "num_candidates": num_candidates,
            "enable_qc": enable_qc,
            "enable_gt_check": enable_gt_check,
            "enable_fp_check": enable_fp_check,
            "enable_pqc": enable_pqc,
            "model_name": model_name,
            "resume": resume,
            "resume_processed_statuses": sorted(self.resume_processed_statuses),
            "save_query_images": save_query_images,
            "start_time": datetime.now().isoformat(),
        }
        self._processed_ids: Set[str] = set()
        if resume:
            for status in self.resume_processed_statuses:
                self._processed_ids |= self._item_ids(status)
        self._save_config()
        logger.info("Synthesis run directory: %s", self.run_dir)

    @staticmethod
    def _normalize_resume_statuses(values: Optional[Iterable[str]]) -> Set[str]:
        if values is None:
            return set(STATUSES)
        statuses = {str(v).strip().lower() for v in values if str(v).strip()}
        invalid = statuses - set(STATUSES)
        if invalid:
            raise ValueError(f"Invalid resume_processed_statuses: {sorted(invalid)}. Allowed: {list(STATUSES)}")
        return statuses

    @staticmethod
    def _slug(value: Optional[str], max_len: int = 32) -> str:
        if not value:
            return "na"
        slug = re.sub(r"\s+", "-", str(value).strip().lower().replace("\\", "-").replace("/", "-"))
        return re.sub(r"[^a-z0-9._-]", "", slug)[:max_len]

    def _signature(self) -> str:
        """Settings that change an item's meaning; budgets are excluded to keep resume stable."""
        parts = [
            f"gt{self.gt_caption_mode}",
            f"m{self.num_candidates}",
            f"qc{int(self.enable_qc)}",
            f"gtqc{int(self.enable_qc and self.enable_gt_check)}",
            f"fpqc{int(self.enable_qc and self.enable_fp_check)}",
            f"pqc{int(self.enable_pqc)}",
        ]
        if self.model_name:
            parts.append(self._slug(self.model_name))
        return "_".join(parts)

    def _item_ids(self, status: str) -> Set[str]:
        return {child.name for child in self.status_dirs[status].iterdir() if child.is_dir()}

    def _save_config(self) -> None:
        path = self.run_dir / "config.json"
        if not path.exists():
            _write_json(path, self.config)
            return
        if json.loads(path.read_text(encoding="utf-8")) != self.config:
            # Never overwrite the config of an existing run; keep the new one alongside.
            digest = hashlib.sha1(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:10]
            _write_json(self.run_dir / f"config_override_{digest}.json", self.config)
            logger.warning("Config differs from the existing run; saved config_override_%s.json", digest)

    def is_already_processed(self, item_id: str) -> bool:
        return self.resume and item_id in self._processed_ids

    def save(self, result_dict: Dict[str, Any]) -> Path:
        """Write one workflow result to its status directory."""
        query_img = result_dict.pop("_query_img", None)
        result = SynthesisResult.from_dict(result_dict)
        if result.status not in STATUSES:
            raise ValueError(f"Unknown result status: {result.status}")
        item_dir = self.status_dirs[result.status] / result.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        record = result.to_dict()

        if result.status == "success":
            _write_json(item_dir / "metadata.json", record)
            (item_dir / "captions_compare.txt").write_text(
                f"Query image: {result.query_image_filename or result.item_id}\n\n"
                f"Ground-truth caption:\n{result.gt_caption or 'N/A'}\n\n"
                f"False-positive caption:\n{result.fp_caption or 'N/A'}\n\n"
                f"Edited caption:\n{result.edited_caption}\n",
                encoding="utf-8",
            )
            if result.edit_instruction:
                _write_json(item_dir / "directive.json", {"edit_instruction": result.edit_instruction})
            if query_img is not None and self.save_query_images:
                query_img.save(item_dir / "query_image.jpg", format="JPEG", quality=95)
            logger.info("[SUCCESS] %s", result.item_id)
        elif result.status == "qc_rejected":
            _write_json(item_dir / "analysis.json", record)
            logger.info("[QC_REJECTED] %s: %s", result.item_id, result.error)
        else:
            _write_json(item_dir / "error.json", {"item_id": result.item_id, "error": result.error,
                                                  "timestamp": result.timestamp})
            logger.warning("[FAILED] %s: %s", result.item_id, result.error)
        self._processed_ids.add(result.item_id)
        return item_dir

    def statistics(self) -> Dict[str, Any]:
        """Cumulative counts over all invocations of this run directory (one status per item id)."""
        counts, seen = {}, set()
        for status in STATUSES:
            ids = self._item_ids(status) - seen
            counts[status] = len(ids)
            seen |= ids
        total = len(seen)
        return {"total": total, **counts, "success_rate": counts["success"] / total if total else 0.0}

    def save_summary(self, token_usage: Optional[Dict[str, int]] = None) -> Path:
        summary = {
            "run_info": {**self.run_metadata, "end_time": datetime.now().isoformat()},
            "statistics": self.statistics(),
            "token_usage_this_invocation": token_usage or {},
        }
        path = self.run_dir / "summary.json"
        _write_json(path, summary)
        logger.info("Summary saved to %s", path)
        return path
