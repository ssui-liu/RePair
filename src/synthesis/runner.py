"""Run a workflow over work items, serially or with a thread pool, with resume support."""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .api import TokenTracker
from .store import ResultStore
from .work_items import WorkItem

logger = logging.getLogger(__name__)


class BatchRunner:
    """Process work items with `workflow.process(item, dataset)` and log every result.

    API calls are I/O bound, so parallel mode uses threads; results are saved by
    the main thread only.
    """

    def __init__(self, workflow: Any, store: ResultStore, dataset: Any,
                 token_tracker: Optional[TokenTracker] = None):
        self.workflow = workflow
        self.store = store
        self.dataset = dataset
        self.token_tracker = token_tracker or TokenTracker()
        self.counts = {"success": 0, "qc_rejected": 0, "failed": 0, "resumed": 0}

    def _pending(self, work_items: List[WorkItem]) -> List[WorkItem]:
        pending = [item for item in work_items if not self.store.is_already_processed(item.item_id)]
        self.counts["resumed"] = len(work_items) - len(pending)
        if self.counts["resumed"]:
            logger.info("Resume: skipping %d already-processed items", self.counts["resumed"])
        return pending

    def _record(self, item: WorkItem, result: Optional[Dict[str, Any]], error: Optional[Exception]) -> None:
        if error is not None:
            logger.error("Error processing %s: %s", item.item_id, error)
            result = {"item_id": item.item_id, "task_type": "i2t", "status": "failed", "error": str(error),
                      "query_imgid": item.query_imgid, "fp_sentid": item.fp_sentid, "fp_index": item.fp_index}
        self.store.save(result)
        self.counts[result["status"]] += 1

    def _progress(self, bar: tqdm) -> None:
        c = self.counts
        bar.set_postfix_str(f"ok {c['success']} | qc-rejected {c['qc_rejected']} | fail {c['failed']} | "
                            f"resumed {c['resumed']} | {self.token_tracker.progress_string()}")

    def run(self, work_items: List[WorkItem], max_workers: int = 1, show_progress: bool = True) -> Dict[str, int]:
        """Process all pending items; returns per-status counts (incl. `resumed`)."""
        pending = self._pending(work_items)
        logger.info("Processing %d items with %d worker(s)", len(pending), max_workers)
        with tqdm(total=len(pending), desc="[I2T]", disable=not show_progress, unit="item") as bar:
            if max_workers <= 1:
                for item in pending:
                    try:
                        self._record(item, self.workflow.process(item, self.dataset), None)
                    except Exception as error:
                        self._record(item, None, error)
                    bar.update(1)
                    self._progress(bar)
                return dict(self.counts)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                queue = iter(pending)
                running = {}
                # Keep at most `max_workers` items in flight.
                for item in queue:
                    running[executor.submit(self.workflow.process, item, self.dataset)] = item
                    if len(running) >= max_workers:
                        break
                while running:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = running.pop(future)
                        error = future.exception()
                        self._record(item, None if error else future.result(), error)
                        next_item = next(queue, None)
                        if next_item is not None:
                            running[executor.submit(self.workflow.process, next_item, self.dataset)] = next_item
                        bar.update(1)
                        self._progress(bar)
        return dict(self.counts)
