"""
Collate anchors into local 3x3 grids (paper Algorithm 2).

For an anchor (I^a, T^a) the grid rows are matched (image, text) pairs:

    row 0: (I^a, T^a)              the original anchor pair
    row 1: (I_q, T_edit)           sampled from `i2t_pair_eids` (RePair I2T)
    row 2: (I_edit, T_q)           sampled from `t2i_pair_eids` (RePair T2I)

so I = [I^a, I_q, I_edit] and T = [T^a, T_edit, T_q]: diagonal entries are the
positives and the six off-diagonal entries are hard negatives (paper Eq. 4).
An anchor without a link of some kind gets a smaller grid; a 1x1 grid has
zero loss. Only rows 0 enter the global InfoNCE.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .graph import ElementIndex

logger = logging.getLogger(__name__)


class GridCollator:
    """Build the grids of a batch of anchors and encode their distinct images/texts once."""

    def __init__(
        self,
        processor: Any,
        id2image: Callable[[int], Any],
        id2text: Callable[[int], str],
        index: "ElementIndex",
        pairs_per_direction: int = 1,
        seed: Optional[int] = None,
        log_batch_stats: bool = False,
    ) -> None:
        """
        Args:
            processor: CLIP processor.
            id2image / id2text: Fetch an image / caption by id (see AnchorDataset).
            index: Element index holding the RePair links.
            pairs_per_direction: Hard pairs sampled per direction (I2T, T2I); 1 gives 3x3 grids.
            seed: Seed of the link-sampling RNG.
            log_batch_stats: Log grid composition for every batch.
        """
        self.processor = processor
        self.id2image = id2image
        self.id2text = id2text
        self.index = index
        self.pairs_per_direction = max(1, int(pairs_per_direction))
        self.rng = random.Random(seed)
        self._worker_seed: Optional[int] = None
        self.log_batch_stats = log_batch_stats

    def _sync_worker_rng(self) -> None:
        # In DataLoader workers each process holds a copy of this collator; give
        # every worker (and epoch) its own reproducible stream via torch's worker seed.
        info = torch.utils.data.get_worker_info()
        if info is not None and info.seed != self._worker_seed:
            self._worker_seed = info.seed
            self.rng = random.Random(info.seed)

    def _sample_links(self, eids: List[int]) -> List[int]:
        unique = list(dict.fromkeys(eids))
        if not unique:
            return []
        return self.rng.sample(unique, min(self.pairs_per_direction, len(unique)))

    def _build_grid(self, eid: int) -> Tuple[List[int], List[int]]:
        """Return the matched image ids and text ids of one anchor's grid."""
        anchor = self.index.get_element(eid)
        image_ids, text_ids = [anchor.image_id], [anchor.caption_id]
        for links in (anchor.i2t_pair_eids, anchor.t2i_pair_eids):
            for neighbor_eid in self._sample_links(links):
                neighbor = self.index.get_element(neighbor_eid)
                # Keep rows paired: dropping only a duplicated modality would shift the diagonal.
                if neighbor.image_id not in image_ids and neighbor.caption_id not in text_ids:
                    image_ids.append(neighbor.image_id)
                    text_ids.append(neighbor.caption_id)
        return image_ids, text_ids

    @staticmethod
    def _pad(rows: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        width = max(len(row) for row in rows)
        padded = torch.full((len(rows), width), -1, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(row)] = True
        return padded, mask

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Empty batch received")
        self._sync_worker_rng()
        grids = [self._build_grid(int(sample["eid"])) for sample in samples]

        # Encode every distinct image/text of the batch once.
        unique_images = list(dict.fromkeys(i for image_ids, _ in grids for i in image_ids))
        unique_texts = list(dict.fromkeys(t for _, text_ids in grids for t in text_ids))
        image_pos = {image_id: k for k, image_id in enumerate(unique_images)}
        text_pos = {text_id: k for k, text_id in enumerate(unique_texts)}
        grid_images, mask = self._pad([[image_pos[i] for i in image_ids] for image_ids, _ in grids])
        grid_texts, _ = self._pad([[text_pos[t] for t in text_ids] for _, text_ids in grids])

        pixel_values = self.processor(
            images=[self.id2image(i) for i in unique_images], return_tensors="pt"
        )["pixel_values"]
        text_batch = self.processor(
            text=[self.id2text(t) for t in unique_texts], padding=True, truncation=True, return_tensors="pt"
        )

        if self.log_batch_stats:
            sizes = mask.sum(dim=1)
            logger.info(
                "[BATCH_STATS] anchors=%d, grid_rows=%s, unique_images=%d, unique_texts=%d",
                len(grids), {int(n): int((sizes == n).sum()) for n in sizes.unique()},
                len(unique_images), len(unique_texts),
            )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "image_ids": torch.tensor(unique_images, dtype=torch.long),
            "text_ids": torch.tensor(unique_texts, dtype=torch.long),
            # Row 0 of every grid: the original anchor pair (global InfoNCE).
            "anchor_image_indices": grid_images[:, 0].clone(),
            "anchor_text_indices": grid_texts[:, 0].clone(),
            "grid_image_indices": grid_images,
            "grid_text_indices": grid_texts,
            "grid_mask": mask,
        }
