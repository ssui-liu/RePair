"""
Datasets of a split: original (image, caption) pairs, or anchors with RePair hard pairs.

    cache = KarpathyCache("flickr30k", "data_cache")
    pairs = load_pair_dataset(cache, "test")
    anchors, index = load_anchor_dataset(cache, "train", {"repair-i2t": None, "repair-t2i": None})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from . import hard_pairs
from .graph import ElementIndex, build_element_index
from .karpathy import KarpathyCache

logger = logging.getLogger(__name__)


class PairDataset(Dataset):
    """Original (image, caption) pairs of a split."""

    def __init__(self, hf_dataset, pairs: List[Tuple[int, int, int, int, str]]) -> None:
        """`pairs` are (imgid, sentid, row_idx, local_idx, caption) tuples (`KarpathyCache.pairs_and_mapping`)."""
        self.hf_dataset = hf_dataset
        self.samples = [(imgid, sentid, row, caption) for imgid, sentid, row, _, caption in pairs]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, text_id, row, text = self.samples[idx]
        return {"image": self.hf_dataset[row]["image"], "text": text, "image_id": image_id, "text_id": text_id}


class AnchorDataset(Dataset):
    """The anchors (original pairs) of an element index; items are `{"eid": ...}` for `GridCollator`."""

    def __init__(
        self,
        hf_dataset,
        index: ElementIndex,
        imgid_to_idx: Dict[int, int],
        sentid_to_info: Dict[int, Dict],
        t2i_records: Optional[hard_pairs.T2IRecords] = None,
    ) -> None:
        """
        Args:
            hf_dataset: Split returned by `KarpathyCache.load_split`.
            index: Element index with the RePair links.
            imgid_to_idx: Karpathy imgid -> row of `hf_dataset`.
            sentid_to_info: Karpathy sentid -> {"text", ...} for the split's captions.
            t2i_records: Source of the edited images (negative image ids).
        """
        self.hf_dataset = hf_dataset
        self.index = index
        self.imgid_to_idx = imgid_to_idx
        self.sentid_to_info = sentid_to_info
        self.t2i_records = t2i_records
        logger.info("AnchorDataset: %d anchors, %d elements", len(index.anchor_eids), index.num_elements)

    def __len__(self) -> int:
        return len(self.index.anchor_eids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"eid": self.index.anchor_eids[idx]}

    def get_image(self, image_id: int) -> Any:
        """PIL image of a Karpathy imgid or of an edited (T2I) image id."""
        if image_id < 0:
            if self.t2i_records is None:
                raise RuntimeError(f"Edited image {image_id} requested but repair-t2i records are not loaded.")
            return self.t2i_records.get_image(*hard_pairs.decode_t2i_image_id(image_id))
        row = self.imgid_to_idx.get(image_id)
        if row is None:
            raise KeyError(f"image_id {image_id} is not in this split")
        return self.hf_dataset[row]["image"]

    def get_text(self, text_id: int) -> str:
        """Text of a Karpathy sentid or of an edited caption id."""
        info = self.sentid_to_info.get(text_id)
        if info is not None:
            return info["text"]
        text = self.index.get_text_content(text_id)
        if text is None:
            raise KeyError(f"text_id {text_id} is not in this split or among the edited captions")
        return text


def _load_split(cache: KarpathyCache, split: str, subsample: Optional[int], shuffle_subsample: bool, seed: int):
    hf_dataset = cache.load_split(split)
    if subsample and 0 < subsample < len(hf_dataset):
        if shuffle_subsample:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        hf_dataset = hf_dataset.select(range(subsample))
    return hf_dataset


def load_pair_dataset(cache: KarpathyCache, split: str, subsample: Optional[int] = None,
                      shuffle_subsample: bool = True, seed: int = 42) -> PairDataset:
    """Original pairs of `split`; `subsample` keeps that many images (shuffled with `seed` first)."""
    hf_dataset = _load_split(cache, split, subsample, shuffle_subsample, seed)
    pairs, _ = cache.pairs_and_mapping(hf_dataset)
    return PairDataset(hf_dataset, pairs)


def load_anchor_dataset(
    cache: KarpathyCache,
    split: str,
    hard_pair_paths: Dict[str, Optional[str]],
    subsample: Optional[int] = None,
    shuffle_subsample: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[AnchorDataset, ElementIndex]:
    """Anchors of `split` linked to the enabled hard pairs.

    Args:
        hard_pair_paths: Enabled types in order -> records path (None: default cache path).
    """
    hf_dataset = _load_split(cache, split, subsample, shuffle_subsample, seed)
    _, mapping = cache.pairs_and_mapping(hf_dataset)
    paths = {name: path or hard_pairs.default_path(cache.cache_dir, cache.dataset_key, name)
             for name, path in hard_pair_paths.items()}
    index, records = build_element_index(
        hf_dataset, paths, keep_sentence_ids=set(mapping["sentid_to_info"]), verbose=verbose
    )
    dataset = AnchorDataset(hf_dataset, index, mapping["imgid_to_idx"], mapping["sentid_to_info"],
                            t2i_records=records.get("repair-t2i"))
    return dataset, index
