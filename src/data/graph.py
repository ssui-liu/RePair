"""
The element graph of a training split.

An element is one (image_id, caption_id) pair. Every original caption gives an
anchor element; RePair hard pairs (see `hard_pairs.py`) are local-only elements
linked from the anchors whose caption (`i2t_pair_eids`) or image
(`t2i_pair_eids`) they edit. `GridCollator` turns an anchor and one sampled link
of each kind into the grid of paper Eq. 4. Only anchors enter the global InfoNCE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

from . import hard_pairs
from .karpathy import metadata_column

logger = logging.getLogger(__name__)


@dataclass
class Element:
    """One (image, caption) pair and the RePair hard pairs linked to it.

    Attributes:
        eid: Unique element id.
        image_id: Karpathy imgid, or a negative id of an edited (T2I) image.
        caption_id: Karpathy sentid, or an id allocated for an edited (I2T) caption.
        i2t_pair_eids: Hard pairs of I2T records whose FP caption is this caption.
        t2i_pair_eids: Hard pairs of T2I records whose FP image is this image.
        is_anchor: True for original pairs; False for hard pairs.
        caption_text: Text of an edited caption (original texts come from the split).
    """

    eid: int
    image_id: int
    caption_id: int
    i2t_pair_eids: List[int] = field(default_factory=list)
    t2i_pair_eids: List[int] = field(default_factory=list)
    is_anchor: bool = True
    caption_text: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.eid, int) or self.eid < 0:
            raise ValueError(f"eid must be a non-negative integer, got {self.eid}")
        if not isinstance(self.image_id, int) or not isinstance(self.caption_id, int):
            raise ValueError(f"image_id/caption_id must be integers, got {self.image_id}/{self.caption_id}")


@dataclass
class ElementCollection:
    """Mutable element list used while the graph is built (anchors first).

    Attributes:
        synthetic_texts: caption_id -> text of edited captions.
        text_to_id: stripped caption text -> caption_id (identical edits share an id).
        next_text_id: First caption id not used by the split.
    """

    elements: List[Element] = field(default_factory=list)
    synthetic_texts: Dict[int, str] = field(default_factory=dict)
    text_to_id: Dict[str, int] = field(default_factory=dict)
    next_text_id: int = 0

    def add_element(self, **kwargs) -> Element:
        element = Element(eid=len(self.elements), **kwargs)
        self.elements.append(element)
        return element

    def caption_id_for(self, text: str) -> int:
        """Caption id of `text`, allocating one for new edited text."""
        caption_id = self.text_to_id.get(text)
        if caption_id is None:
            caption_id = self.next_text_id
            self.next_text_id += 1
            self.synthetic_texts[caption_id] = text
            self.text_to_id[text] = caption_id
        return caption_id

    def anchors_by_caption(self) -> Dict[int, Element]:
        by_caption: Dict[int, Element] = {}
        for elem in self.elements:
            if elem.is_anchor:
                by_caption.setdefault(elem.caption_id, elem)
        return by_caption

    def anchors_by_image(self) -> Dict[int, Element]:
        """imgid -> the image's first anchor element (lowest eid)."""
        by_image: Dict[int, Element] = {}
        for elem in self.elements:
            if elem.is_anchor:
                by_image.setdefault(elem.image_id, elem)
        return by_image


class ElementIndex:
    """Elements by eid, the anchor list, and the texts of edited captions."""

    def __init__(self, elements: List[Element], synthetic_texts: Optional[Dict[int, str]] = None) -> None:
        if not elements:
            raise ValueError("Cannot build index from empty element list")
        self.eid2elem: Dict[int, Element] = {}
        for elem in elements:
            if elem.eid in self.eid2elem:
                raise ValueError(f"Duplicate element ID: {elem.eid}")
            self.eid2elem[elem.eid] = elem
        self.anchor_eids: List[int] = sorted(e.eid for e in elements if e.is_anchor)
        self._text_cache: Dict[int, str] = dict(synthetic_texts or {})
        for elem in elements:
            if elem.caption_text is not None:
                self._text_cache.setdefault(elem.caption_id, elem.caption_text)
        for eid in self._linked_eids(elements):
            if eid not in self.eid2elem:
                raise ValueError(f"Hard-pair link points to a missing element: {eid}")
        logger.info("ElementIndex: %d elements, %d anchors", len(self.eid2elem), len(self.anchor_eids))

    @staticmethod
    def _linked_eids(elements: Iterable[Element]) -> Iterable[int]:
        for elem in elements:
            yield from elem.i2t_pair_eids
            yield from elem.t2i_pair_eids

    def get_element(self, eid: int) -> Element:
        try:
            return self.eid2elem[eid]
        except KeyError:
            raise KeyError(f"Element ID {eid} not found in index") from None

    def get_text_content(self, caption_id: int) -> Optional[str]:
        """Text of an edited caption, or None."""
        return self._text_cache.get(caption_id)

    @property
    def num_elements(self) -> int:
        return len(self.eid2elem)


def build_base_collection(hf_dataset, verbose: bool = True) -> ElementCollection:
    """One anchor element per (image, caption) pair of the split, in dataset order."""
    collection = ElementCollection()
    imgids, captions_by_row, sentids_by_row = (
        metadata_column(hf_dataset, name) for name in ("imgid", "caption", "sentids")
    )
    rows = zip(imgids, captions_by_row, sentids_by_row, strict=True)
    for imgid, captions, sentids in tqdm(rows, total=len(imgids), desc="Building elements",
                                         disable=not verbose, dynamic_ncols=True):
        for caption, sentid in zip(captions, sentids, strict=True):
            collection.add_element(image_id=int(imgid), caption_id=int(sentid))
            collection.text_to_id[str(caption).strip()] = int(sentid)
            collection.next_text_id = max(collection.next_text_id, int(sentid) + 1)
    logger.info("Base elements: %d anchors from %d images", len(collection.elements), len(imgids))
    return collection


def build_element_index(
    hf_dataset,
    hard_pair_paths: Dict[str, Any],
    keep_sentence_ids: Optional[Set[int]] = None,
    verbose: bool = True,
) -> Tuple[ElementIndex, Dict[str, Any]]:
    """Build a split's element index and link the enabled hard pairs (in the given order).

    Args:
        hard_pair_paths: hard-pair type (`repair-i2t` / `repair-t2i`) -> records path.
        keep_sentence_ids: sentids of the split (T2I records of other captions are skipped).

    Returns the index and the loaded record sets by type (the T2I records also serve
    the edited images during training).
    """
    collection = build_base_collection(hf_dataset, verbose=verbose)
    image_ids = {e.image_id for e in collection.elements}
    loaded: Dict[str, Any] = {}
    for name, path in hard_pair_paths.items():
        logger.info("Linking hard pairs: %s (%s)", name, path)
        if name == "repair-i2t":
            loaded[name] = hard_pairs.I2TRecords(path)
            hard_pairs.link_i2t_pairs(collection, loaded[name].load(), keep_image_ids=image_ids, verbose=verbose)
        elif name == "repair-t2i":
            loaded[name] = hard_pairs.T2IRecords(path)
            hard_pairs.link_t2i_pairs(collection, loaded[name].load(), keep_sentence_ids=keep_sentence_ids,
                                      verbose=verbose)
        else:
            raise ValueError(f"Unknown hard-pair type: {name}; expected one of {hard_pairs.HARD_PAIR_TYPES}")
    index = ElementIndex(collection.elements, synthetic_texts=collection.synthetic_texts)
    return index, loaded
