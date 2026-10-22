"""
Data pipeline for RePair training and retrieval.

- karpathy.py: local Karpathy caches (splits, official IDs, lazy image decoding)
- hard_pairs.py: RePair I2T/T2I records and how they link into the element graph
- graph.py: elements, anchors and hard-pair links of a training split
- datasets.py: pair and anchor datasets
- collator.py: anchors -> local 3x3 grids
"""

from .collator import GridCollator
from .datasets import AnchorDataset, PairDataset, load_anchor_dataset, load_pair_dataset
from .graph import Element, ElementIndex
from .karpathy import KarpathyCache

__all__ = [
    "AnchorDataset",
    "Element",
    "ElementIndex",
    "GridCollator",
    "KarpathyCache",
    "PairDataset",
    "load_anchor_dataset",
    "load_pair_dataset",
]
