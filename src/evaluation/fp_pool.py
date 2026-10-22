"""
False-positive pools F(q) (paper Eq. 1) and their files.

For each query all cross-modal candidates are ranked, every annotated positive is
excluded, and the K_mine highest-scoring remaining candidates are kept, also for
queries whose GT already ranks first. Pools are written by `scripts/retrieve.py`:

    <results>/<dataset>/<retriever>/retrieval/<split>/{i2t,t2i}_fp_pool.json

Files carry `mining_strategy: top_k_non_gt`, which synthesis checks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

DIRECTIONS = {"i2t": "image_to_text", "t2i": "text_to_image"}


def fp_pool_path(retrieval_dir: Path, direction: str) -> Path:
    return Path(retrieval_dir) / f"{direction}_fp_pool.json"


def find_fp_pool(results_dir: Path, dataset_key: str, retriever: str, split: str, direction: str) -> Path:
    """Pool file of a retriever's run on a split."""
    return fp_pool_path(Path(results_dir) / dataset_key / retriever / "retrieval" / split, direction)


def load_fp_pool(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FP pool not found: {path} (run scripts/retrieve.py --save-fp-pools)")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "samples" not in data:
        raise ValueError(f"Invalid FP-pool file (no 'samples'): {path}")
    logger.info("Loaded %d FP-pool samples from %s", len(data["samples"]), path)
    return data


def _ranked(similarity_row: torch.Tensor) -> np.ndarray:
    return torch.argsort(similarity_row, descending=True).cpu().numpy()


def _top_k_excluding(order: np.ndarray, excluded: Sequence[int], k: int) -> List[int]:
    """First k entries of `order` not in `excluded` (only the head of `order` is scanned)."""
    excluded = set(excluded)
    return [int(i) for i in order[:k + len(excluded)] if int(i) not in excluded][:k]


def _scores(similarity_row: torch.Tensor, indices: List[int]) -> List[float]:
    return similarity_row[torch.as_tensor(indices, dtype=torch.long)].tolist() if indices else []


def i2t_fp_pool(similarity, caption_to_image: List[int], all_captions: List[str], image_filenames: List[str],
                k: int, id_mapping: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Per image query: its GT captions (best rank) and the top-k non-GT captions.

    `similarity` is [N_images, N_texts]; `id_mapping` (sentid_list, idx_to_imgid)
    converts row indices to Karpathy IDs.
    """
    sentid_list = id_mapping.get("sentid_list") if id_mapping else None
    idx_to_imgid = id_mapping.get("idx_to_imgid") if id_mapping else None
    captions_of_image: Dict[int, List[int]] = {}
    for caption_idx, image_idx in enumerate(caption_to_image):
        captions_of_image.setdefault(image_idx, []).append(caption_idx)

    def sentid(caption_idx: int) -> int:
        return int(sentid_list[caption_idx]) if sentid_list else caption_idx

    samples = []
    for img_idx in range(similarity.shape[0]):
        row = similarity[img_idx]
        order = _ranked(row)
        gt_indices = captions_of_image[img_idx]
        position = np.empty_like(order)
        position[order] = np.arange(order.size)
        best_rank, best_idx = min((int(position[c]) + 1, c) for c in gt_indices)
        fp_indices = _top_k_excluding(order, gt_indices, k)
        samples.append({
            "query_imgid": int(idx_to_imgid[img_idx] if idx_to_imgid else img_idx),
            "query_image_filename": image_filenames[img_idx],
            "ground_truth": {
                "sentids": [sentid(c) for c in gt_indices],
                "best_sentid": sentid(best_idx),
                "best_text": all_captions[best_idx],
                "best_similarity_score": float(row[best_idx]),
                "best_rank": best_rank,
            },
            "false_positives": {
                "sentids": [sentid(c) for c in fp_indices],
                "texts": [all_captions[c] for c in fp_indices],
                "similarity_scores": _scores(row, fp_indices),
            },
        })
    return samples


def t2i_fp_pool(similarity, caption_to_image: List[int], all_captions: List[str], image_filenames: List[str],
                k: int, id_mapping: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Per text query: its GT image (rank, score) and the top-k non-GT images (`similarity`: [N_texts, N_images])."""
    sentid_list = id_mapping.get("sentid_list") if id_mapping else None
    idx_to_imgid = id_mapping.get("idx_to_imgid") if id_mapping else None
    samples = []
    for query_idx in range(similarity.shape[0]):
        row = similarity[query_idx]
        order = _ranked(row)
        gt_idx = caption_to_image[query_idx]
        fp_indices = _top_k_excluding(order, [gt_idx], k)
        samples.append({
            "query_sentid": int(sentid_list[query_idx] if sentid_list else query_idx),
            "query_text": all_captions[query_idx],
            "ground_truth": {
                "imgid": int(idx_to_imgid[gt_idx] if idx_to_imgid else gt_idx),
                "image_filename": image_filenames[gt_idx],
                "similarity_score": float(row[gt_idx]),
                "rank": int(np.flatnonzero(order == gt_idx)[0]) + 1,
            },
            "false_positives": {
                "imgids": [int(idx_to_imgid[i]) if idx_to_imgid else i for i in fp_indices],
                "similarity_scores": _scores(row, fp_indices),
            },
        })
    return samples


def write_fp_pool(path: Path, direction: str, samples: List[Dict[str, Any]], k: int, model_name: str,
                  dataset_name: str, split: str, id_mapping: Optional[Dict] = None) -> None:
    """Write a pool file (compact JSON: it holds k candidates for every training query)."""
    if k < 1:
        raise ValueError("mining_top_k must be positive")
    metadata = {
        "model_name": model_name,
        "dataset": dataset_name,
        "split": split,
        "task": DIRECTIONS[direction],
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(samples),
        "mining_strategy": "top_k_non_gt",
        "mining_top_k": k,
        "id_type": "karpathy_official" if id_mapping else "sequential",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "samples": samples}, ensure_ascii=False), encoding="utf-8")
    logger.info("%s FP pool saved to: %s", direction.upper(), path)
