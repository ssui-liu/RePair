"""Full-ranking retrieval metrics (Recall@K, mean/median rank, MRR, mAP, NDCG@K)."""

import logging
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def relevant_ranks(similarity_row: torch.Tensor, relevant: Sequence[int]) -> List[int]:
    """Sorted 1-based ranks of the `relevant` candidates under a full descending sort."""
    order = torch.argsort(similarity_row, descending=True).cpu().numpy()
    position = np.empty_like(order)
    position[order] = np.arange(order.size)
    return sorted(int(position[candidate]) + 1 for candidate in relevant)


class Metrics:
    """Compute the enabled metrics of one retrieval direction."""

    SUPPORTED = ("recall", "mean_rank", "median_rank", "mrr", "map", "ndcg")

    def __init__(self, metrics_list: List[Union[str, Dict]] = None, top_k: List[int] = None, use_tqdm: bool = True):
        """
        Args:
            metrics_list: Metric names without @K (default: all supported).
            top_k: K values of Recall@K and NDCG@K (default: 1, 5, 10).
            use_tqdm: Show a progress bar over queries.
        """
        names = metrics_list if metrics_list is not None else list(self.SUPPORTED)
        names = [m if isinstance(m, str) else m.get("name") for m in names]
        self.metrics_list = [m for m in names if m in self.SUPPORTED]
        self.top_k = sorted(set(top_k if top_k is not None else [1, 5, 10]))
        self.use_tqdm = use_tqdm
        logger.info("Enabled metrics: %s; top_k: %s", ", ".join(self.metrics_list), self.top_k)

    def ranks_for_task(self, similarity, ground_truth: List[List[int]]) -> List[List[int]]:
        """Relevant-candidate ranks of every query; one sort per query row."""
        queries = range(similarity.shape[0])
        if self.use_tqdm:
            queries = tqdm(queries, desc="Ranking queries", leave=True)
        return [relevant_ranks(similarity[i], ground_truth[i]) for i in queries]

    def compute_for_task(self, similarity, ground_truth: List[List[int]]) -> Dict[str, float]:
        """`similarity` is [queries, candidates] (a tensor or `SimilarityRows`)."""
        ranks = self.ranks_for_task(similarity, ground_truth)
        best = np.array([r[0] for r in ranks])
        results: Dict[str, float] = {}
        if "recall" in self.metrics_list:
            for k in self.top_k:
                results[f"R@{k}"] = float(np.mean(best <= k)) * 100
        if "mean_rank" in self.metrics_list:
            results["MeanR"] = float(np.mean(best))
        if "median_rank" in self.metrics_list:
            results["MedianR"] = float(np.median(best))
        if "mrr" in self.metrics_list:
            results["MRR"] = float(np.mean(1.0 / best))
        if "map" in self.metrics_list:
            aps = [float(np.mean([j / r for j, r in enumerate(query_ranks, start=1)])) for query_ranks in ranks]
            results["mAP"] = float(np.mean(aps))
        if "ndcg" in self.metrics_list:
            for k in self.top_k:
                ndcgs = []
                for query_ranks in ranks:
                    dcg = sum(1.0 / np.log2(r + 1) for r in query_ranks if r <= k)
                    idcg = np.sum([1.0 / np.log2(i + 1) for i in range(1, min(len(query_ranks), k) + 1)])
                    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
                results[f"NDCG@{k}"] = float(np.mean(ndcgs))
        return results
