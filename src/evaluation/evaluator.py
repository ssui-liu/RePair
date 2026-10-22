"""Bidirectional full-ranking retrieval evaluation and FP-pool export."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.data.karpathy import metadata_column

from .fp_pool import i2t_fp_pool, t2i_fp_pool, write_fp_pool
from .metrics import Metrics, relevant_ranks
from .similarity import SimilarityRows

logger = logging.getLogger(__name__)

_TASK_ALIASES = {"i2t": "i2t", "image_to_text": "i2t", "t2i": "t2i", "text_to_image": "t2i"}


class RetrievalEvaluator:
    """Image-to-text and text-to-image retrieval over a whole split."""

    def __init__(self, metrics_list: List[Union[str, Dict]] = None, top_k: List[int] = None,
                 tasks: List[str] = None, use_tqdm: bool = True):
        """
        Args:
            metrics_list: Metric names without @K (see `Metrics`).
            top_k: K values for Recall@K / NDCG@K.
            tasks: Directions to run: "i2t"/"image_to_text", "t2i"/"text_to_image" (default: both).
            use_tqdm: Show progress bars.
        """
        self.metrics = Metrics(metrics_list, top_k, use_tqdm=use_tqdm)
        self.tasks = []
        for task in tasks or ["i2t", "t2i"]:
            if task in _TASK_ALIASES:
                self.tasks.append(_TASK_ALIASES[task])
            else:
                logger.warning("Ignoring unsupported task type: %s", task)
        self.tasks = self.tasks or ["i2t", "t2i"]

    @staticmethod
    def _ground_truth_i2t(caption_to_image: List[int]) -> List[List[int]]:
        ground_truth: List[List[int]] = [[] for _ in range(len(set(caption_to_image)))]
        for caption_idx, image_idx in enumerate(caption_to_image):
            ground_truth[image_idx].append(caption_idx)
        return ground_truth

    def evaluate(
        self,
        image_features: torch.Tensor,
        caption_features: torch.Tensor,
        caption_to_image: List[int],
        model_name: str = None,
        dataset=None,
        all_captions: Optional[List[str]] = None,
        compute_metrics: bool = True,
        save_tail: bool = False,
        tail_k: int = 10,
        tail_output_dir: Optional[str] = None,
        save_fp_pools: bool = False,
        i2t_pool_path: Optional[str] = None,
        t2i_pool_path: Optional[str] = None,
        dataset_name: str = "flickr30k",
        split: str = "test",
        id_mapping: Optional[Dict] = None,
        mining_top_k: int = 100,
    ) -> Dict:
        """Compute metrics and optionally write FP pools / worst T2I cases.

        Args:
            image_features: [N_images, D] L2-normalized features (row = dataset row).
            caption_features: [N_texts, D] L2-normalized features.
            caption_to_image: Image row of every caption.
            dataset, all_captions: Needed for FP pools and tail cases.
            save_fp_pools: Write the top-`mining_top_k` non-GT pools (Eq. 1) of both
                directions to `i2t_pool_path` / `t2i_pool_path`.
            id_mapping: `sentid_list` / `idx_to_imgid` to export Karpathy IDs.
        """
        if image_features.shape[0] != len(set(caption_to_image)):
            raise ValueError(f"{image_features.shape[0]} image features but "
                             f"{len(set(caption_to_image))} images in caption_to_image")
        if caption_features.shape[0] != len(caption_to_image):
            raise ValueError(f"{caption_features.shape[0]} caption features but "
                             f"{len(caption_to_image)} captions in caption_to_image")
        mining = save_fp_pools and dataset is not None and all_captions is not None
        if mining and mining_top_k < 1:
            raise ValueError("mining_top_k must be positive")
        filenames = metadata_column(dataset, "filename") if mining or (save_tail and dataset is not None) else None
        pool_args = dict(model_name=model_name or "unknown", dataset_name=dataset_name, split=split,
                         id_mapping=id_mapping)
        logger.info("Evaluating %d images and %d texts", image_features.shape[0], caption_features.shape[0])

        results_by_task: Dict[str, Dict[str, float]] = {}
        if "i2t" in self.tasks:
            sim_i2t = SimilarityRows(image_features, caption_features)
            if compute_metrics:
                results_by_task["Image-to-Text"] = self.metrics.compute_for_task(
                    sim_i2t, self._ground_truth_i2t(caption_to_image)
                )
            if mining:
                samples = i2t_fp_pool(sim_i2t, caption_to_image, all_captions, filenames, mining_top_k, id_mapping)
                write_fp_pool(i2t_pool_path or "./results/i2t_fp_pool.json", "i2t", samples, mining_top_k,
                              **pool_args)

        if "t2i" in self.tasks:
            sim_t2i = SimilarityRows(caption_features, image_features)
            if compute_metrics:
                results_by_task["Text-to-Image"] = self.metrics.compute_for_task(
                    sim_t2i, [[image_idx] for image_idx in caption_to_image]
                )
            if mining:
                samples = t2i_fp_pool(sim_t2i, caption_to_image, all_captions, filenames, mining_top_k, id_mapping)
                write_fp_pool(t2i_pool_path or "./results/t2i_fp_pool.json", "t2i", samples, mining_top_k,
                              **pool_args)
            if save_tail and dataset is not None and all_captions is not None:
                self._save_tail_t2i(sim_t2i, caption_to_image, dataset, filenames, all_captions,
                                    tail_k, tail_output_dir or "./results/tail_t2i", id_mapping)

        results = {"model_name": model_name, "metrics": results_by_task}
        if compute_metrics:
            self.print_results(results)
        return results

    def evaluate_model(
        self,
        dataset,
        all_captions: List[str],
        caption_to_image: List[int],
        processor,
        model,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> Dict:
        """Encode a split with an in-memory model (used for validation during training)."""
        from src.data.feature_extractor import CLIPFeatureExtractor

        extractor = CLIPFeatureExtractor(device=device, batch_size=batch_size, model=model, processor=processor)
        return self.evaluate(
            image_features=extractor.encode_images(dataset),
            caption_features=extractor.encode_texts(all_captions),
            caption_to_image=caption_to_image,
            model_name=getattr(model.config, "name_or_path", None),
        )

    @staticmethod
    def print_results(results: Dict) -> None:
        lines = [f"Retrieval results ({results.get('model_name')})"]
        for task_name, metrics in results["metrics"].items():
            lines.append(f"{task_name}:")
            for key in sorted(k for k in metrics if k.startswith("R@")):
                lines.append(f"  {key}: {metrics[key]:.2f}%")
            for key in ("MedianR", "MeanR"):
                if key in metrics:
                    lines.append(f"  {key}: {metrics[key]:.2f}")
            for key in sorted(k for k in metrics if k in ("mAP", "MRR") or k.startswith("NDCG@")):
                lines.append(f"  {key}: {metrics[key]:.4f}")
        logger.info("\n".join(lines))

    @staticmethod
    def save_results(results: Dict, save_path: str = "./results/evaluation_results.json") -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Results saved to: %s", save_path)

    def _save_tail_t2i(self, sim_t2i, caption_to_image, dataset, filenames, all_captions,
                       tail_k, output_dir, id_mapping=None) -> None:
        """Save the `tail_k` text queries with the worst GT rank (images, texts, summary.json)."""
        sentid_list = id_mapping.get("sentid_list") if id_mapping else None
        idx_to_imgid = id_mapping.get("idx_to_imgid") if id_mapping else None
        imgid = (lambda i: int(idx_to_imgid[i])) if idx_to_imgid else int
        captions_by_row = metadata_column(dataset, "caption")
        ranks = np.array([relevant_ranks(sim_t2i[i], [gt])[0] for i, gt in enumerate(caption_to_image)])
        out_root = Path(output_dir)
        for position, cap_idx in enumerate(np.argsort(ranks)[::-1][:tail_k], start=1):
            rank, row = int(ranks[cap_idx]), sim_t2i[cap_idx]
            gt_idx = caption_to_image[cap_idx]
            top10 = [int(i) for i in torch.argsort(row, descending=True)[:10]]
            case_dir = out_root / f"{position:02d}_rank{rank}"
            (case_dir / "prediction").mkdir(parents=True, exist_ok=True)
            dataset[gt_idx]["image"].save(case_dir / "gt_image.jpg")
            dataset[top10[0]]["image"].save(case_dir / "pred_image.jpg")
            (case_dir / "query_text.txt").write_text(str(all_captions[cap_idx]), encoding="utf-8")
            (case_dir / "gt_texts.txt").write_text("\n".join(map(str, captions_by_row[gt_idx])), encoding="utf-8")
            for j, img_idx in enumerate(top10, start=1):
                stem = case_dir / "prediction" / f"{j:02d}_{filenames[img_idx]}"
                dataset[img_idx]["image"].save(f"{stem}.jpg")
                Path(f"{stem}.txt").write_text("\n".join(map(str, captions_by_row[img_idx])), encoding="utf-8")
            summary = {
                "query_sentid": int(sentid_list[cap_idx]) if sentid_list else int(cap_idx),
                "gt_imgid": imgid(gt_idx),
                "top1_imgid": imgid(top10[0]),
                "caption_index": int(cap_idx),
                "gt_image_index": int(gt_idx),
                "top1_image_index": top10[0],
                "rank": rank,
                "score_gt": float(row[gt_idx]),
                "score_top1": float(row[top10[0]]),
                "top10_imgids": [imgid(i) for i in top10],
                "top10_scores": [float(row[i]) for i in top10],
                "top10_filenames": [filenames[i] for i in top10],
                **{f"hit@{k}": rank <= k for k in self.metrics.top_k},
            }
            (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
