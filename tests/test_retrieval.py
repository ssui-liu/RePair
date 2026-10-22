import torch

from scripts.retrieve import parse_args
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.metrics import Metrics
from src.evaluation.similarity import SimilarityRows


def test_chunked_full_ranking_matches_dense_for_all_metrics():
    generator = torch.Generator().manual_seed(15)
    q = torch.randn(7, 8, generator=generator)
    c = torch.randn(13, 8, generator=generator)
    dense = q @ c.T
    rows = SimilarityRows(q, c, chunk_size=2)
    for i in range(7):
        torch.testing.assert_close(rows[i], dense[i])
        assert rows._scores.shape[0] <= 2
    truth = [[i, i + 1] for i in range(7)]
    metrics = Metrics(use_tqdm=False)
    assert metrics.compute_for_task(rows, truth) == metrics.compute_for_task(dense, truth)


def test_mining_includes_gt_first_queries_and_excludes_all_positives(tmp_path):
    # Two GT captions for image 0; its GT ranks first, yet its non-GT pool must exist.
    images = torch.eye(3)
    captions = torch.tensor([[1., 0., 0.], [.9, .1, 0.], [0., 1., 0.], [0., 0., 1.]])
    data = [{"filename": f"{i}.jpg"} for i in range(3)]
    evaluator = RetrievalEvaluator(metrics_list=["recall"], use_tqdm=False)
    result = evaluator.evaluate(images, captions, [0, 0, 1, 2], dataset=data,
                                all_captions=["a", "b", "c", "d"],
                                save_fp_pools=True, mining_top_k=1,
                                t2i_pool_path=str(tmp_path / "t2i.json"),
                                i2t_pool_path=str(tmp_path / "i2t.json"))
    import json
    i2t = json.loads((tmp_path / "i2t.json").read_text())
    t2i = json.loads((tmp_path / "t2i.json").read_text())
    assert result["metrics"]["Image-to-Text"]["R@1"] == 100
    assert i2t["metadata"]["mining_strategy"] == "top_k_non_gt"
    assert i2t["samples"][0]["ground_truth"]["best_rank"] == 1
    for sample in i2t["samples"]:
        assert len(sample["false_positives"]["sentids"]) == 1
        assert not set(sample["false_positives"]["sentids"]) & set(sample["ground_truth"]["sentids"])
    for sample in t2i["samples"]:
        assert len(sample["false_positives"]["imgids"]) == 1
        assert sample["ground_truth"]["imgid"] not in sample["false_positives"]["imgids"]


def test_retrieval_cli_defaults_compute_metrics_and_honor_config():
    args = parse_args([])
    assert not args.fp_pools_only
    assert args.save_fp_pools is None and args.split is None
    assert parse_args(["--no-save-fp-pools"]).save_fp_pools is False
    assert parse_args(["--fp-pools-only"]).fp_pools_only is True


def _reference_metrics(similarity, truth, top_k=(1, 5, 10)):
    """Pre-refactor algorithm: separate full sorts for ranks, mAP and every NDCG@K."""
    import numpy as np
    sorted_rows = [torch.argsort(similarity[i], descending=True).numpy() for i in range(similarity.shape[0])]
    ranks = [sorted(int(np.where(order == g)[0][0]) + 1 for g in gts)
             for order, gts in zip(sorted_rows, truth, strict=True)]
    best = np.array([r[0] for r in ranks])
    out = {f"R@{k}": float(np.mean(best <= k)) * 100 for k in top_k}
    out.update(MeanR=float(np.mean(best)), MedianR=float(np.median(best)), MRR=float(np.mean(1.0 / best)))
    out["mAP"] = float(np.mean([np.mean([j / r for j, r in enumerate(rk, 1)]) for rk in ranks]))
    for k in top_k:
        values = []
        for rk in ranks:
            dcg = 0.0
            for r in rk:
                if r <= k:
                    dcg += 1.0 / np.log2(r + 1)
            idcg = np.sum([1.0 / np.log2(i + 1) for i in range(1, min(len(rk), k) + 1)])
            values.append(dcg / idcg if idcg > 0 else 0.0)
        out[f"NDCG@{k}"] = float(np.mean(values))
    return out


def test_single_sort_metrics_equal_the_reference_algorithm():
    generator = torch.Generator().manual_seed(4)
    images, captions = torch.randn(20, 8, generator=generator), torch.randn(60, 8, generator=generator)
    caption_to_image = [i // 3 for i in range(60)]
    truth_i2t = RetrievalEvaluator._ground_truth_i2t(caption_to_image)
    assert truth_i2t[2] == [6, 7, 8]
    metrics = Metrics(use_tqdm=False)
    for sim, truth in ((images @ captions.T, truth_i2t), (captions @ images.T, [[i] for i in caption_to_image])):
        assert metrics.compute_for_task(SimilarityRows(sim, torch.eye(sim.shape[1]), chunk_size=7), truth) \
            == _reference_metrics(sim, truth)


def test_mining_pool_equals_full_sort_reference(tmp_path):
    import json
    generator = torch.Generator().manual_seed(9)
    images, captions = torch.randn(6, 4, generator=generator), torch.randn(18, 4, generator=generator)
    caption_to_image = [i // 3 for i in range(18)]
    RetrievalEvaluator(metrics_list=["recall"], use_tqdm=False).evaluate(
        images, captions, caption_to_image, dataset=[{"filename": f"{i}.jpg"} for i in range(6)],
        all_captions=[f"c{i}" for i in range(18)], compute_metrics=False, save_fp_pools=True,
        mining_top_k=4, t2i_pool_path=str(tmp_path / "t2i.json"), i2t_pool_path=str(tmp_path / "i2t.json"))
    i2t = json.loads((tmp_path / "i2t.json").read_text())["samples"]
    t2i = json.loads((tmp_path / "t2i.json").read_text())["samples"]
    for img, sample in enumerate(i2t):
        order = torch.argsort(images[img] @ captions.T, descending=True).tolist()
        assert sample["false_positives"]["sentids"] == [c for c in order if c // 3 != img][:4]
        assert sample["ground_truth"]["best_rank"] == min(order.index(c) for c in range(3 * img, 3 * img + 3)) + 1
    for cap, sample in enumerate(t2i):
        order = torch.argsort(captions[cap] @ images.T, descending=True).tolist()
        assert sample["false_positives"]["imgids"] == [i for i in order if i != cap // 3][:4]
        assert sample["ground_truth"]["rank"] == order.index(cap // 3) + 1
