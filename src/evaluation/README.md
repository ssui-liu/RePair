# Retrieval evaluation

`RetrievalEvaluator` computes I2T and T2I retrieval metrics over the complete candidate
set: Recall@K uses the best-ranked annotated positive; mean/median rank, MRR, mAP and
NDCG@K are also available. Features must be L2-normalized, and caption order must match
`caption_to_image`.

`SimilarityRows` computes the score matrix one chunk of query rows at a time: every
row still covers all candidates, so ranking is exact while peak memory stays bounded.
Each query row is sorted once and all metrics are derived from the ranks of its
relevant candidates.

FP pools (`scripts/retrieve.py --save-fp-pools`, `fp_pool.py`) are separate from metrics:

- I2T: `query_imgid`, `ground_truth.sentids` / `best_sentid` / `best_rank`,
  `false_positives.sentids` / `texts` / `similarity_scores`.
- T2I: `query_sentid`, `query_text`, `ground_truth.imgid` / `rank`,
  `false_positives.imgids` / `similarity_scores`.

Each pool `{i2t,t2i}_fp_pool.json` holds the `mining_top_k` highest-scoring **non-GT**
candidates (paper Eq. 1), also for queries whose GT ranks first; I2T excludes every
annotated caption of the query image. Metadata records `mining_strategy: top_k_non_gt`,
`mining_top_k`, dataset, split and the ID convention. The files are compact JSON (one
pool per training query).

```bash
python scripts/retrieve.py --split test
python scripts/retrieve.py --split train --save-fp-pools --mining-top-k 100
python scripts/retrieve.py --split train --fp-pools-only
```
