# RePair synthesis output layout

`scripts/synthesize_i2t.py` writes one directory per work item:

```text
results/<dataset>/<retriever>/synthesis/<split>/i2t/<signature>/<status>/<item_id>/
```

Example (paper recipe):

```text
results/flickr30k/vit-base-patch32/synthesis/train/i2t/gtfirst_m3_qc1_gtqc1_fpqc1_pqc1_gemini-2.5-flash/success/6160_fp0/
```

| `<status>` | Meaning | Files |
| --- | --- | --- |
| `success` | Edited caption selected | `metadata.json`, `captions_compare.txt`, `directive.json` (+ `query_image.jpg` with `synthesis.save_query_images`) |
| `qc_rejected` | Rejected by QC | `analysis.json` |
| `failed` | API or parsing error | `error.json` |

The run directory also holds `config.json` (secrets redacted) and `summary.json`.
`metadata.json` records `item_id`, `status`, `qc_status`, `query_imgid`,
`query_image_filename`, `fp_sentid`, `fp_index`, `gt_rank`, `gt_caption`, `fp_caption`,
`edit_instruction`, `edited_caption`, `num_candidates`, `selection_cosine` (when PQC
selects among M > 1 edits) and `timestamp`.

## Signature

| Field | Meaning |
| --- | --- |
| `gt{first,last,all}` | `gt_caption_mode`: ground-truth caption(s) shown to QC |
| `m<M>` | `num_candidates`: edits per (query, FP) pair (paper M=3) |
| `qc{0,1}`, `gtqc{0,1}`, `fpqc{0,1}` | QC and its two criteria (GT consistency, FP validity) |
| `pqc{0,1}` | CLIP-cosine selection among the M edits (paper PQC) |
| `<model>` | Slug of the generator model |

`max_queries`, `fps_per_query`/`fp_top_pct` and `max_pairs` are run budgets: they are
recorded in `summary.json` but excluded from the signature, so a run can be extended
and resumed in place. When resuming, items that already have a directory under one of
`synthesis.resume_processed_statuses` (default: all statuses) are skipped;
`["success", "qc_rejected"]` re-runs failed items.

## Item ID

```text
<query_imgid>_fp<fp_index>        e.g. 6160_fp0
```

`fp_index` is the position of the FP caption in the query's FP pool. `prepare_i2t.py`
checks `fp_sentid` and the FP text against the pool.

## Paper setting

`config/synthesis/i2t.yaml`: Gemini 2.5 Flash, d_I2T=5 FP captions per query from a
K_mine=100 FP pool, one joint QC call, one instruction G(q, p_fp), M=3 edits and cosine
selection of one edit per pair.
