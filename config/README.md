# Configuration guide

Run commands from the repository root. Relative data-cache and output paths are
resolved against the repository root. Explicit CLI arguments override the YAML;
omitted flags keep the YAML values.

| File | Used by | Purpose |
| --- | --- | --- |
| `retrieval.yaml` | `scripts/retrieve.py` | Both retrieval directions; metrics on, FP-pool export off |
| `synthesis/i2t.yaml` | `scripts/synthesize_i2t.py` | Paper I2T recipe: d_I2T=5 FPs per query, QC, M=3 edits, cosine PQC |
| `synthesis/i2t_demo.yaml`, `synthesis/i2t_quick_test.yaml` | `scripts/synthesize_i2t.py` | Small demo / quick end-to-end check; check their limits first |
| `synthesis/t2i_reference.yaml` | — | Paper reference settings of the external T2I editor; not a runner config |
| `training/repair_full.yaml` | `scripts/train.py` | Full model: I2T + T2I hard pairs |
| `training/repair_i2t.yaml`, `training/repair_t2i.yaml` | `scripts/train.py` | One hard-pair direction |

Every stage names its dataset `dataset.key` (`flickr30k` | `coco30k`) and its cache root
`dataset.cache_dir`. Switch every stage together with
`--dataset-key coco30k --data-cache /path/to/cache`.

## Training

Training YAMLs are validated strictly: an unknown or misspelled key is an error, so no
setting is silently ignored. Frequently used overrides: `--seed`, `--output-dir`,
`--batch-size`, `--gradient-accumulation-steps`, `--epochs`, `--learning-rate`,
`--grid-lambda`, `--dataloader-num-workers`, `--fp16/--no-fp16`, `--hard-pairs` or
`--vanilla`. Use a distinct output directory for every seed and recipe.

- `hard_pairs.enabled` lists `repair-i2t` / `repair-t2i`; records default to
  `<cache_dir>/RePair/<dataset>/{I2T,T2I}`, `hard_pairs.paths` overrides them.
- `--batch-size` is the number of original anchors per microbatch, i.e. the global
  InfoNCE candidate set. Gradient accumulation averages microbatch gradients before an
  update but does not enlarge the InfoNCE candidate set.
- `dataloader_num_workers` (4 in the recipes) loads images and builds grids in worker
  processes; every worker draws its own reproducible pair-sampling stream.
- `sampling.pairs_per_direction: 1` samples one hard pair of each direction per anchor
  (3x3 grids); `loss.grid_lambda` is the paper's λ.
- W&B is off by default; `--use-wandb` / `--no-use-wandb` override it.
- Each run saves `train.log`, `effective_config.json`, `cli_args.json`, `results.json`
  and a reloadable `final/` checkpoint with its processor. Training logs report the mean
  `loss_global` and `loss_grid` terms next to the total loss; validation retrieval
  metrics appear as `eval_i2t/*` and `eval_t2i/*` every `eval_steps` optimizer steps.

## Retrieval and FP pools

A local checkpoint writes to `<checkpoint>/../retrieval/<split>/`; a hub model writes
to `results/<dataset>/<model>/retrieval/<split>/`. `--output-dir` overrides the base.
`--save-fp-pools` writes `{i2t,t2i}_fp_pool.json` next to the metrics;
`--fp-pools-only` writes the pools only. `evaluation.mining_top_k` /
`--mining-top-k` (paper K_mine=100) is independent of the Recall@K cutoffs.

## Synthesis

The key is read from `GEMINI_API_KEY` (fallback `OPENAI_API_KEY`); `OPENAI_BASE_URL`
or `--base-url` selects another OpenAI-compatible endpoint (`api` section). Secrets are
redacted from saved configs. `--dry-run` prints the resolved config without API calls,
model loading or output writes. Transient API errors are retried with exponential
backoff as set in the `retry` section. `paths.retriever` names the retriever directory whose
FP pool is used. Run directories are named by the settings signature described in
[OUTPUT_LAYOUT.md](synthesis/OUTPUT_LAYOUT.md), which also holds the resume record;
keep separate result roots for distinct retrievers.

## Cache layout

```text
data_cache/<dataset>/dataset_<dataset>.json
data_cache/<dataset>/<train|val|test>/images_parquet/part-*.parquet
data_cache/RePair/<dataset>/I2T/synthesis_success.parquet
data_cache/RePair/<dataset>/T2I/part-*.parquet
```

See the [README](../README.md) for the commands and
[paper alignment](../PAPER_ALIGNMENT.md) for the paper-to-code mapping.
