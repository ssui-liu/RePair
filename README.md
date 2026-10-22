# RePair: Turning Retrieval Failures into Counterfactual Hard Pairs

**EMNLP 2026 (Main Conference, Oral)** · [Paper (arXiv:2608.29604)](https://arxiv.org/abs/2608.29604)

RePair mines cross-modal confusions, verifies their validity, minimally edits the
false positives, and trains CLIP with global InfoNCE plus local contrastive grids.
This repository contains bidirectional retrieval and FP-pool mining, I2T synthesis,
and training with I2T/T2I hard pairs; the Flux T2I image editor, synthesized hard
pairs and trained checkpoints are not included. [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)
maps the method to the code.

## Install and test

Use Python 3.11 in a fresh environment. Install a PyTorch build suitable for your
CPU/CUDA environment, then the pinned library versions:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests run offline on generated images, a tiny randomly initialized CLIP model and
mock LLM responses. Full training uses CUDA and FP16; CPU runs need
`--device cpu --no-fp16`.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Prepare data | `scripts/prepare_dataset.py` | `data_cache/<dataset>/` |
| 2. Mine FP pools | `scripts/retrieve.py --save-fp-pools` | `results/<dataset>/<retriever>/retrieval/<split>/{i2t,t2i}_fp_pool.json` |
| 3a. I2T synthesis | `scripts/synthesize_i2t.py` | `results/<dataset>/<retriever>/synthesis/train/i2t/<run-name>/` |
| 3b. I2T hard pairs | `scripts/prepare_i2t.py` | `data_cache/RePair/<dataset>/I2T/synthesis_success.parquet` |
| 3c. T2I hard pairs | `scripts/prepare_t2i.py` (external edits) | `data_cache/RePair/<dataset>/T2I/part-*.parquet` |
| 4. Train | `scripts/train.py` | `checkpoints/<run>/final/` |
| 5. Evaluate | `scripts/retrieve.py --model-name …/final` | `checkpoints/<run>/retrieval/<split>/` |

## 1. Prepare local data

Supply legally obtained images and Karpathy annotations. The importer keeps all
captions of each selected image, validates IDs, writes sharded image bytes and
records the split IDs and annotation checksum in `data_cache/<dataset>/manifest.json`.
It refuses to overwrite a cache.

```bash
python scripts/prepare_dataset.py \
  --annotations /path/to/dataset_flickr30k.json \
  --images-root /path/to/flickr30k_images \
  --dataset-key flickr30k --output-root data_cache

python scripts/prepare_dataset.py \
  --annotations /path/to/dataset_coco.json \
  --images-root /path/to/coco2014 \
  --dataset-key coco30k --train-size 29000 --seed 42 \
  --output-root data_cache
```

COCO images may be in `train2014/` and `val2014/` below `--images-root`. By default only
COCO 2014 training images in the annotation's train split are eligible for sampling;
validation and test membership is preserved. `--train-image-ids ids.json` accepts an
explicit JSON list of Karpathy image IDs instead of sampling, and
`--keep-annotation-splits` imports a split file's train/val/test membership unchanged.

## 2. Mine FP pools

Run from the repository root. The default dataset is Flickr30K; use
`--dataset-key coco30k` consistently across commands to switch datasets.

```bash
python scripts/retrieve.py --split train --save-fp-pools --mining-top-k 100
```

For every query, this ranks all cross-modal candidates, excludes **all** annotated
positives, and keeps the top 100 non-GT candidates (the FP pool, paper Eq. 1), also for
queries whose GT already ranks first. Retrieval metrics are written as well
(`evaluation_results.json`); `--fp-pools-only` skips them.

## 3. Synthesize and import hard pairs

### I2T

The paper recipe uses Gemini 2.5 Flash: one joint GT/FP QC call, one instruction
`G(query, FP)` and M=3 caption edits per retained FP, of which the edit with the highest
CLIP cosine similarity to the original FP is kept (PQC, Eq. 3). The ground-truth caption
is used by QC only.

```bash
# Preview the resolved, redacted config: no API calls, model loading or output writes.
python scripts/synthesize_i2t.py --config config/synthesis/i2t.yaml --dry-run

# Set GEMINI_API_KEY in your environment, then run synthesis (uses paid API calls).
python scripts/synthesize_i2t.py --config config/synthesis/i2t.yaml --resume
```

The config uses Google's [OpenAI-compatible Gemini endpoint](https://ai.google.dev/gemini-api/docs/openai).
`OPENAI_API_KEY` is accepted as a fallback; `--base-url`/`OPENAI_BASE_URL` select another
compatible endpoint. Keep credentials in environment variables.

Pass the run-directory name printed by synthesis as `--run-name` (for the paper
recipe it is the one below):

```bash
python scripts/prepare_i2t.py --datasets flickr30k \
  --run-name gtfirst_m3_qc1_gtqc1_fpqc1_pqc1_gemini-2.5-flash
```

The conversion checks every record against the FP pool that produced it and never
overwrites an existing parquet file. Output layout and work-item IDs are described in
[OUTPUT_LAYOUT.md](config/synthesis/OUTPUT_LAYOUT.md).

### T2I

The T2I image editor is not part of this repository; its paper settings (Flux2-9B,
8 steps, 512×512, three candidates, CLIP-cosine selection) are listed in
[`config/synthesis/t2i_reference.yaml`](config/synthesis/t2i_reference.yaml). Edits that
were generated, quality-checked and selected elsewhere are imported from JSONL records:

```json
{"query_sentid": 123, "fp_imgid": 456, "gt_imgid": 789, "edited_image_path": "edits/123.png"}
```

`query_sentid` is the original caption ID; the two image IDs are original Karpathy
image IDs. Paths are relative to the JSONL file (absolute paths also work).

```bash
python scripts/prepare_t2i.py \
  --records /path/to/selected_edits.jsonl \
  --annotations data_cache/flickr30k/dataset_flickr30k.json \
  --output data_cache/RePair/flickr30k/T2I
```

The importer checks training-split membership, GT ownership and that every query/FP
pair appears once.

## 4. Train and evaluate

```bash
# I2T only; requires the I2T parquet from step 3.
python scripts/train.py --config config/training/repair_i2t.yaml \
  --seed 42 --output-dir checkpoints/repair-i2t-seed42

# Full model; requires BOTH I2T and T2I hard pairs.
python scripts/train.py --config config/training/repair_full.yaml \
  --seed 42 --output-dir checkpoints/repair-full-seed42

python scripts/retrieve.py --model-name checkpoints/repair-full-seed42/final \
  --split test --eval-tasks i2t t2i
```

The T2I-only recipe is `config/training/repair_t2i.yaml`. `--vanilla` disables the hard
pairs for an original-data baseline. W&B is disabled by default. Each run saves its
effective config, CLI arguments, summary metrics, `train.log`, and `final/` model
**with its processor**. Checkpoints saved without a processor can use
`--processor-name openai/clip-vit-base-patch32` at evaluation.

Each anchor's grid holds its original pair, the RePair I2T pair linked to its caption
and the RePair T2I pair linked to its image (paper Eq. 4); rows stay matched. Global
InfoNCE uses only the original anchors, and the symmetric diagonal grid loss is
averaged over the whole anchor batch (an anchor-only grid contributes zero, Eq. 5).
The loss weight is `loss.grid_lambda` (paper λ=0.5). Training logs report
`loss_global` and `loss_grid` separately.

The training configs use hyperparameters within the paper's search ranges (Appendix D).
`--batch-size` is the number of anchors per microbatch, i.e. the global InfoNCE
candidate set; gradient accumulation averages microbatches without enlarging that set.
See [configuration details](config/README.md).

## Citation

```bibtex
@inproceedings{liu2026repair,
  title     = {RePair: Turning Retrieval Failures into Counterfactual Hard Pairs},
  author    = {Liu, Siyi and Zhu, Xiaorong and Du, Enjun and Zuo, Xinyu and Duan, Lisheng and
               Liang, Haijin and Ma, Jin and Pu, Junfu and Zhang, Yongqi},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Code is under the [MIT License](LICENSE). Dataset and model licenses remain with
their respective owners.
