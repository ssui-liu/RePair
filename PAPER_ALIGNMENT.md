# Paper-to-code mapping

Equation numbers follow the EMNLP 2026 camera-ready version of **RePair: Turning
Retrieval Failures into Counterfactual Hard Pairs**
([arXiv:2608.29604](https://arxiv.org/abs/2608.29604)).

## Method

| Paper | Code |
| --- | --- |
| FP pool: top K_mine=100 non-GT candidates per query, all positives excluded (Eq. 1) | `src/evaluation/fp_pool.py` |
| Retain d_I2T=5 and d_T2I=1 (Eq. 2) | `config/synthesis/i2t.yaml`, `config/synthesis/t2i_reference.yaml` |
| Joint QC: GT consistency and FP validity in one call | `src/synthesis/i2t/steps.py`, `prompts.py` |
| Instruction G(q, p_fp); the GT is used only by QC | `src/synthesis/i2t/workflow.py`, `prompts.py` |
| M=3 edits, the one with maximal CLIP cosine to p_fp kept (PQC, Eq. 3) | `workflow.py`, `cosine_selector.py` |
| Grid I = [I^a, I^+(T^hn), I^hn], T = [T^a, T^hn, T^+(I^hn)] (Eq. 4) | `src/data/hard_pairs.py`, `graph.py`, `collator.py` |
| Global InfoNCE over original anchors (Appendix Eq. 8) | `src/training/losses.py` (`global_infonce_loss`) |
| Grid loss: symmetric diagonal CE, sum over grids divided by B (Eq. 5) | `losses.py` (`grid_loss`) |
| L = L_global + λ L_grid, λ=0.5 (Eq. 6) | `losses.py` (`RePairLoss`), training configs |
| Full-ranking retrieval evaluation | `src/evaluation/similarity.py`, `metrics.py` |

**How grid rows are formed.** Both neighbor rows of an anchor (I^a, T^a) come from
synthesis records in which the anchor itself was the retrieved false positive:

- a `repair-i2t` record (query image I_q, FP caption T^a, selected edit T_edit) gives the row
  (I^hn, T^+(I^hn)) = (I_q, T_edit);
- a `repair-t2i` record (query caption T_q, FP image I^a, selected edit I_edit) gives the row
  (I^+(T^hn), T^hn) = (I_edit, T_q). T2I records are linked from the first caption
  anchor of their FP image, so each record enters exactly one anchor's grid.

Each edited item is therefore the output of the record in which the other element
of its row served as the query (the paper's footnote on I^hn/T^hn). When several records link
the same anchor, one of each kind is sampled per epoch. An anchor without a link of
some kind gets a smaller grid; no unverified neighbor is inserted to fill it.

## Naming glossary

| Paper | Code | Config / CLI |
| --- | --- | --- |
| FP pool F(q), K_mine | `fp_pool.py`, `{i2t,t2i}_fp_pool.json` | `evaluation.mining_top_k`, `--save-fp-pools`, `--mining-top-k` |
| d_I2T, d_T2I | `WorkItemBuilder.fps_per_query` | `synthesis.fps_per_query`, `--fps-per-query` |
| QC criteria 1 / 2 | `I2TSteps.quality_check` | `quality_control.enable_gt_check` / `enable_fp_check` |
| G(q, p_fp) | `I2TSteps.edit_instruction` | — |
| M edit candidates | `I2TWorkflow.num_candidates` | `synthesis.num_candidates`, `--num-candidates` |
| PQC (Eq. 3) | `CosineSelector` | `quality_control.enable_pqc`, `--no-pqc` |
| I2T / T2I hard pairs | `repair-i2t` / `repair-t2i`, `Element.i2t_pair_eids` / `t2i_pair_eids` | `hard_pairs.enabled`, `--hard-pairs`, `--vanilla` |
| Anchor pair (I^a, T^a) | `Element.is_anchor`, `AnchorDataset` | — |
| 3×3 grid (Eq. 4) | `GridCollator`, batch keys `grid_*` | `sampling.pairs_per_direction` (1 → 3×3) |
| L_global, L_grid, λ | `global_infonce_loss`, `grid_loss`, `RePairLoss.grid_lambda` | `loss.grid_lambda`, `--grid-lambda` |

## Settings

| Setting | Paper | Code |
| --- | --- | --- |
| Backbone / input | CLIP ViT-B/32, 224×224, 77 tokens | `model.model_name` (`openai/clip-vit-base-patch32`) and its processor |
| QC, instruction, I2T editing | Gemini 2.5 Flash | `api` section of `config/synthesis/i2t.yaml` |
| T2I editor | Flux2-9B, 8 steps, 512×512 | `config/synthesis/t2i_reference.yaml` (settings only; editor not included) |
| Grid weight | λ=0.5 | `loss.grid_lambda` in all training configs |
| Optimizer / scheduler | AdamW, weight decay 0.1, warmup 0.1, cosine, FP16 | Training configs |
| LR search | 1e-5, 5e-6, 2e-6, 1e-6 | Training configs: 2e-6 |
| Batch / epochs search | 64, 128, 256, 512, 1024 / 3–10 | Training configs: 256 anchors per update (128×2 or 64×4), 10 epochs |

`--batch-size` anchors form the global InfoNCE candidate set of a microbatch; gradient
accumulation averages several microbatches per update without enlarging that set.
