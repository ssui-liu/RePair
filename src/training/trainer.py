"""
RePair CLIP fine-tuning (paper Sec. 3.3, Algorithm 2).

Every batch holds B original anchor pairs and their local grids (`GridCollator`).
The loss is the global InfoNCE over the anchors plus lambda times the grid loss
(`RePairLoss`, paper Eq. 6). Full-ranking validation retrieval runs before
training, every `eval_steps` optimizer steps and after training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import transformers
from transformers import AutoProcessor, CLIPModel, Trainer, TrainingArguments

from src.data import GridCollator, KarpathyCache, load_anchor_dataset
from src.evaluation.evaluator import RetrievalEvaluator
from src.training.callbacks import RetrievalEvalCallback, format_retrieval_logs
from src.training.config import TrainingJobConfig
from src.training.losses import RePairLoss

logger = logging.getLogger(__name__)


def _check_gradient_accumulation(steps: int) -> None:
    # transformers 4.46-4.47 skip the 1/steps loss scaling for custom losses without
    # labels, so accumulated microbatch gradients would be summed instead of averaged.
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if steps > 1 and (major, minor) < (4, 48):
        raise RuntimeError(
            f"transformers {transformers.__version__} sums gradients over accumulation steps for "
            "this loss; install the pinned version from requirements.txt (>=4.48)."
        )


class RePairTrainer(Trainer):
    """Trainer whose loss is the RePair objective on `GridCollator` batches."""

    def __init__(self, *args, loss_fn, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = self.loss_fn(model, inputs)
        return (loss, None) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if "loss" in logs:  # training log: add the mean global and grid terms since the last log
            logs = {**logs, **self.loss_fn.consume_logs()}
        super().log(logs, *args, **kwargs)


def train_repair(config: TrainingJobConfig) -> Dict[str, Any]:
    """Train, save `<output_dir>/final` (model + processor) and return a JSON-serializable summary."""
    if not isinstance(config, TrainingJobConfig):
        config = TrainingJobConfig.model_validate(config)
    training = config.training
    _check_gradient_accumulation(training.gradient_accumulation_steps)
    device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hard_pairs = config.hard_pairs.active()
    logger.info("Device: %s | hard pairs: %s", device, list(hard_pairs) or "none (vanilla)")

    cache = KarpathyCache(config.dataset.key, config.dataset.cache_dir)
    train_ds, train_index = load_anchor_dataset(
        cache,
        training.train_split,
        hard_pairs,
        subsample=config.subsample_train,
        shuffle_subsample=config.shuffle_subsample,
        seed=config.seed,
        verbose=config.verbose,
    )
    eval_split = cache.load_split(training.eval_split)
    all_captions, caption_to_image, _ = cache.caption_mapping(eval_split)

    processor = AutoProcessor.from_pretrained(config.model.model_name)
    model = CLIPModel.from_pretrained(config.model.model_name).to(device)
    evaluator = RetrievalEvaluator(
        metrics_list=training.eval_metrics, top_k=training.eval_top_k, tasks=training.eval_tasks, use_tqdm=False
    )

    def evaluate(eval_model) -> Dict:
        return evaluator.evaluate_model(
            dataset=eval_split,
            all_captions=all_captions,
            caption_to_image=caption_to_image,
            processor=processor,
            model=eval_model,
            device=device,
            batch_size=training.per_device_eval_batch_size,
        )

    use_wandb = config.wandb.enable
    if use_wandb:
        import wandb

        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            group=config.wandb.group,
            job_type=config.wandb.job_type,
            notes=config.wandb.notes,
            tags=config.wandb.tags,
            config=config.model_dump(),
        )

    args = TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=float(training.num_train_epochs),
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        dataloader_num_workers=training.dataloader_num_workers,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        fp16=training.fp16,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        # Validation retrieval runs in RetrievalEvalCallback; the Trainer eval loop is unused.
        eval_strategy="no",
        # Samples are {"eid": ...}; the collator, not the model signature, defines the inputs.
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else ["none"],
        seed=config.seed,
        data_seed=config.seed,
        use_cpu=device == "cpu",
    )
    eval_callback = RetrievalEvalCallback(evaluate, eval_steps=training.eval_steps)
    trainer = RePairTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=GridCollator(
            processor=processor,
            id2image=train_ds.get_image,
            id2text=train_ds.get_text,
            index=train_index,
            pairs_per_direction=config.sampling.pairs_per_direction,
            seed=config.seed,
            log_batch_stats=training.log_batch_stats,
        ),
        loss_fn=RePairLoss(grid_lambda=config.loss.grid_lambda),
        callbacks=[eval_callback],
    )
    eval_callback.trainer = trainer

    logger.info("Validation retrieval before training")
    initial_results = evaluate(model)
    trainer.log(format_retrieval_logs(initial_results))

    trainer.train()
    final_dir = Path(training.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    logger.info("Validation retrieval after training")
    final_results = evaluate(model)
    trainer.log(format_retrieval_logs(final_results))
    if use_wandb:
        import wandb

        wandb.finish()

    return {
        "model_name": config.model.model_name,
        "dataset": config.dataset.key,
        "device": device,
        "seed": config.seed,
        "subsample_train": config.subsample_train,
        "hard_pairs": {"enabled": list(hard_pairs), "paths": hard_pairs},
        "loss": {"grid_lambda": config.loss.grid_lambda},
        "trainer": {"global_step": trainer.state.global_step},
        "eval": {"initial": initial_results, "final": final_results},
    }
