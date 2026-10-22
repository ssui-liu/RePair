"""Validation retrieval during training (callback) and its log keys."""

from __future__ import annotations

from typing import Callable, Dict

from transformers import TrainerCallback

_TASK_KEYS = {"Image-to-Text": "i2t", "Text-to-Image": "t2i"}


def format_retrieval_logs(results: Dict) -> Dict[str, float]:
    """{"metrics": {"Image-to-Text": {"R@1": ...}}} -> {"eval_i2t/R@1": ...}.

    The `eval_` prefix makes reporting integrations (e.g. W&B) group these under eval/.
    """
    return {
        f"eval_{_TASK_KEYS.get(task, task)}/{name}": float(value)
        for task, metrics in results.get("metrics", {}).items()
        for name, value in metrics.items()
    }


class RetrievalEvalCallback(TrainerCallback):
    """Run full-ranking validation retrieval every `eval_steps` optimizer steps."""

    def __init__(self, evaluate: Callable[[object], Dict], eval_steps: int) -> None:
        self.evaluate = evaluate
        self.eval_steps = eval_steps
        self.trainer = None  # set after the Trainer is built; its log() reaches all integrations

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step % self.eval_steps == 0:
            self.trainer.log(format_retrieval_logs(self.evaluate(model)))
