"""
Cosine-similarity-based post-edit selection (paper Eq. 3).

Given the original false-positive text p^fp and M edited candidates,
selects the candidate closest to p^fp in CLIP embedding space:

    p^edit = argmax_m cos(z(p^fp), z(p^edit_m))

This enforces the minimal-change principle: the selected edit preserves
the shared scaffold while correcting only the failure-causing residual.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CosineSelector:
    """Select the edited candidate closest to the original FP in CLIP text-embedding space."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import CLIPModel, CLIPProcessor

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = CLIPModel.from_pretrained(self._model_name).to(device).eval()
            processor = CLIPProcessor.from_pretrained(self._model_name)
            self._processor, self._device, self._model = processor, device, model
            logger.info(f"CosineSelector: loaded {self._model_name} on {device}")

    @torch.no_grad()
    def select_closest(self, fp_text: str, candidates: List[str]) -> Tuple[str, float]:
        """Return (best_candidate, cosine_score) closest to fp_text in embedding space."""
        if len(candidates) == 1:
            return candidates[0], 1.0

        self._ensure_loaded()

        all_texts = [fp_text] + candidates
        inputs = self._processor(text=all_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        embeds = self._model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        fp_embed = embeds[0:1]
        cand_embeds = embeds[1:]
        similarities = (cand_embeds @ fp_embed.T).squeeze(-1)

        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        logger.debug(
            f"CosineSelector: scores={[f'{s:.4f}' for s in similarities.tolist()]}, "
            f"selected idx={best_idx} (score={best_score:.4f})"
        )
        return candidates[best_idx], best_score
