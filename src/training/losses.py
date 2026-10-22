"""
RePair objective (paper Eq. 6): L = L_global + lambda * L_grid.

- `global_infonce_loss`: symmetric InfoNCE over the original anchor pairs (Appendix Eq. 8).
- `grid_loss`: symmetric diagonal CE inside every local grid, averaged over the B anchors (Eq. 5).
"""

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-6))


def global_infonce_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                        logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over matched (image_i, text_i) anchor pairs."""
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    targets = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t = nn.functional.cross_entropy(logits_per_image.t(), targets)
    return 0.5 * (loss_i + loss_t)


def grid_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    grid_image_indices: torch.Tensor,
    grid_text_indices: torch.Tensor,
    grid_mask: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Local grid loss averaged over all B anchors of the batch (paper Eq. 5).

    Row i of grid b pairs image `grid_image_indices[b, i]` with text
    `grid_text_indices[b, i]`; padded rows have `grid_mask` False. With
    S_b = logit_scale * Z_I Z_T^T over the n_b valid rows,

        L_grid = 1/B * sum_b 1/(2 n_b) * sum_i [ -log softmax_j(S_b)[i, i]
                                                 -log softmax_k(S_b)[i, i] ]

    A 1x1 grid contributes exactly zero but still counts in B.
    """
    if grid_mask.size(0) == 0:
        raise ValueError("Grid loss requires a non-empty anchor batch.")
    valid_ids = (grid_image_indices >= 0) & (grid_text_indices >= 0)
    if bool((grid_mask & ~valid_ids).any()):
        raise ValueError("Grid image/text indices must contain matched pairs.")

    images = image_embeds[grid_image_indices.clamp(min=0)]          # [B, G, D]
    texts = text_embeds[grid_text_indices.clamp(min=0)]             # [B, G, D]
    scores = logit_scale * images @ texts.transpose(1, 2)           # [B, G, G]
    scores = scores.to(torch.promote_types(scores.dtype, torch.float32))  # fp16 -> fp32, as cross_entropy
    # Padding is filled with a finite minimum (not -inf) so all-padding rows stay NaN-free.
    pair_mask = grid_mask[:, :, None] & grid_mask[:, None, :]
    scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
    image_to_text = scores.log_softmax(dim=2).diagonal(dim1=1, dim2=2)
    text_to_image = scores.log_softmax(dim=1).diagonal(dim1=1, dim2=2)
    rows = grid_mask.float()
    per_grid = -((image_to_text + text_to_image) * rows).sum(dim=1) / (2 * rows.sum(dim=1).clamp(min=1))
    return per_grid.sum() / grid_mask.size(0)


@dataclass
class RePairLoss:
    """L_global over the anchors + `grid_lambda` * L_grid, from a `GridCollator` batch."""

    grid_lambda: float = 0.5
    # Running sums of the two terms since the last `consume_logs` call.
    _sums: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _count: int = field(default=0, repr=False)

    def __call__(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Each distinct image/text of the batch is encoded once.
        image_embeds = l2_normalize(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_embeds = l2_normalize(
            model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        )
        logit_scale = model.logit_scale.exp()
        loss_global = global_infonce_loss(
            image_embeds.index_select(0, inputs["anchor_image_indices"]),
            text_embeds.index_select(0, inputs["anchor_text_indices"]),
            logit_scale,
        )
        loss_grid = grid_loss(image_embeds, text_embeds, inputs["grid_image_indices"],
                              inputs["grid_text_indices"], inputs["grid_mask"], logit_scale)
        if model.training:
            for name, value in (("loss_global", loss_global), ("loss_grid", loss_grid)):
                self._sums[name] = self._sums.get(name, 0) + value.detach().float()
            self._count += 1
        return loss_global + self.grid_lambda * loss_grid

    def consume_logs(self) -> Dict[str, float]:
        """Mean of each loss term since the previous call (empty if none)."""
        if not self._count:
            return {}
        logs = {name: round(float(total) / self._count, 4) for name, total in self._sums.items()}
        self._sums, self._count = {}, 0
        return logs
