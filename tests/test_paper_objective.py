import tempfile

import pytest
import torch
import torch.nn.functional as F
from transformers import TrainingArguments

from src.data.collator import GridCollator
from src.data.graph import Element, ElementIndex
from src.training.losses import grid_loss
from src.training.trainer import RePairTrainer


def per_grid_reference(images, texts, image_rows, text_rows, scale):
    """Eq. 5 written grid by grid: mean symmetric diagonal CE, summed and divided by B."""
    total = images.sum() * 0
    for image_ids, text_ids in zip(image_rows, text_rows, strict=True):
        if len(image_ids) > 1:
            scores = scale * images[image_ids] @ texts[text_ids].T
            target = torch.arange(len(image_ids))
            total = total + 0.5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target))
    return total / len(image_rows)


def padded(rows):
    width = max(map(len, rows))
    ids = torch.tensor([row + [-1] * (width - len(row)) for row in rows])
    return ids, ids >= 0


def test_grid_matches_symmetric_diagonal_ce_and_counts_zero_grids():
    images = torch.tensor([[1., 0.], [0., 1.], [.6, .8]], requires_grad=True)
    texts = torch.tensor([[.8, .2], [.1, .9], [.7, .3]], requires_grad=True)
    ids, mask = padded([[0, 1, 2], [1]])
    scale = torch.tensor(2., requires_grad=True)
    actual = grid_loss(images, texts, ids, ids, mask, scale)
    scores = scale * images @ texts.T
    # One 3x3 grid and one 1x1 grid. The latter contributes zero but stays in B=2.
    expected = (F.cross_entropy(scores, torch.arange(3)) + F.cross_entropy(scores.T, torch.arange(3))) / 4
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(images.grad).all() and torch.isfinite(texts.grad).all()
    assert scale.grad.abs() > 0


def test_batched_grid_loss_matches_per_grid_reference_with_gradients():
    generator = torch.Generator().manual_seed(3)
    image_rows = [[0, 4, 7], [1], [2, 5], [3, 6, 8], [0, 8]]
    text_rows = [[2, 6, 9], [0], [1, 5], [3, 7, 4], [8, 2]]
    inputs = [F.normalize(torch.randn(n, 6, generator=generator), dim=-1).requires_grad_() for n in (9, 10)]
    scale = torch.tensor(3.5, requires_grad=True)

    expected = per_grid_reference(*inputs, image_rows, text_rows, scale)
    expected_grads = torch.autograd.grad(expected, [*inputs, scale])
    image_ids, mask = padded(image_rows)
    text_ids, _ = padded(text_rows)
    actual = grid_loss(*inputs, image_ids, text_ids, mask, scale)
    actual_grads = torch.autograd.grad(actual, [*inputs, scale])

    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)


def test_grid_rejects_unmatched_image_text_rows():
    embeds = torch.eye(2)
    with pytest.raises(ValueError, match="matched pairs"):
        grid_loss(embeds, embeds, torch.tensor([[0, 1]]), torch.tensor([[0, -1]]),
                  torch.tensor([[True, True]]), torch.tensor(1.))


def test_collator_keeps_original_anchor_and_paired_diagonal():
    elements = [
        Element(0, 10, 100, i2t_pair_eids=[1], t2i_pair_eids=[2]),
        Element(1, 20, 200, is_anchor=False),
        # Shares the anchor caption: dropping only the text would shift the diagonal.
        Element(2, 30, 100, is_anchor=False),
    ]
    collator = GridCollator(None, None, None, ElementIndex(elements), seed=3)
    assert collator._build_grid(0) == ([10, 20], [100, 200])


class _ToyLoss:
    def __call__(self, model, inputs):
        return (model.weight * inputs["x"]).mean()

    def consume_logs(self):
        return {}


def test_gradient_accumulation_averages_microbatches():
    # d loss / d w = 1 for every microbatch: averaging gives w = -lr after one update.
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2,
                             gradient_accumulation_steps=4, max_steps=1, learning_rate=1.0, optim="sgd",
                             lr_scheduler_type="constant", weight_decay=0.0, max_grad_norm=1e9,
                             report_to=[], use_cpu=True, save_strategy="no", remove_unused_columns=False)
    trainer = RePairTrainer(model=model, args=args, loss_fn=_ToyLoss(),
                                   train_dataset=[{"x": torch.tensor(1.0)}] * 8,
                                   data_collator=lambda rows: {"x": torch.stack([r["x"] for r in rows])})
    trainer.train()
    torch.testing.assert_close(model.weight.detach(), torch.tensor([[-1.0]]))
