"""Batched, L2-normalized CLIP image/text features."""

import logging
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """Encode a split's images and captions with a CLIP model.

    Pass `model`/`processor` to reuse in-memory objects (e.g. during training);
    otherwise both are loaded from `model_name` (`processor_name` overrides the
    processor source, for checkpoints saved without one).
    """

    def __init__(
        self,
        model_name: Optional[str] = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
        model: Optional[torch.nn.Module] = None,
        processor=None,
        processor_name: Optional[str] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            logger.info("Loading CLIP model %s on %s", model_name, self.device)
            model = AutoModel.from_pretrained(model_name)
        self.model = model.to(self.device)
        if processor is None:
            processor = AutoProcessor.from_pretrained(processor_name or model_name)
        self.processor = processor
        self.model.eval()

    @torch.no_grad()
    def extract_image_features(self, images: List) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        features = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device))
        return features / features.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def encode_images(self, dataset) -> torch.Tensor:
        """Features [N_images, D] of `dataset[i]["image"]` for every row."""
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        batches = []
        for start in tqdm(range(0, len(dataset), self.batch_size), desc="Encoding images", leave=True):
            images = [dataset[j]["image"] for j in range(start, min(start + self.batch_size, len(dataset)))]
            batches.append(self.extract_image_features(images).cpu())
        return torch.cat(batches)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Features [N_texts, D] of `texts`."""
        if not texts:
            raise ValueError("Text list is empty")
        batches = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding texts", leave=True):
            batches.append(self.extract_text_features(texts[start:start + self.batch_size]).cpu())
        return torch.cat(batches)
