"""Small, local fixtures: no pretrained downloads, dataset downloads or API calls."""

import json
import os
import string
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Set before importing Hugging Face modules; CI must work without credentials.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "repair-test-hf"))


@pytest.fixture
def karpathy_source(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i, split in enumerate(["train"] * 4 + ["val"] * 2 + ["test"] * 2):
        filename = f"image_{i}.png"
        Image.new("RGB", (16, 16), (i * 25, 20, 255 - i * 25)).save(images / filename)
        rows.append({"imgid": i, "filename": filename, "split": split,
                     "sentences": [{"sentid": i * 2 + j, "raw": f"a {color} image"}
                                   for j, color in enumerate(("red", "blue"))]})
    annotations = tmp_path / "dataset.json"
    annotations.write_text(json.dumps({"images": rows}))
    return annotations, images


@pytest.fixture
def local_cache(tmp_path, karpathy_source):
    from scripts.prepare_dataset import prepare_dataset

    annotations, images = karpathy_source
    root = tmp_path / "cache"
    prepare_dataset(annotations, images, root, "flickr30k", shard_size=2)
    return root


@pytest.fixture
def tiny_clip(tmp_path):
    import torch
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel,
                              CLIPProcessor, CLIPTextConfig, CLIPTokenizer,
                              CLIPVisionConfig)

    torch.manual_seed(7)
    torch.set_num_threads(1)
    path = tmp_path / "tiny-clip"
    path.mkdir()
    tokens = ["<|startoftext|>", "<|endoftext|>"]
    tokens += list(string.ascii_lowercase) + [c + "</w>" for c in string.ascii_lowercase]
    (path / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}))
    (path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab_file=str(path / "vocab.json"),
                              merges_file=str(path / "merges.txt"), model_max_length=77)
    processor = CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(size={"shortest_edge": 16},
                                          crop_size={"height": 16, "width": 16}),
    )
    text = CLIPTextConfig(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          max_position_embeddings=77, bos_token_id=0,
                          eos_token_id=1, pad_token_id=1)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2,
                              image_size=16, patch_size=8)
    config = CLIPConfig.from_text_vision_configs(text, vision, projection_dim=8)
    CLIPModel(config).save_pretrained(path)
    processor.save_pretrained(path)
    return path
