from pathlib import Path

import pytest

from config.config_loader import load_config
from scripts.train import parse_args
from src.training.config import build_job_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["repair_i2t", "repair_t2i", "repair_full"])
def test_shipped_training_recipes_build_and_cli_overrides_are_explicit(name):
    raw = load_config(ROOT / f"config/training/{name}.yaml")
    raw["seed"], raw["shuffle_subsample"] = 13, False
    raw["wandb"]["enable"] = True
    args = parse_args(["--dataset-key", "coco30k", "--no-use-wandb", "--no-fp16", "--epochs", "0"])
    config = build_job_config(raw_config=raw, args=args, project_root=ROOT)
    assert config.loss.grid_lambda == .5 and config.sampling.pairs_per_direction == 1
    assert config.dataset.key == "coco30k"
    assert Path(config.dataset.cache_dir) == ROOT / "data_cache"
    assert Path(config.training.output_dir) == ROOT / raw["training"]["output_dir"]
    assert config.training.num_train_epochs == 0  # explicit zero is not dropped
    assert config.seed == 13 and not config.shuffle_subsample
    assert not config.wandb.enable and not config.training.fp16
    assert set(config.hard_pairs.active()) == {"repair_i2t": {"repair-i2t"}, "repair_t2i": {"repair-t2i"},
                                               "repair_full": {"repair-i2t", "repair-t2i"}}[name]


def test_training_config_rejects_unknown_keys_and_types():
    raw = load_config(ROOT / "config/training/repair_i2t.yaml")
    with pytest.raises(ValueError, match="learning_rat"):
        build_job_config(raw_config={**raw, "training": {**raw["training"], "learning_rat": 1e-5}},
                         args=parse_args([]), project_root=ROOT)
    with pytest.raises(ValueError, match="Unknown top-level"):
        build_job_config(raw_config={**raw, "optimizer": {}}, args=parse_args([]), project_root=ROOT)
    with pytest.raises(ValueError, match="Unknown hard-pair types"):
        build_job_config(raw_config={**raw, "hard_pairs": {"enabled": ["repair-x"]}},
                         args=parse_args([]), project_root=ROOT)
    vanilla = build_job_config(raw_config=raw, args=parse_args(["--vanilla"]), project_root=ROOT)
    assert not vanilla.hard_pairs.active()


def test_prepare_i2t_returns_failure_when_input_is_missing(tmp_path):
    from scripts.prepare_i2t import main
    with pytest.raises(RuntimeError, match="Conversion failed"):
        main(["--input-root", str(tmp_path / "missing"), "--output-root", str(tmp_path / "out"),
              "--run-name", "missing-run", "--datasets", "flickr30k"])


def test_prepare_dataset_can_keep_a_released_split(tmp_path, karpathy_source):
    from scripts.prepare_dataset import prepare_dataset

    annotations, images = karpathy_source
    manifest = prepare_dataset(annotations, images, tmp_path / "cache", "coco30k", keep_annotation_splits=True)
    assert manifest["annotation_splits_kept"] and manifest["splits"]["train"]["image_ids"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_dataset(annotations, images, tmp_path / "other", "coco30k", train_size=2,
                        keep_annotation_splits=True)
