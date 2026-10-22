import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from scripts.prepare_dataset import prepare_dataset
from src.data import GridCollator, KarpathyCache, load_anchor_dataset, load_pair_dataset
from src.data import karpathy
from src.data.hard_pairs import decode_t2i_image_id, encode_t2i_image_id
from src.training.config import TrainingJobConfig
from src.training.trainer import train_repair

BOTH = {"repair-i2t": None, "repair-t2i": None}


def write_t2i_parquet(directory, rows):
    directory.mkdir(parents=True)
    records = []
    for query, fp, gt, color in rows:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        records.append({"query_sentid": query, "fp_imgid": fp, "gt_imgid": gt, "edited_image": buffer.getvalue()})
    pq.write_table(pa.Table.from_pylist(records), directory / "part-00000.parquet")


@pytest.fixture
def hard_pair_cache(local_cache):
    # Fixture captions: image i has sentids 2i ("a red image") and 2i+1 ("a blue image").
    root = local_cache / "RePair" / "flickr30k"
    (root / "I2T").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"query_imgid": 1, "fp_sentid": 0, "edited_caption": "a red counterfactual image"}
    ]), root / "I2T" / "synthesis_success.parquet")
    write_t2i_parquet(root / "T2I", [(4, 0, 2, "red"), (4, 0, 2, "blue")])
    return local_cache


def test_prepared_cache_matches_ids_and_never_overwrites(local_cache, karpathy_source):
    dataset = load_pair_dataset(KarpathyCache("flickr30k", str(local_cache)), "train")
    assert len(dataset) == 8
    assert {dataset[i]["image_id"] for i in range(8)} == {0, 1, 2, 3}
    manifest = json.loads((local_cache / "flickr30k" / "manifest.json").read_text())
    assert manifest["splits"]["val"]["image_ids"] == [4, 5]
    annotations, images = karpathy_source
    with pytest.raises(FileExistsError):
        prepare_dataset(annotations, images, local_cache, "flickr30k")


def test_image_subsampling_is_reproducible_and_preserves_eval(tmp_path, karpathy_source):
    annotations, images = karpathy_source
    first = prepare_dataset(annotations, images, tmp_path / "a", "flickr30k", train_size=2)
    second = prepare_dataset(annotations, images, tmp_path / "b", "flickr30k", train_size=2)
    assert first == second
    assert first["splits"]["train"]["captions"] == 4
    assert first["splits"]["val"]["image_ids"] == [4, 5]
    ids = tmp_path / "ids.json"
    ids.write_text("[4]")
    with pytest.raises(ValueError, match="outside the eligible"):
        prepare_dataset(annotations, images, tmp_path / "bad", "flickr30k", train_image_ids=ids)
    assert not (tmp_path / "bad").exists()


def test_metadata_comes_from_annotations_without_decoding_images(tmp_path, karpathy_source, monkeypatch):
    annotations, images = karpathy_source
    source = json.loads(annotations.read_text())
    # Identical captions of one image must keep their own sentids (no text matching).
    source["images"][0]["sentences"][1]["raw"] = source["images"][0]["sentences"][0]["raw"]
    annotations.write_text(json.dumps(source))
    prepare_dataset(annotations, images, tmp_path / "cache", "flickr30k")
    cache = KarpathyCache("flickr30k", str(tmp_path / "cache"))
    split = cache.load_split("train")

    decoded = []
    real_decode = karpathy._decode_image
    monkeypatch.setattr(karpathy, "_decode_image", lambda value: decoded.append(1) or real_decode(value))
    captions, caption_to_image, ids = cache.caption_mapping(split)
    pairs, mapping = cache.pairs_and_mapping(split)
    assert decoded == []
    assert ids["sentid_list"] == list(range(8)) and caption_to_image == [0, 0, 1, 1, 2, 2, 3, 3]
    assert captions[0] == captions[1] and sorted(mapping["sentid_to_info"]) == list(range(8))
    assert [pair[1] for pair in pairs] == list(range(8))
    assert split[1]["image"].size == (16, 16) and decoded == [1]


def test_hard_pair_links_follow_the_paper_grid(hard_pair_cache):
    cache = KarpathyCache("flickr30k", str(hard_pair_cache))
    dataset, index = load_anchor_dataset(cache, "train", BOTH, verbose=False)
    assert len(dataset) == 8 and all(index.get_element(e).image_id >= 0 for e in index.anchor_eids)

    # I2T record: the anchor of the FP caption (sentid 0, image 0) links (query image 1, edit).
    fp_anchor = next(e for e in index.eid2elem.values() if e.caption_id == 0 and e.is_anchor)
    (i2t_eid,) = fp_anchor.i2t_pair_eids
    i2t = index.get_element(i2t_eid)
    assert i2t.image_id == 1 and dataset.get_text(i2t.caption_id) == "a red counterfactual image"

    # T2I records: the first anchor of the FP image (0) links (edit k, query caption 4).
    t2i = [index.get_element(eid) for eid in fp_anchor.t2i_pair_eids]
    assert [decode_t2i_image_id(e.image_id) for e in t2i] == [(4, 0), (4, 1)]
    assert {e.caption_id for e in t2i} == {4} and not any(e.is_anchor for e in t2i)
    assert dataset.get_image(encode_t2i_image_id(4, 0)).getpixel((0, 0)) == (255, 0, 0)
    assert dataset.get_image(encode_t2i_image_id(4, 1)).getpixel((0, 0)) == (0, 0, 255)


def test_records_outside_a_subsampled_split_are_skipped(hard_pair_cache):
    cache = KarpathyCache("flickr30k", str(hard_pair_cache))
    dataset, index = load_anchor_dataset(cache, "train", {"repair-i2t": None}, subsample=1,
                                         shuffle_subsample=False, verbose=False)
    assert len(dataset) == 2  # image 0 only; the I2T query image 1 is outside the subsample
    assert all(not e.i2t_pair_eids for e in index.eid2elem.values())


def test_grids_build_in_dataloader_worker_processes(hard_pair_cache, tiny_clip):
    dataset, index = load_anchor_dataset(KarpathyCache("flickr30k", str(hard_pair_cache)), "train", BOTH,
                                         verbose=False)
    collator = GridCollator(AutoProcessor.from_pretrained(tiny_clip), dataset.get_image,
                            dataset.get_text, index, seed=0)
    batches = list(torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=collator))
    assert sum(len(b["anchor_image_indices"]) for b in batches) == 8
    assert max(int(b["grid_mask"].sum(1).max()) for b in batches) == 3


def test_full_cpu_train_save_reload_and_retrieval(hard_pair_cache, tiny_clip, tmp_path):
    config = TrainingJobConfig.model_validate({
        "model": {"model_name": str(tiny_clip), "device": "cpu"},
        "dataset": {"key": "flickr30k", "cache_dir": str(hard_pair_cache)},
        "training": {"output_dir": str(tmp_path / "run"), "num_train_epochs": 1,
                     "per_device_train_batch_size": 4, "gradient_accumulation_steps": 2,
                     "per_device_eval_batch_size": 4, "logging_steps": 1, "eval_steps": 1,
                     "save_steps": 10, "learning_rate": 1e-3, "eval_metrics": ["recall"], "fp16": False},
        "hard_pairs": {"enabled": ["repair-i2t", "repair-t2i"]},
        "loss": {"grid_lambda": .5},
        "seed": 17, "verbose": False,
    })
    before = CLIPModel.from_pretrained(tiny_clip).text_projection.weight.detach().clone()
    summary = train_repair(config)
    final = tmp_path / "run" / "final"
    reloaded = CLIPModel.from_pretrained(final)
    AutoProcessor.from_pretrained(final)
    assert summary["trainer"]["global_step"] == 1
    assert set(summary["eval"]["final"]["metrics"]) == {"Image-to-Text", "Text-to-Image"}
    assert torch.isfinite(reloaded.text_projection.weight).all()
    assert not torch.equal(before, reloaded.text_projection.weight)
    assert summary["seed"] == 17
    # The public inference path must compute metrics by default for this checkpoint.
    from config.config_loader import load_config
    from scripts.retrieve import parse_args, run_retrieval
    args = parse_args(["--model-name", str(final), "--data-cache", str(hard_pair_cache),
                       "--device", "cpu", "--split", "test", "--batch-size", "4",
                       "--output-dir", str(tmp_path / "eval"), "--no-tqdm"])
    result = run_retrieval(load_config(args.config), args)
    saved = json.loads((tmp_path / "eval/test/evaluation_results.json").read_text())
    assert saved == result and result["metrics"]["Image-to-Text"]["R@1"] >= 0


def test_training_logs_report_both_loss_terms(hard_pair_cache, tiny_clip, tmp_path, monkeypatch):
    from src.training.trainer import RePairTrainer

    logged = []
    original_log = RePairTrainer.log

    def recording_log(self, logs, *args, **kwargs):
        original_log(self, logs, *args, **kwargs)
        logged.append(dict(self.state.log_history[-1]) if self.state.log_history else dict(logs))

    monkeypatch.setattr(RePairTrainer, "log", recording_log)
    train_repair(TrainingJobConfig.model_validate({
        "model": {"model_name": str(tiny_clip), "device": "cpu"},
        "dataset": {"key": "flickr30k", "cache_dir": str(hard_pair_cache)},
        "training": {"output_dir": str(tmp_path / "run"), "num_train_epochs": 1,
                     "per_device_train_batch_size": 8, "logging_steps": 1, "eval_steps": 100,
                     "save_steps": 100, "learning_rate": 1e-3, "eval_metrics": ["recall"]},
        "hard_pairs": {"enabled": ["repair-i2t", "repair-t2i"]},
        "verbose": False,
    }))
    train_logs = [logs for logs in logged if "loss" in logs]
    assert train_logs and {"loss_global", "loss_grid"} <= set(train_logs[0])
    assert any(key.startswith("eval_i2t/") for logs in logged for key in logs)


def test_selected_t2i_import_rejects_split_leaks_and_duplicate_pairs(tmp_path, local_cache):
    from scripts.prepare_t2i import prepare_t2i
    from src.data.hard_pairs import T2IRecords
    annotations = local_cache / "flickr30k/dataset_flickr30k.json"
    image = tmp_path / "selected.png"
    Image.new("RGB", (8, 8), "red").save(image)
    records = tmp_path / "selected.jsonl"
    row = {"query_sentid": 0, "gt_imgid": 0, "fp_imgid": 1, "edited_image_path": image.name}
    records.write_text(json.dumps(row) + "\n")
    out = tmp_path / "imported"
    assert prepare_t2i(records, annotations, out)["records"] == 1
    assert T2IRecords(out).get_image(0).getpixel((0, 0)) == (255, 0, 0)
    records.write_text(json.dumps(dict(row, fp_imgid=4)) + "\n")
    with pytest.raises(ValueError, match="invalid training"):
        prepare_t2i(records, annotations, tmp_path / "leaked")
    assert not (tmp_path / "leaked").exists()
    records.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_t2i(records, annotations, tmp_path / "duplicates")


def test_train_script_writes_run_artifacts(hard_pair_cache, tiny_clip, tmp_path):
    from config.config_loader import load_config
    from scripts.train import parse_args, run_training

    run_dir = tmp_path / "script-run"
    args = parse_args(["--config", "config/training/repair_full.yaml",
                       "--model-name", str(tiny_clip), "--device", "cpu", "--data-cache", str(hard_pair_cache),
                       "--output-dir", str(run_dir), "--epochs", "1", "--batch-size", "8",
                       "--gradient-accumulation-steps", "1", "--dataloader-num-workers", "0",
                       "--no-fp16", "--no-verbose", "--seed", "3"])
    results = run_training(load_config(args.config), args)
    effective = json.loads((run_dir / "effective_config.json").read_text())
    assert effective["seed"] == 3 and effective["training"]["per_device_train_batch_size"] == 8
    assert json.loads((run_dir / "results.json").read_text()) == json.loads(json.dumps(results))
    assert (run_dir / "train.log").exists() and (run_dir / "final" / "config.json").exists()
