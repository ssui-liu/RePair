import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from src.synthesis.api import APIClient
from src.synthesis.i2t.steps import I2TSteps
from src.synthesis.i2t.workflow import I2TWorkflow
from src.synthesis.retry import RetryableError, RetryConfig, is_retryable, with_retry
from src.synthesis.store import ResultStore, SynthesisResult
from src.synthesis.work_items import WorkItemBuilder


def mining_sample():
    return {"query_imgid": 0, "query_image_filename": "0.png",
            "ground_truth": {"best_rank": 1, "best_text": "GT_ONLY_SECRET", "best_sentid": 0, "sentids": [0]},
            "false_positives": {"sentids": [2], "texts": ["a blue image"]}}


class MockAPI:
    def __init__(self):
        self.calls = Counter()
        self.prompts = {}

    def image_text_to_json(self, prompt, image, step_name):
        self.calls[step_name] += 1
        self.prompts[step_name] = prompt
        if step_name == "i2t_quality_check":
            return {"gt_valid": True, "gt_reason": "matches", "fp_too_similar": False, "fp_reason": "wrong color"}
        return {"edit_instruction": "Change blue to red."}

    def text_to_text(self, prompt, step_name):
        self.calls[step_name] += 1
        return f"a red image variant {self.calls[step_name]}"


class MockSelector:
    def select_closest(self, original, candidates):
        assert original == "a blue image" and len(candidates) == 3
        return candidates[1], 0.95


def test_m_candidates_produce_one_pair_and_gt_is_only_used_for_qc():
    items = WorkItemBuilder().build([mining_sample()], sentid_to_text={0: "GT_ONLY_SECRET", 2: "a blue image"})
    assert len(items) == 1 and items[0].item_id == "0_fp0"  # No M x M expansion.
    api = MockAPI()
    workflow = I2TWorkflow(api, {2: "a blue image"}, {0: 0}, num_candidates=3,
                           cosine_selector=MockSelector())
    result = workflow.process(items[0], [{"image": Image.new("RGB", (8, 8))}])
    assert result["status"] == "success" and result["edited_caption"].endswith("2")
    assert result["query_imgid"] == 0 and result["fp_sentid"] == 2
    assert api.calls == {"i2t_quality_check": 1, "i2t_generate_instruction": 1,
                         "i2t_edit_caption": 3}
    assert "GT_ONLY_SECRET" in api.prompts["i2t_quality_check"]
    assert "GT_ONLY_SECRET" not in api.prompts["i2t_generate_instruction"]
    persisted = SynthesisResult.from_dict(result).to_dict()
    assert persisted["num_candidates"] == 3 and persisted["selection_cosine"] == .95


@pytest.mark.parametrize("response", [{}, {"gt_valid": "true", "fp_too_similar": False}])
def test_malformed_qc_is_not_accepted(response):
    class API:
        def image_text_to_json(self, **kwargs):
            return response
    with pytest.raises(ValueError, match="boolean"):
        I2TSteps(API()).quality_check(Image.new("RGB", (2, 2)), "gt", "fp")


def test_qc_api_error_is_not_accepted():
    class API:
        def image_text_to_json(self, **kwargs):
            raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        I2TSteps(API()).quality_check(Image.new("RGB", (2, 2)), "gt", "fp")


def test_retries_happen_once_and_only_for_transient_errors():
    import httpx
    import openai

    request = httpx.Request("POST", "https://example.invalid")
    assert is_retryable(openai.APITimeoutError(request))
    assert is_retryable(openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None))
    assert is_retryable(RetryableError("no choices"))
    assert not is_retryable(openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None))
    assert not is_retryable(ValueError("QC must return boolean"))

    class Reply:
        def __init__(self, text):
            self.text, self.input_tokens, self.output_tokens, self.finish_reason = text, 11, 7, "stop"

    class FlakyClient:
        calls = 0

        def complete(self, prompt, image=None):
            FlakyClient.calls += 1
            if FlakyClient.calls == 1:
                raise RetryableError("no choices")
            return Reply('```json\n{"gt_valid": true}\n```')

    client = APIClient(FlakyClient(), FlakyClient(), RetryConfig(max_retries=2, initial_delay=0))
    assert client.image_text_to_json("prompt", image=b"jpeg") == {"gt_valid": True}
    assert FlakyClient.calls == 2
    assert client.token_tracker.totals() == {"total_input_tokens": 11, "total_output_tokens": 7,
                                             "total_tokens": 18, "total_calls": 1}

    attempts = []

    @with_retry(RetryConfig(max_retries=3, initial_delay=0))
    def permanent_error():
        attempts.append(1)
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        permanent_error()
    assert len(attempts) == 1


def test_store_redacts_secrets_and_names_the_run_by_its_settings(tmp_path):
    store = ResultStore(tmp_path, "i2t", max_queries=1, num_candidates=3, model_name="Gemini 2.5/Flash",
                        config={"api": {"api_key": "test-credential-never-persist"}})
    assert store.run_dir == tmp_path / "i2t" / "gtfirst_m3_qc1_gtqc1_fpqc1_pqc1_gemini-2.5-flash"
    saved = (store.run_dir / "config.json").read_text()
    assert "test-credential-never-persist" not in saved and "[REDACTED]" in saved


def test_qc_rejections_and_failures_are_stored_by_status_and_can_be_rerun(tmp_path):
    class RejectingAPI(MockAPI):
        def image_text_to_json(self, prompt, image, step_name):
            if step_name == "i2t_quality_check":
                return {"gt_valid": True, "fp_too_similar": True, "fp_reason": "already valid"}
            return super().image_text_to_json(prompt, image, step_name)

    api = RejectingAPI()
    item = WorkItemBuilder().build([mining_sample()], {0: "GT_ONLY_SECRET", 2: "a blue image"})[0]
    rejected = I2TWorkflow(api, {}, {0: 0}).process(item, [{"image": Image.new("RGB", (8, 8))}])
    assert rejected["status"] == "qc_rejected" and rejected["qc_status"] == "rejected"
    assert "already valid" in rejected["error"] and "i2t_generate_instruction" not in api.calls
    store = ResultStore(tmp_path, "i2t", 1)
    assert (store.save(rejected) / "analysis.json").exists()
    assert (store.save({"item_id": "1_fp0", "status": "failed", "error": "timeout"}) / "error.json").exists()
    assert store.statistics() == {"total": 2, "success": 0, "qc_rejected": 1, "failed": 1, "success_rate": 0.0}
    rerun = ResultStore(tmp_path, "i2t", 1, resume_processed_statuses=["success", "qc_rejected"])
    assert rerun.is_already_processed("0_fp0") and not rerun.is_already_processed("1_fp0")
    with pytest.raises(ValueError, match="Unknown result status"):
        store.save({"item_id": "2_fp0", "status": "pending"})
    with pytest.raises(ValueError, match="Invalid resume_processed_statuses"):
        ResultStore(tmp_path, "i2t", 1, resume_processed_statuses=["pending"])


def test_dry_run_needs_no_key_or_data_and_creates_no_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "GEMINI_API_KEY"}}
    run = subprocess.run([sys.executable, str(root / "scripts/synthesize_i2t.py"),
                          "--config", str(root / "config/synthesis/i2t.yaml"), "--dry-run"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["synthesis"]["num_candidates"] == 3
    assert list(tmp_path.iterdir()) == []


def _write_fp_pool(results, samples):
    pool = results / "flickr30k/vit-base-patch32/retrieval/train/i2t_fp_pool.json"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"metadata": {"mining_strategy": "top_k_non_gt", "mining_top_k": 100},
                                "samples": samples}))
    return pool


def test_runner_end_to_end_with_resume_and_conversion(tmp_path, local_cache):
    import pyarrow.parquet as pq
    from config.config_loader import load_config
    from scripts.prepare_i2t import convert_dataset
    from scripts.synthesize_i2t import merge_args_with_config, parse_args, run

    results = tmp_path / "results"
    samples = []
    for imgid in range(4):  # fixture captions: image i has sentids 2i ("red") and 2i+1 ("blue")
        fp = [2 * ((imgid + 1) % 4) + 1, 2 * ((imgid + 2) % 4) + 1]
        samples.append({"query_imgid": imgid, "query_image_filename": f"image_{imgid}.png",
                        "ground_truth": {"best_rank": imgid + 1, "best_text": "a red image",
                                         "best_sentid": 2 * imgid, "sentids": [2 * imgid, 2 * imgid + 1]},
                        "false_positives": {"sentids": fp, "texts": ["a blue image"] * 2}})
    _write_fp_pool(results, samples)
    common = ["--config", "config/synthesis/i2t.yaml", "--results-dir", str(results),
              "--data-cache", str(local_cache), "--fps-per-query", "2", "--no-progress"]

    class Selector:
        def select_closest(self, original, candidates):
            return candidates[0], 0.9

    def invoke(*extra):
        args = parse_args(common + list(extra))
        return run(merge_args_with_config(load_config(args.config), args),
                   api_client=MockAPI(), cosine_selector=Selector())

    counts = invoke("--workers", "1")  # serial path
    assert counts["success"] == 8 and counts["resumed"] == 0
    # A second (thread-pool) invocation resumes: nothing is re-processed.
    assert invoke("--workers", "4")["resumed"] == 8

    run_dir = next((results / "flickr30k/vit-base-patch32/synthesis/train/i2t").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["statistics"]["success"] == 8
    metadata = json.loads((run_dir / "success/3_fp1/metadata.json").read_text())
    assert metadata["query_imgid"] == 3 and metadata["fp_sentid"] == 3

    manifest = convert_dataset(input_root=results, output_root=tmp_path / "parquet", dataset="flickr30k",
                               run_name=run_dir.name, batch_size=3)
    rows = pq.read_table(tmp_path / "parquet/flickr30k/I2T/synthesis_success.parquet").to_pylist()
    assert manifest["num_records"] == 8
    assert {(r["query_imgid"], r["fp_sentid"]) for r in rows} == {(s["query_imgid"], fp) for s in samples
                                                                 for fp in s["false_positives"]["sentids"]}


def test_synthesis_log_converts_to_training_parquet_with_verified_fp_id(tmp_path):
    import pyarrow.parquet as pq
    from scripts.prepare_i2t import convert_dataset

    sample = mining_sample()
    results = tmp_path / "results"
    _write_fp_pool(results, [sample])
    item = WorkItemBuilder().build([sample], {0: "GT_ONLY_SECRET", 2: "a blue image"})[0]
    result = I2TWorkflow(MockAPI(), {}, {0: 0}, num_candidates=3,
                         cosine_selector=MockSelector()).process(item, [{"image": Image.new("RGB", (8, 8))}])
    store = ResultStore(results / "flickr30k/vit-base-patch32/synthesis/train", "i2t", 1, num_candidates=3)
    sample_dir = store.save(result)
    options = dict(input_root=results, output_root=tmp_path / "parquet", dataset="flickr30k",
                   run_name=store.run_dir.name, batch_size=2)
    manifest = convert_dataset(**options)
    path = options["output_root"] / "flickr30k/I2T/synthesis_success.parquet"
    row, = pq.read_table(path).to_pylist()
    assert manifest["num_records"] == 1 and row["query_imgid"] == 0 and row["fp_sentid"] == 2
    assert row["edited_caption"].endswith("2") and row["item_id"] == "0_fp0"
    with pytest.raises(FileExistsError):
        convert_dataset(**options)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fp_caption"] = "mismatched FP-pool provenance"
    metadata_path.write_text(json.dumps(metadata))
    options["output_root"] = tmp_path / "bad-parquet"
    with pytest.raises(ValueError, match="No valid synthesis"):
        convert_dataset(**options)
    assert not list(options["output_root"].rglob("*.parquet"))


def test_cosine_selector_returns_the_edit_closest_to_the_fp(tiny_clip):
    import torch
    from src.synthesis.i2t.cosine_selector import CosineSelector

    selector = CosineSelector(model_name=str(tiny_clip), device="cpu")
    candidates = ["a red dog", "a blue cat on a mat", "a dog"]
    best, score = selector.select_closest("a red dog", candidates)
    assert best == "a red dog" and abs(score - 1.0) < 1e-5  # identical text -> cosine 1 (Eq. 3 argmax)
    inputs = selector._processor(text=["a red cat"] + candidates, return_tensors="pt", padding=True)
    with torch.no_grad():
        z = torch.nn.functional.normalize(selector._model.get_text_features(**inputs), dim=-1)
    expected = candidates[int((z[1:] @ z[0]).argmax())]
    assert selector.select_closest("a red cat", candidates)[0] == expected


def test_chat_client_sends_one_image_before_the_prompt(monkeypatch):
    from types import SimpleNamespace
    from src.synthesis.api import ChatClient, ChatConfig

    sent = []

    class Completions:
        def create(self, **kwargs):
            sent.append(kwargs)
            message = SimpleNamespace(content='{"ok": true}')
            usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)

    client = ChatClient(ChatConfig(model="m", api_key="test-key", max_tokens=64))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    response = client.complete("describe", image=Image.new("RGB", (4, 4)))
    content = sent[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and content[1]["text"] == "describe"
    assert sent[0]["max_tokens"] == 64 and (response.input_tokens, response.output_tokens) == (12, 3)
    assert client.client is not None and ChatClient(ChatConfig(api_key="k")).client.max_retries == 0
