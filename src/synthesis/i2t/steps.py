"""
I2T synthesis steps as model calls: joint QC, edit instruction G(q, p_fp) and
caption editing E(p_fp, tau_edit).

Every call either returns a validated value or raises; a failed or malformed QC
reply is never treated as acceptance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Tuple

from .prompts import I2TPrompts

logger = logging.getLogger(__name__)

_STRICT_JSON = ('\n\nIMPORTANT: You MUST return ONLY valid JSON with a single key "edit_instruction". '
                "Do NOT include any extra text, markdown, or commentary. Ensure all quotes are closed.")
_STRICT_CAPTION = ("\n\nIMPORTANT: Output must be a single non-empty English sentence caption. "
                   "Do NOT output just punctuation (e.g., '.' or '...'). Only output the corrected caption.")


def _clean_caption(text: str, max_chars: int) -> str:
    """First non-empty line, unquoted and truncated."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    out = lines[0] if lines else ""
    if len(out) >= 2 and out[0] == out[-1] == '"':
        out = out[1:-1].strip()
    return out[:max_chars].rstrip()


def _is_valid_caption(text: str) -> bool:
    return len(text.strip()) >= 2 and re.fullmatch(r"[\W_]+", text.strip()) is None


class I2TSteps:
    def __init__(self, api_client: Any):
        """`api_client` provides `image_text_to_json` and `text_to_text` (see `APIClient`)."""
        self.api_client = api_client
        self.prompts = I2TPrompts()

    def quality_check(self, query_image, gt_caption: str, fp_caption: str) -> Tuple[bool, bool, str]:
        """One call evaluating both criteria.

        Returns (gt_valid, fp_is_benign, reason): criterion 1 (the GT caption matches the
        image) and criterion 2 (the FP caption is already a valid description).
        """
        result = self.api_client.image_text_to_json(
            prompt=self.prompts.build_qc_prompt(gt_caption, fp_caption),
            image=query_image,
            step_name="i2t_quality_check",
        )
        gt_valid, fp_is_benign = result.get("gt_valid"), result.get("fp_too_similar")
        if type(gt_valid) is not bool or type(fp_is_benign) is not bool:
            raise ValueError("QC must return boolean gt_valid and fp_too_similar")
        reason = "; ".join(str(result[k]) for k in ("gt_reason", "fp_reason") if result.get(k))
        return gt_valid, fp_is_benign, reason

    def edit_instruction(self, query_image, fp_caption: str) -> str:
        """G(q, p_fp): the GT caption is deliberately not part of this prompt."""
        prompt = self.prompts.build_edit_instruction_prompt(fp_caption)

        def ask(text: str, step_name: str) -> str:
            value = self.api_client.image_text_to_json(prompt=text, image=query_image, step_name=step_name)
            value = value.get("edit_instruction") or ""
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value).strip()

        try:
            instruction = ask(prompt, "i2t_generate_instruction")
        except json.JSONDecodeError as e:
            logger.warning("[I2T] Instruction JSON parse error (%s); retrying with a stricter prompt", e)
            instruction = ""
        except Exception as e:
            raise RuntimeError(f"I2T edit_instruction failed: {e}") from e
        if not instruction:
            try:
                instruction = ask(prompt + _STRICT_JSON, "i2t_generate_instruction_retry")
            except Exception as e:
                raise RuntimeError(f"I2T edit_instruction failed after retry: {e}") from e
        if not instruction:
            # No generic fallback: it tends to trigger full rewrites instead of minimal edits.
            raise RuntimeError("I2T edit_instruction was empty after retry")
        return instruction

    def edit_caption(self, fp_caption: str, instruction: str, max_output_chars: int = 256) -> str:
        """E(p_fp, tau_edit): one sampled caption edit."""
        prompt = self.prompts.build_caption_edit_prompt(original_caption=fp_caption, edit_instruction=instruction)
        caption = _clean_caption(self.api_client.text_to_text(prompt=prompt, step_name="i2t_edit_caption"),
                                 max_output_chars)
        if _is_valid_caption(caption):
            return caption
        caption = _clean_caption(
            self.api_client.text_to_text(prompt=prompt + _STRICT_CAPTION, step_name="i2t_edit_caption_retry"),
            max_output_chars,
        )
        if not _is_valid_caption(caption):
            raise RuntimeError(f"I2T edit_caption produced invalid output after retry: {caption!r}")
        return caption
