"""
Model access for synthesis.

- `ChatClient`: one chat-completions request to an OpenAI-compatible endpoint (the
  paper recipe uses Gemini 2.5 Flash through Google's OpenAI-compatible API), text
  only or one image plus text. The SDK's own retries are disabled.
- `APIClient`: the calls used by the pipeline (text -> text, image + text -> JSON)
  behind the single retry layer of `retry.py`, with token accounting.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .retry import RetryableError, RetryConfig, with_retry

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, bytes]


@dataclass
class ChatConfig:
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    timeout_seconds: float = 120.0
    max_tokens: Optional[int] = None


@dataclass
class ChatResponse:
    text: str
    input_tokens: Optional[int] = None   # server-reported usage, when available
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


def _image_part(image: ImageInput) -> Dict[str, Any]:
    if isinstance(image, (str, Path)):
        data, suffix = Path(image).read_bytes(), Path(image).suffix.lower().lstrip(".")
        media_type = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        data, media_type = buffer.getvalue(), "jpeg"
    elif isinstance(image, bytes):
        data, media_type = image, "jpeg"
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    url = f"data:image/{media_type};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


class ChatClient:
    def __init__(self, cfg: ChatConfig):
        from openai import OpenAI

        if not cfg.api_key:
            raise ValueError("API key not provided (set GEMINI_API_KEY or OPENAI_API_KEY).")
        self.cfg = cfg
        kwargs = {"api_key": cfg.api_key, "timeout": float(cfg.timeout_seconds), "max_retries": 0}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self.client = OpenAI(**kwargs)

    def complete(self, prompt: str, image: Optional[ImageInput] = None) -> ChatResponse:
        content: Union[str, List[Dict[str, Any]]] = prompt
        if image is not None:
            content = [_image_part(image), {"type": "text", "text": prompt}]
        kwargs: Dict[str, Any] = {}
        if self.cfg.max_tokens:
            kwargs["max_tokens"] = int(self.cfg.max_tokens)
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": content}],
            temperature=float(self.cfg.temperature),
            **kwargs,
        )
        if not getattr(resp, "choices", None):
            raise RetryableError("Endpoint returned no choices")
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
        )


class TokenTracker:
    """Thread-safe token/call counters of a synthesis run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def record_call(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            self.calls += 1

    def totals(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_input_tokens": self.input_tokens,
                "total_output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "total_calls": self.calls,
            }

    def progress_string(self) -> str:
        t = self.totals()
        return f"Tokens: {t['total_tokens']:,} (in: {t['total_input_tokens']:,}, out: {t['total_output_tokens']:,})"


def _estimated_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse the JSON object in a model reply (tolerates code fences and surrounding prose)."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


class APIClient:
    """Pipeline calls on a text client and a vision client (`ChatClient` instances)."""

    def __init__(self, text_client: Any, vision_client: Any, retry_config: Optional[RetryConfig] = None,
                 token_tracker: Optional[TokenTracker] = None):
        self.text_client = text_client
        self.vision_client = vision_client
        self.retry_config = retry_config or RetryConfig()
        self.token_tracker = token_tracker or TokenTracker()

    def _record(self, prompt: str, response: ChatResponse, image: bool) -> None:
        # Server-reported usage when available; otherwise a rough estimate (~1.3 tokens/word, 500/image).
        input_tokens = response.input_tokens
        if input_tokens is None:
            input_tokens = _estimated_tokens(prompt) + (500 if image else 0)
        output_tokens = response.output_tokens
        if output_tokens is None:
            output_tokens = _estimated_tokens(response.text)
        self.token_tracker.record_call(input_tokens, output_tokens)

    def text_to_text(self, prompt: str, step_name: str = "text_to_text") -> str:
        @with_retry(self.retry_config)
        def call():
            return self.text_client.complete(prompt)

        response = call()
        self._record(prompt, response, image=False)
        return response.text

    def image_text_to_json(self, prompt: str, image, step_name: str = "image_text_to_json") -> Dict[str, Any]:
        @with_retry(self.retry_config)
        def call():
            response = self.vision_client.complete(prompt, image=image)
            self._record(prompt, response, image=True)
            try:
                return parse_json_response(response.text)
            except json.JSONDecodeError as error:
                logger.warning("[%s] JSON parse error: %s | finish_reason=%r | head=%r",
                               step_name, error, response.finish_reason, response.text[:300])
                raise

        return call()
