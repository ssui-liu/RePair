"""Retry with exponential backoff for transient API failures (the pipeline's only retry layer)."""

from __future__ import annotations

import functools
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_STATUS = {408, 409, 429}
_TRANSIENT_PATTERNS = (
    "rate limit", "quota", "too many requests", "timeout", "timed out", "connection",
    "unavailable", "resource exhausted", "temporarily", "overloaded",
)


class RetryableError(Exception):
    """Raise to request a retry (e.g. an empty or malformed response)."""


@dataclass
class RetryConfig:
    """`max_retries` extra attempts; delay = min(initial_delay * backoff_factor**attempt, max_delay)."""

    max_retries: int = 3
    initial_delay: float = 2.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0


def is_retryable(exception: Exception) -> bool:
    """Transient: connection/timeout errors, HTTP 408/409/429/5xx, malformed JSON, RetryableError."""
    if isinstance(exception, (RetryableError, json.JSONDecodeError)):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover
        openai = None
    if openai is not None:
        if isinstance(exception, openai.APIConnectionError):  # includes APITimeoutError
            return True
        if isinstance(exception, openai.APIStatusError):
            return exception.status_code in _TRANSIENT_STATUS or exception.status_code >= 500
    message = str(exception).lower()
    return any(pattern in message for pattern in _TRANSIENT_PATTERNS)


def with_retry(config: RetryConfig) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator retrying transient failures; other exceptions propagate immediately."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    if attempt >= config.max_retries or not is_retryable(error):
                        if attempt:
                            logger.error("[Retry] Giving up after %d attempts: %s", attempt + 1, error)
                        raise
                    delay = min(config.initial_delay * config.backoff_factor ** attempt, config.max_delay)
                    logger.warning("[Retry] Attempt %d/%d failed: %s. Retrying in %.1fs",
                                   attempt + 1, config.max_retries + 1, error, delay)
                    time.sleep(delay)
            raise AssertionError("unreachable")

        return wrapper

    return decorator
