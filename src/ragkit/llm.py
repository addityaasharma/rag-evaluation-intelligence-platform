import json
import re
import threading
import time
from functools import lru_cache
from typing import Any, Protocol

import litellm

# Different providers reject different sampling params; let LiteLLM drop unsupported ones.
litellm.drop_params = True


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class RateLimiter:
    """Spaces calls evenly so a free-tier requests-per-minute quota is never exceeded."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / requests_per_minute
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        time.sleep(max(0.0, slot - now))


@lru_cache
def shared_limiter(requests_per_minute: int) -> RateLimiter:
    """One limiter per rate, shared by all clients because providers meter per API key."""
    return RateLimiter(requests_per_minute)


class LiteLLMClient:
    """Provider-agnostic chat completion through LiteLLM."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 3,
        requests_per_minute: int | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retries = retries
        self._limiter = shared_limiter(requests_per_minute) if requests_per_minute else None

    def complete(self, system: str, user: str) -> str:
        if self._limiter:
            self._limiter.wait()
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            num_retries=self._retries,
        )
        return (response.choices[0].message.content or "").strip()


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object in model output, tolerating prose or code fences."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise ValueError(f"No JSON object in model output: {raw!r}")
    data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data
