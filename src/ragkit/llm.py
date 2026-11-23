from typing import Protocol

import litellm

# Different providers reject different sampling params; let LiteLLM drop unsupported ones.
litellm.drop_params = True


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class LiteLLMClient:
    """Provider-agnostic chat completion through LiteLLM."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retries = retries

    def complete(self, system: str, user: str) -> str:
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
