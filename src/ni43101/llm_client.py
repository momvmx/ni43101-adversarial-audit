"""Reusable OpenAI-compatible client with structured-output fallback."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

import orjson
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


SchemaT = TypeVar("SchemaT", bound=BaseModel)
ChatMessage = dict[str, str]


class LLMClientError(RuntimeError):
    """Base exception for controlled LLM client failures."""


class LLMAPIError(LLMClientError):
    """Raised after a provider request fails or exhausts API retries."""


class StructuredOutputError(LLMClientError):
    """Raised when a response cannot be validated against the requested schema."""


class StructuredCompletionClient(Protocol):
    """Minimal interface required by extraction agents."""

    def structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[SchemaT],
    ) -> SchemaT: ...


def _is_retryable_api_error(error: BaseException) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


_retry_api_call = retry(
    retry=retry_if_exception(_is_retryable_api_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
    reraise=True,
)


class OpenAICompatibleClient:
    """OpenAI Python Client wrapper for OpenAI-compatible providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")

        self.model = model
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[SchemaT],
    ) -> SchemaT:
        """Return a Pydantic-validated completion.

        Native Pydantic structured output is attempted first. Providers that reject
        native ``response_format`` fall back to JSON-only prompting and local
        ``model_validate_json`` validation.
        """

        if not messages:
            raise ValueError("messages must not be empty")

        copied_messages = [dict(message) for message in messages]
        try:
            completion = self._native_completion(copied_messages, response_model)
        except Exception as error:
            if self._native_format_is_unsupported(error):
                return self._fallback_completion(copied_messages, response_model)
            if isinstance(error, LLMClientError):
                raise
            if isinstance(error, (APIConnectionError, APITimeoutError, APIStatusError)):
                raise LLMAPIError("structured completion API request failed") from error
            if isinstance(error, ValidationError):
                return self._fallback_completion(copied_messages, response_model)
            raise LLMClientError("unexpected structured completion failure") from error

        try:
            message = self._first_message(completion)
            parsed = getattr(message, "parsed", None)
            if isinstance(parsed, response_model):
                return parsed
            if parsed is not None:
                try:
                    return response_model.model_validate(parsed)
                except ValidationError as error:
                    raise StructuredOutputError(
                        f"response failed {response_model.__name__} validation"
                    ) from error

            return self._validate_content(
                getattr(message, "content", None),
                response_model,
            )
        except StructuredOutputError:
            # Some OpenAI-compatible providers accept a Pydantic response_format
            # but do not actually honor the supplied schema. Retry once with the
            # explicit JSON-only prompt and local Pydantic validation.
            return self._fallback_completion(copied_messages, response_model)

    @_retry_api_call
    def _native_completion(
        self,
        messages: list[ChatMessage],
        response_model: type[SchemaT],
    ) -> Any:
        return self._client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
            temperature=0,
        )

    @_retry_api_call
    def _json_mode_completion(self, messages: list[ChatMessage]) -> Any:
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )

    @_retry_api_call
    def _prompt_only_json_completion(self, messages: list[ChatMessage]) -> Any:
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

    def _fallback_completion(
        self,
        messages: list[ChatMessage],
        response_model: type[SchemaT],
    ) -> SchemaT:
        fallback_messages = [
            {
                "role": "system",
                "content": self._json_only_instruction(response_model),
            },
            *messages,
        ]
        try:
            completion = self._json_mode_completion(fallback_messages)
        except Exception as error:
            if not self._native_format_is_unsupported(error):
                if isinstance(
                    error,
                    (APIConnectionError, APITimeoutError, APIStatusError),
                ):
                    raise LLMAPIError("JSON fallback API request failed") from error
                raise LLMClientError("unexpected JSON fallback failure") from error

            # Some OpenAI-compatible providers reject JSON mode entirely. Keep
            # the prompt-only path as the final compatibility fallback.
            try:
                completion = self._prompt_only_json_completion(fallback_messages)
            except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                raise LLMAPIError("JSON fallback API request failed") from error
            except Exception as error:
                raise LLMClientError("unexpected JSON fallback failure") from error

        message = self._first_message(completion)
        return self._validate_content(
            getattr(message, "content", None),
            response_model,
        )

    @staticmethod
    def _native_format_is_unsupported(error: BaseException) -> bool:
        if isinstance(error, (AttributeError, NotImplementedError, TypeError)):
            return True
        return isinstance(error, APIStatusError) and error.status_code in {
            400,
            404,
            415,
            422,
        }

    @staticmethod
    def _first_message(completion: Any) -> Any:
        try:
            return completion.choices[0].message
        except (AttributeError, IndexError, TypeError) as error:
            raise StructuredOutputError(
                "provider returned no assistant message"
            ) from error

    @staticmethod
    def _validate_content(
        content: Any,
        response_model: type[SchemaT],
    ) -> SchemaT:
        if not isinstance(content, str) or not content.strip():
            raise StructuredOutputError("provider returned empty structured content")
        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise StructuredOutputError(
                f"response is not valid {response_model.__name__} JSON"
            ) from error

    @staticmethod
    def _json_only_instruction(response_model: type[BaseModel]) -> str:
        schema = orjson.dumps(response_model.model_json_schema()).decode("utf-8")
        return (
            "Return exactly one valid JSON object and no Markdown, code fences, "
            "commentary, or surrounding text. The JSON must satisfy this schema: "
            f"{schema}"
        )


class FakeLLMClient:
    """Deterministic structured client for tests; it never performs API calls."""

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[SchemaT],
    ) -> SchemaT:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "response_model": response_model,
            }
        )
        if not self._responses:
            raise LLMClientError("FakeLLMClient has no queued response")

        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, response_model):
            return response

        try:
            if isinstance(response, str):
                return response_model.model_validate_json(response)
            return response_model.model_validate(response)
        except ValidationError as error:
            raise StructuredOutputError(
                f"fake response is not valid {response_model.__name__} data"
            ) from error
