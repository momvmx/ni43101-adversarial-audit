from types import SimpleNamespace

import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from ni43101.llm_client import (
    LLMAPIError,
    OpenAICompatibleClient,
    StructuredOutputError,
)


class _Answer(BaseModel):
    value: int


class _FallbackCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.created_messages: list[dict[str, str]] | None = None
        self.created_response_format: object | None = None

    def parse(self, **_: object) -> object:
        raise TypeError("provider does not support native response_format")

    def create(self, **kwargs: object) -> object:
        self.created_messages = kwargs["messages"]  # type: ignore[assignment]
        self.created_response_format = kwargs.get("response_format")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ]
        )


class _FailingCompletions:
    def __init__(self) -> None:
        self.attempts = 0

    def parse(self, **_: object) -> object:
        self.attempts += 1
        raise APIConnectionError(request=object())  # type: ignore[arg-type]


class _InvalidNativeCompletions(_FallbackCompletions):
    def parse(self, **_: object) -> object:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"wrong": 1}'))
            ]
        )


class _RejectJsonModeCompletions(_FallbackCompletions):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.create_attempts = 0

    def create(self, **kwargs: object) -> object:
        self.create_attempts += 1
        if kwargs.get("response_format") is not None:
            raise TypeError("provider does not support JSON mode")
        return super().create(**kwargs)


def _client(content: str) -> tuple[OpenAICompatibleClient, _FallbackCompletions]:
    completions = _FallbackCompletions(content)
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.invalid",
        model="test-model",
        client=transport,
    )
    return client, completions


def test_native_format_falls_back_to_json_and_local_validation() -> None:
    client, completions = _client('{"value": 7}')

    result = client.structured_completion(
        [{"role": "user", "content": "return seven"}],
        _Answer,
    )

    assert result.value == 7
    assert completions.created_messages is not None
    assert completions.created_response_format == {"type": "json_object"}
    assert "no Markdown" in completions.created_messages[0]["content"]
    assert '"value"' in completions.created_messages[0]["content"]


def test_invalid_fallback_json_raises_controlled_error() -> None:
    client, _ = _client("not valid json")

    with pytest.raises(StructuredOutputError, match="not valid _Answer JSON"):
        client.structured_completion(
            [{"role": "user", "content": "return seven"}],
            _Answer,
        )


def test_invalid_native_schema_retries_with_json_only_fallback() -> None:
    completions = _InvalidNativeCompletions('{"value": 11}')
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.invalid",
        model="test-model",
        client=transport,
    )

    result = client.structured_completion(
        [{"role": "user", "content": "return eleven"}],
        _Answer,
    )

    assert result.value == 11
    assert completions.created_messages is not None
    assert completions.created_response_format == {"type": "json_object"}
    assert "no Markdown" in completions.created_messages[0]["content"]


def test_json_mode_rejection_falls_back_to_prompt_only_json() -> None:
    completions = _RejectJsonModeCompletions('{"value": 13}')
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.invalid",
        model="test-model",
        client=transport,
    )

    result = client.structured_completion(
        [{"role": "user", "content": "return thirteen"}],
        _Answer,
    )

    assert result.value == 13
    assert completions.create_attempts == 2
    assert completions.created_response_format is None


def test_api_retry_stops_after_three_attempts() -> None:
    completions = _FailingCompletions()
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://provider.invalid",
        model="test-model",
        client=transport,
    )

    with pytest.raises(LLMAPIError):
        client.structured_completion(
            [{"role": "user", "content": "return seven"}],
            _Answer,
        )

    assert completions.attempts == 3
