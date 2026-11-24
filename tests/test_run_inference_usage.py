"""run_inference reports each reply's usage through ``on_inference_usage``."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types.beta import BetaUsage
from google.genai.types import GenerateContentResponseUsageMetadata
from openai.types import CompletionUsage

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

CONTEXT = LLMContext(messages=[{"role": "user", "content": "hi"}])


async def _reported(service, run) -> list:
    reports = []
    service.add_event_handler("on_inference_usage", lambda _service, data: reports.append(data))
    await run()
    assert [(data.processor, data.model) for data in reports] == [
        (service.name, service._settings.model)
    ]
    return [data.value for data in reports]


@pytest.mark.asyncio
async def test_openai_compatible_reports_gross_input_with_cached_tokens():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4.1"))
    service._client = AsyncMock()
    service._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details={"cached_tokens": 40},
        ),
    )
    [usage] = await _reported(service, lambda: service.run_inference(CONTEXT))
    assert (usage.prompt_tokens, usage.cache_read_input_tokens) == (100, 40)
    assert (usage.completion_tokens, usage.total_tokens) == (20, 120)


@pytest.mark.asyncio
async def test_google_counts_thinking_as_output_and_cached_content():
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.5-flash")
    )
    response = SimpleNamespace(
        candidates=[],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            cached_content_token_count=40,
            candidates_token_count=15,
            thoughts_token_count=5,
        ),
    )
    with patch.object(service._client.aio.models, "generate_content", return_value=response):
        [usage] = await _reported(service, lambda: service.run_inference(CONTEXT))
    assert usage == LLMTokenUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cache_read_input_tokens=40,
        reasoning_tokens=5,
    )


@pytest.mark.asyncio
async def test_anthropic_reports_net_input_with_cache_in_total():
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-4-5")
    )
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        stop_reason="end_turn",
        usage=BetaUsage(
            input_tokens=60,
            output_tokens=20,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=10,
        ),
    )
    [usage] = await _reported(service, lambda: service.run_inference(CONTEXT))
    assert (usage.prompt_tokens, usage.completion_tokens) == (60, 20)
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (40, 10)
    assert usage.total_tokens == 130


@pytest.mark.asyncio
async def test_bedrock_reports_converse_usage():
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-sonnet-4-5")
    )
    client = AsyncMock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {
            "inputTokens": 60,
            "outputTokens": 20,
            "totalTokens": 120,
            "cacheReadInputTokens": 40,
        },
    }
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=client)
    session.__aexit__ = AsyncMock(return_value=None)
    with patch.object(service._aws_session, "create_client", return_value=session):
        [usage] = await _reported(service, lambda: service.run_inference(CONTEXT))
    assert (usage.prompt_tokens, usage.completion_tokens) == (60, 20)
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (40, 0)
    assert usage.total_tokens == 120
