"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import typing

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import DEFAULT_REASONING, DEFAULT_VERBOSITY, BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a structured completion using OpenAI's beta parse API."""
        # Reasoning models don't support temperature
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
            or ('reasoning' in model and 'non-reasoning' not in model)
        )

        # xAI/Grok models don't support the OpenAI Responses API —
        # fall back to chat completions with JSON mode
        is_xai_model = 'grok' in model

        if is_xai_model:
            return await self._create_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
            )

        request_kwargs = {
            'model': model,
            'input': messages,  # type: ignore
            'max_output_tokens': max_tokens,
            'text_format': response_model,  # type: ignore
        }

        temperature_value = temperature if not is_reasoning_model else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value

        # Only include reasoning and verbosity parameters for reasoning models
        if is_reasoning_model and reasoning is not None:
            request_kwargs['reasoning'] = {'effort': reasoning}  # type: ignore

        if is_reasoning_model and verbosity is not None:
            request_kwargs['text'] = {'verbosity': verbosity}  # type: ignore

        response = await self.client.responses.parse(**request_kwargs)

        return response

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format."""
        # Reasoning models don't support temperature and use max_completion_tokens
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
            or ('reasoning' in model and 'non-reasoning' not in model)
        )

        # Use json_schema format when we have a Pydantic model — this enforces
        # valid, complete JSON at the API level (constrained decoding).
        # Falls back to basic json_object mode when no schema is provided.
        if response_model is not None:
            schema_name = getattr(response_model, '__name__', 'structured_response')
            json_schema = response_model.model_json_schema()
            response_format: dict = {
                'type': 'json_schema',
                'json_schema': {
                    'name': schema_name,
                    'schema': json_schema,
                },
            }
        else:
            response_format = {'type': 'json_object'}

        if is_reasoning_model:
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
        else:
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
