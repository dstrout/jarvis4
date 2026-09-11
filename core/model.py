"""
Jarvis4 model wrapper — logging and provider-specific behavior.

Wraps OpenAIChatCompletionsModel to:
1. Log every LLM call with full request/response/reasoning detail
2. Support provider-specific parameters (Cerebras reasoning_effort, etc.)
3. Provide a ModelProvider for per-model configuration

Works with any OpenAI-compatible API. Cerebras-specific behavior
(reasoning field extraction) activates automatically when present.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from agents import ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.interface import Model, ModelProvider
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem
from agents.tool import Tool
from agents.models.interface import ModelResponse, ModelTracing

from core.logging_handler import get_logger

log = logging.getLogger("jarvis4.model")


class Jarvis4Model(OpenAIChatCompletionsModel):
    """OpenAI Chat Completions model with logging and optional reasoning support."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        agent_name: str = "unknown",
    ):
        super().__init__(model=model, openai_client=openai_client)
        self.agent_name = agent_name

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        t0 = time.perf_counter()

        response = await super().get_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract data from output items for logging
        content = None
        reasoning = None
        tool_calls = None

        for item in response.output:
            item_dict = item if isinstance(item, dict) else (
                item.model_dump() if hasattr(item, "model_dump") else {}
            )
            item_type = item_dict.get("type", "")

            if item_type == "reasoning":
                summaries = item_dict.get("summary", [])
                reasoning_parts = []
                for s in summaries:
                    text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                    if text:
                        reasoning_parts.append(text)
                if reasoning_parts:
                    reasoning = "\n".join(reasoning_parts)

            elif item_type == "message":
                contents = item_dict.get("content", [])
                text_parts = []
                for c in contents:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text_parts.append(c.get("text", ""))
                    elif hasattr(c, "type") and c.type == "output_text":
                        text_parts.append(c.text)
                if text_parts:
                    content = "\n".join(text_parts)

            elif item_type == "function_call":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "name": item_dict.get("name", "?"),
                    "arguments": item_dict.get("arguments", "{}"),
                })

        # Build a simplified messages list for logging
        messages_for_log = []
        if system_instructions:
            messages_for_log.append({"role": "system", "content": system_instructions})
        if isinstance(input, str):
            messages_for_log.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for item in input[-10:]:
                if isinstance(item, dict):
                    messages_for_log.append(item)

        usage_dict = None
        if response.usage:
            usage_dict = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        get_logger().log_llm_call(
            agent_name=self.agent_name,
            model=str(self.model),
            messages=messages_for_log,
            response_content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            elapsed_ms=elapsed_ms,
            usage=usage_dict,
        )

        if reasoning:
            log.info(
                "  [%s] reasoning (%d chars): %s",
                self.agent_name,
                len(reasoning),
                reasoning[:150] + "..." if len(reasoning) > 150 else reasoning,
            )

        return response


class Jarvis4Provider(ModelProvider):
    """Model provider that creates Jarvis4Model instances with logging."""

    def __init__(self, client: AsyncOpenAI, default_model: str = "gpt-oss-120b"):
        self._client = client
        self._default_model = default_model

    def get_model(self, model_name: str | None) -> Model:
        return Jarvis4Model(
            model=model_name or self._default_model,
            openai_client=self._client,
            agent_name="via_provider",
        )


# Backward-compatible aliases
CerebrasModel = Jarvis4Model
CerebrasProvider = Jarvis4Provider
