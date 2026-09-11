"""
Patch the Agents SDK converter to support Cerebras's `reasoning` field.

Cerebras returns chain-of-thought in a `reasoning` field on the chat completion
message (visible via model_extra when using the OpenAI client). The SDK only
handles DeepSeek's `reasoning_content`. This patch adds symmetric support for
Cerebras's convention.

Apply by calling `apply_patches()` before any agent operations.

UPSTREAM PR NOTES:
-----------------
File: agents/models/chatcmpl_converter.py
Class: Converter

Change 1 - message_to_output_items() around line 129:
  BEFORE:
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        ...create ResponseReasoningItem from message.reasoning_content...

  AFTER:
    # Check for reasoning content from various providers:
    # - DeepSeek/LiteLLM: message.reasoning_content
    # - Cerebras: message.reasoning (in model_extra via OpenAI client)
    reasoning_text = None
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        reasoning_text = message.reasoning_content
    elif hasattr(message, "reasoning") and message.reasoning:
        reasoning_text = message.reasoning
    if reasoning_text:
        ...create ResponseReasoningItem from reasoning_text...

Change 2 - items_to_messages() around line 746 (in reasoning item handling):
  BEFORE:
    elif (model and "deepseek" in model.lower() ...):
        ...set pending_reasoning_content...

  AFTER:
    elif (model and "deepseek" in model.lower() ...):
        ...set pending_reasoning_content...
    # Cerebras uses a `reasoning` field (not `reasoning_content`)
    elif (model and _is_cerebras_model(model) ...):
        ...set pending_reasoning_field...

Change 3 - items_to_messages() in function_call handling (around line 623):
  BEFORE:
    if pending_reasoning_content:
        asst["reasoning_content"] = pending_reasoning_content

  AFTER:
    if pending_reasoning_content:
        asst["reasoning_content"] = pending_reasoning_content
    elif pending_reasoning_field:
        asst["reasoning"] = pending_reasoning_field

Change 4 - items_to_messages() flush_assistant_message (around line 468):
  Add: pending_reasoning_field = None  (reset alongside pending_reasoning_content)
"""

from __future__ import annotations

import logging
from typing import Any

from openai.types.chat import ChatCompletionMessage

from agents.models.chatcmpl_converter import Converter
from agents.items import TResponseOutputItem
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

from agents.models.fake_id import FAKE_RESPONSES_ID

log = logging.getLogger("jarvis.patches")

# Models that use the `reasoning` field convention (not `reasoning_content`)
CEREBRAS_MODEL_PATTERNS = ("gpt-oss",)


def _is_cerebras_reasoning_model(model: str) -> bool:
    """Check if a model uses Cerebras's `reasoning` field convention."""
    model_lower = model.lower()
    return any(p in model_lower for p in CEREBRAS_MODEL_PATTERNS)


# Store references to original methods
_original_message_to_output_items = Converter.message_to_output_items
_original_items_to_messages = Converter.items_to_messages


@classmethod  # type: ignore[misc]
def _patched_message_to_output_items(
    cls,
    message: ChatCompletionMessage,
    provider_data: dict[str, Any] | None = None,
) -> list[TResponseOutputItem]:
    """Patched to also check for Cerebras `reasoning` field."""

    # Check for Cerebras reasoning (in model_extra from OpenAI client)
    cerebras_reasoning = getattr(message, "reasoning", None)

    # If Cerebras reasoning is present but no reasoning_content, inject it
    # so the original method's hasattr check finds it
    if cerebras_reasoning and not getattr(message, "reasoning_content", None):
        # We can't set attributes on frozen Pydantic models, so we build
        # the reasoning item ourselves and prepend it to the original result
        items = _original_message_to_output_items.__func__(cls, message, provider_data)

        reasoning_kwargs: dict[str, Any] = {
            "id": FAKE_RESPONSES_ID,
            "summary": [Summary(text=cerebras_reasoning, type="summary_text")],
            "type": "reasoning",
        }
        if provider_data:
            reasoning_kwargs["provider_data"] = provider_data

        reasoning_item = ResponseReasoningItem(**reasoning_kwargs)
        return [reasoning_item] + items

    return _original_message_to_output_items.__func__(cls, message, provider_data)


@classmethod  # type: ignore[misc]
def _patched_items_to_messages(
    cls,
    items,
    model: str | None = None,
    preserve_thinking_blocks: bool = False,
    preserve_tool_output_all_content: bool = False,
):
    """Patched to restore Cerebras `reasoning` field on assistant messages."""
    # Call original
    messages = _original_items_to_messages.__func__(
        cls, items, model, preserve_thinking_blocks, preserve_tool_output_all_content,
    )

    if not model or not _is_cerebras_reasoning_model(model):
        return messages

    # Post-process: find reasoning items in the input and attach them
    # to the corresponding assistant messages in the output.
    #
    # The original method ignores reasoning items for non-DeepSeek/Claude models.
    # We need to extract reasoning text from ResponseReasoningItem objects
    # and attach it as a `reasoning` field on the next assistant message.

    # Re-scan input items for reasoning content
    from agents.models.chatcmpl_converter import Converter as Conv

    reasoning_texts = []
    if not isinstance(items, str):
        for item in items:
            if reasoning_item := Conv.maybe_reasoning_message(item):
                summary_items = reasoning_item.get("summary", [])
                for s in summary_items:
                    if isinstance(s, dict) and s.get("text"):
                        reasoning_texts.append(s["text"])

    if not reasoning_texts:
        return messages

    # Attach reasoning to assistant messages that don't already have it.
    # Reasoning items precede the assistant message they belong to in the
    # item stream. We match them positionally: each reasoning text goes
    # with the next assistant message.
    reasoning_iter = iter(reasoning_texts)
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning" not in msg:
            try:
                msg["reasoning"] = next(reasoning_iter)  # type: ignore[typeddict-unknown-key]
            except StopIteration:
                break

    return messages


def apply_patches():
    """Monkey-patch the Converter class to support Cerebras reasoning.

    Safe to call multiple times — patches are idempotent.
    """
    if getattr(Converter, "_cerebras_patched", False):
        return

    Converter.message_to_output_items = _patched_message_to_output_items  # type: ignore[assignment]
    Converter.items_to_messages = _patched_items_to_messages  # type: ignore[assignment]
    Converter._cerebras_patched = True  # type: ignore[attr-defined]

    log.info("Applied Cerebras reasoning patches to Agents SDK Converter")
