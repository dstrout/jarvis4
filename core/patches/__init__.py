"""
Patches to the OpenAI Agents SDK for Cerebras compatibility.

These patches add support for Cerebras's `reasoning` field in chat completion
messages. The upstream SDK only handles DeepSeek's `reasoning_content` and
Anthropic's `thinking_blocks`. Cerebras uses a `reasoning` field on the
message object, which the OpenAI client preserves in `model_extra`.

Changes are isolated here so they can be submitted upstream if desired.

UPSTREAM PR NOTES:
-----------------
1. chatcmpl_converter.py / message_to_output_items():
   - Added check for `message.reasoning` (Cerebras convention) alongside
     existing `message.reasoning_content` (DeepSeek convention).
   - Both fields are mapped to the same ResponseReasoningItem format.

2. chatcmpl_converter.py / items_to_messages():
   - Added Cerebras model detection (model name contains "gpt-oss" or
     "cerebras") alongside existing DeepSeek and Claude handling.
   - Maps ResponseReasoningItem summary text back to `reasoning` field
     on assistant messages (Cerebras convention), parallel to how
     DeepSeek uses `reasoning_content`.

3. These changes are additive and non-breaking. Existing DeepSeek and
   Claude paths are untouched.
"""
