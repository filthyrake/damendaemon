# Flash tool declaration and result-history prompts

`glm_flash_tool_text.render_tool_chat(messages, tools, ...)` independently renders
the pinned Flash text/tool subset. `FlashToolText.tool_chat_tokens` tokenizes that
explicit path; ordinary `chat_tokens` and image admission remain unchanged.
No Jinja template is executed by this adapter.

The system prefix and function serialization match GLM-5.3-Flash revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Function field order is preserved and
`strict` is omitted from the rendered schema exactly as in the pinned template.
Declarations use the schema-bound codec subset documented in
[Flash tool codec](flash-tool-codec.md). Deferred tools, tool references, media,
built-in tools and permissive fallback history forms are unsupported.

Histories allow an optional leading text system message and alternating text
user/assistant turns. Assistant tool calls require unique nonempty IDs and
schema-valid JSON-string arguments. Each call must have exactly one text tool
result with the matching `tool_call_id`. Results may arrive out of order: the
adapter renders them in assistant call order beneath the observation marker.
Missing, duplicate or foreign identities fail. Generation may resume after a
final user message or a complete tool-result block.

Assistant `reasoning_content` is explicit. `clear_thinking=True` removes reasoning
from earlier turns while preserving tool-call reasoning after the latest user,
matching the pinned template. Inline reasoning control markers inside content
are rejected in this narrow adapter. Caller text/result/declaration fields cannot
inject reserved model/tool delimiters.

Limits are 4,096 messages, the codec's 64 declared functions/calls and schema
bounds, 1 MiB complete rendered prompt, and 1,048,576 encoded tokens. All history,
identity, text and byte checks happen before tokenization. Six retained upstream
full-prompt cases also matched official tokenizer IDs during implementation;
ordinary tests replay the prompts with the local synthetic tokenizer. The pinned
fixture preserves both the expected IDs and template hash.

This is prompt/tokenizer capability under #123/#57. Incremental output parsing,
generation integration, official-weight semantic behavior and hardware serving
remain separate qualification work. No tools are executed.
