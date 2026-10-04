# Native Qwen function-call qualification

Chat completions accept function `tools` and auto, none, required or named
`tool_choice`. Required/named modes need the optional structured dependencies.
Initial format scope is the published JSON `<tool_call>` templates in
Qwen2.5-0.5B-Instruct and Qwen3-0.6B. A qualified model/template is required;
other architecture formats do not silently inherit a parser. No tool function
is imported or executed by dllm.

The independent incremental parser recognizes complete declared calls with JSON
object arguments. It preserves ordinary text, reasoning, malformed/undeclared
calls and incomplete output as content. Literal closing tags in argument strings
do not end a call. Streamed calls receive request-local IDs and consecutive
indices; each call is emitted after its complete envelope is validated, rather
than streaming unvalidated partial arguments. Calls inside `<think>` blocks
remain text. Reasoning retains the existing inline-content representation.

Assistant history accepts standard string-encoded function arguments, paired
with text `tool` results and matching `tool_call_id`. It rejects orphan/duplicate
results, reused IDs and unanswered calls. Arguments become objects only in a
copied template input, as required by Qwen2's published template. Template JSON
preserves insertion order, Unicode and angle brackets to match tokenizer
semantics. Caller-owned tool/message objects are copied for queued requests.

At most64 function definitions/calls are supported per request/message. Auto
mode does not promise argument-schema validation: only declared function names
and well-formed object arguments are extracted. Parallel-call policy overrides, scores, custom stop strings and
structured response formats alongside active tools are rejected. `none` omits
tool definitions and returns ordinary text; existing tool history can still be
rendered. A complete call at normal EOS changes the finish reason to
`tool_calls`; token-limit termination keeps `length`, including truncated text.
Additional formats and automatic strict selection remain in #32.

## Real GPU proof

Both A2 CUDA and M4 Metal pass `scripts/verify_tools_service.py`, with Qwen3
thinking disabled and enabled. CUDA uses `DLLM_QWEN3_PRECISION=float32`; Metal
uses its native FP32 path. The script checks:

- Exact tokenizer-reference IDs for tool definitions and result history.
- Actual greedy `get_weather(city="Paris", unit="celsius")` generation.
- A manually supplied synthetic21°C result followed by a completed answer.
- Stream/nonstream content, function arguments, finish reason and usage equality.
- `tool_choice:none`, four concurrent calls with distinct IDs, batch size4,
  host restoration after model switching, and client-disconnect recovery.

The synthetic result is supplied as test data; no weather service is contacted.
The four JSON reports retain the actual replies. These are behavior/correctness
checks, not throughput or broad tool-selection-quality benchmarks.

Parser/API unit cases cover every split of a mixed multiple-call/Unicode example,
character-by-character input, reasoning markers, strings containing closing tags,
escapes/nested values, malformed/duplicate-key/non-finite JSON, undeclared names,
truncation, history validation, unsupported combinations and the call limit.

Pins match the architecture evidence:
Qwen2.5 `7ae557604adf67be50417f59c2c2f167def9a775`;
Qwen3 `c1899de289a04d12100db370d81485cdf75e47ca`.

## Required and named calls

Required mode produces one declared function call. Named mode selects exactly
one declared function. Both constrain the complete name/arguments object with
the existing xgrammar backend and the bounded JSON-schema subset; `anyOf`
branches keep each function name correlated with its own parameters. The model
prompt includes a `<tool_call>` prefix, tokenized with the full rendered prompt.
Only the JSON body is generated. An independent final schema check guards
serialization. Argument requirements apply even when `strict` is omitted;
`strict:true` is accepted for required/named modes and rejected for auto.

These modes disable default Qwen3 thinking and reject explicit thinking. They
do not generate multiple calls in one response. Unsupported schema keywords
fail before admission; existing 16 KiB/256-node/depth16 limits also bound unions.
Incomplete token-limited JSON is returned as content with `length`, never as a
successful forced call. A malformed normal-EOS result is an error.

`verify_forced_tools.py` passes on A2 and M4 for both qualified checkpoints:
required selection, forced translation despite a weather prompt, exact parameter
constants, full-prompt tokenizer equality, streaming parity, mixed required/named
batches, unique IDs, model restoration, truncation and disconnect recovery.
`verify_tool_grammar.py` independently checks actual xgrammar acceptance and
rejection for both modes, including crossed function/parameter branches. Reports
are `forced-tools-cuda.json`, `forced-tools-mps.json` and `tool-grammar.json`.
These are correctness checks, not comparative performance measurements.
