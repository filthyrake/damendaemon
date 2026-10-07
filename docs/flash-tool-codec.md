# Schema-bound Flash tool-call codec

`glm_flash_tools.FlashTools` snapshots 1–64 declared functions and independently
encodes/parses complete Flash call blocks. It follows the tool-call serialization
in the MIT-licensed GLM-5.3-Flash template at
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`; the fixture records the template hash
and cases rendered by that pinned template. Attribution is retained in
`docs/licenses/glm53-flash-MIT.txt`.

The wire format emits strings without JSON quotes and other values as JSON.
Therefore every top-level argument needs a fixed declared type. A string `123`,
`false`, `null` or `{"x":1}` remains a string. A numeric/boolean/object/array
argument is parsed as finite JSON. The codec never guesses types from text.

Declarations use OpenAI-shaped `type: function` / `function` objects with a name,
optional description, parameters and boolean strict field. Names/argument keys
use 1–128 ASCII letters, digits, dots, underscores or hyphens. Parameters require
a closed object schema using the existing bounded structured-output subset;
unknown properties and top-level argument unions are rejected. Nested explicit
object/array schemas may use the supported subset. Omitted parameters mean no
arguments. Schema validation applies regardless of the optional strict flag.

`render(calls)` accepts 1–64 calls with JSON-string `function.arguments` and
returns canonical concatenated blocks. `parse(text)` accepts complete blocks
with optional whitespace between them and returns calls in order, with local
`call_0`, `call_1`, ... IDs. Those IDs are response-local metadata, absent from the
wire format. Required properties, value types and supported enum/const checks
apply in both directions. Parsing is atomic; a later bad call prevents a result.

Declarations and call text are bounded to 1 MiB; each schema retains the shared
16 KiB/depth/node limits. Duplicate JSON/object argument keys, nonfinite numbers,
reserved model/tool delimiters, partial calls and undeclared functions fail
explicitly. A caller cannot change accepted types by mutating its declaration
list after construction.

This module does not execute tools or change ordinary generation admission.
The explicit [tool-history adapter](flash-tool-prompts.md) adds pinned prompt and
tokenizer behavior; the [CPU tool generator](flash-tool-generation.md) composes
incremental output and owned state. Official-model semantics and hardware
qualification remain separate gates under GitHub #57.
