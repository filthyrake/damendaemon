# Flash CPU still-image and tool generation

The explicit `FlashImageToolReferenceGenerator.generate_image_tools` path combines
prepared RGB still images with schema-bound tool output. It reuses image
preprocessing, visual encoding, feature insertion, media-output suppression and
owned language-cache cleanup. Calls are data and are never executed.

```sh
python -m scripts.run_glm_flash_image_reference /path/to/local/snapshot \
  --messages image-conversation.json --tools functions.json --max-new-tokens 512
```

Image messages retain bounded inline PNG/JPEG `image_url` input; the tools file is
local JSON bounded to 1 MiB. The shared pixel decoder receives only image-bearing
user content after the complete Flash history validates. This avoids applying an
unrelated model family's tool naming/history rules. URLs and local image paths
are not fetched. Schemas, call/result identities and typed content validate before
pixel decoding; expanded context validates before checkpoint payload loading.

`FlashImageToolText.image_tool_chat_tokens` is the explicit prepared interface.
User multipart content accepts `text` parts and prepared `image` markers, with
at most 4,096 parts / 256 markers and 1 MiB rendered prompt. Inline decoding keeps
its tighter four-image/default encoded-byte and pixel bounds. Literal media or
model/tool controls in caller text are rejected; video/audio and image/tool
batches remain unsupported. Atomic image marker/grid expansion uses the same
existing still-image tokenizer contract.

The complete tool-history rules still apply: one result per call ID, ordered
rendering by call identity, pinned reasoning retention and schema-aware argument
types. Return subsequent tool results alongside the original image history;
image features are recomputed for each owned request. No prefix/image feature
reuse is claimed. Output uses the same atomic validated call stream as text tools,
while media tokens remain excluded from output sampling.

Context, cache, visual/text weight, preprocessing and workspace bounds are
unchanged. Close abandoned iterators; parser/decode errors release state, and
failed cache releases remain owned for retry. Two combined prompt cases matched
the pinned full template and expanded official tokenizer IDs. Prepared/scripted
fixtures cover image-to-call/result continuation and failure paths; a tiny combined
snapshot exercises the inline CLI. These are software controls, not official
visual/tool semantics, hardware performance or long-context/SLO qualification.
GitHub #57/#59/#60 retain those broader gates.
