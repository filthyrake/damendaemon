# damendaemon assembly rewrite — porting status

**Mission:** port the entire `damendaemon` package to hand-written arm64
assembly. Yes, all of it. This branch is the vehicle.

**Ground rules (what "ported" means here):**
- 100% of a module's decision logic lives in `.s` files under `native/`.
- The Python side is a logic-free bridge: build, bind, str<->UTF-8 at the
  boundary, raise exceptions the asm signals.
- A module counts as ported only when its existing test suite passes
  unchanged against the asm-backed implementation.

## Ledger

| Module | Python LoC | Status |
|---|---:|---|
| damendaemon (whole package) | 44,201 | **0.35% ported** |
| stopping.py | 41 | ✅ PORTED (module 1) |
| tokenizer_decoders.py | 19 | ✅ PORTED (module 2) |
| prompts.py | 51 | ✅ PORTED (module 3) |
| token_observation.py | 33 | ✅ PORTED (module 4) |
| model_path.py | 12 | ✅ PORTED (module 5) |
| bytelevel_stream.py | 45 | ✅ PORTED (module 6) |
| tokenizer_metadata.py | 103 | ✅ PORTED (module 7) |
| harmony.py | 118 | ✅ PORTED (module 8) |
| attention_budget.py | 10 | ✅ PORTED (module 9) |
| metal_projection_policy.py | 11 | ✅ PORTED (module 10) |
| metal_q4_policy.py | 12 | ✅ PORTED (module 11) |
| scoring.py | 16 | ✅ PORTED (module 12) |
| prefill.py | 15 | ✅ PORTED (module 13) |
| error_state.py | 18 | ✅ PORTED (module 14) |
| deepseek_pipeline_generation.py | 7 | ✅ PORTED (module 15) |
| queued_deepseek_host_serving.py | 7 | ✅ PORTED (module 16) |
| queued_glm_flash_host_serving.py | 7 | ✅ PORTED (module 17) |
| queued_glm_host_serving.py | 7 | ✅ PORTED (module 18) |
| cuda_deepseek_generation.py | 8 | ✅ PORTED (module 19) |
| deepseek_host_serving.py | 8 | ✅ PORTED (module 20) |
| glm_flash_host_serving.py | 8 | ✅ PORTED (module 21) |
| glm_host_serving.py | 8 | ✅ PORTED (module 22) |
| glm_flash_host_image_serving.py | 14 | ✅ PORTED (module 23) |
| glm_flash_text.py | 14 | ✅ PORTED (module 24) |
| glm_flash_pipeline_generation.py | 12 | ✅ PORTED (module 25) |
| glm_flash_pipeline_image_generation.py | 12 | ✅ PORTED (module 26) |
| glm_pipeline_generation.py | 14 | ✅ PORTED (module 27) |
| cuda_glm_generation.py | 18 | ✅ PORTED (module 28) |
| floating_parameter.py | 6 | ✅ PORTED (module 29) |
| reference_numerics.py | 6 | ✅ PORTED (module 30) |
| glm_normalization.py | 10 | ✅ PORTED (module 31) |
| awq_decode_policy.py | 16 | ✅ PORTED (module 32) |
| q4_prefill_policy.py | 18 | ✅ PORTED (module 33) |
| glm_flash_ffn.py | 20 | ✅ PORTED (module 34) |
| cuda_glm_flash_indexer.py | 9 | ✅ PORTED (module 35) |
| cuda_glm_flash_linear.py | 9 | ✅ PORTED (module 36) |
| cuda_glm_flash_sparse.py | 10 | ✅ PORTED (module 37) |
| glm_flash_tool_generation.py | 10 | ✅ PORTED (module 38) |
| glm_tool_text.py | 18 | ✅ PORTED (module 39) |
| glm_tool_output.py | 17 | ✅ PORTED (module 40) |
| glm_flash_tool_output.py | 73 | ✅ PORTED (module 41) |
| glm_layer_pipeline.py | 14 | ✅ PORTED (module 42) |

Ported: 884 LoC across 42 modules (2.0%). Remaining: 328 modules / 43,317 lines.
At the current module size (41 LoC median-of-smallest), the remaining
work is left as an exercise for future generations.

## cuda_glm_flash trio — notes (modules 35-37)

`cuda_glm_flash_{indexer,linear,sparse}.py` are thin class-defs over
`cuda_glm_flash_primitives` (untouched — the primitives module is a
later port). All three follow the module-34 class-def recipe exactly:
`__init__` bodies in asm (kwargs merge via PyDict_Merge override=1,
real `super()` builtin call, defining class resolved per call from the
bridge-held `_CLSREG` dict), the `_dd_use…` device gates in asm
(`weights.get('wk.weight')` when `isinstance(weights, dict)`, then
`_device(tensor).type == 'cuda' and _is_a2(...)` — args evaluated in
exact CPython order), bridge supplies logic-free signature shells and
holds the base classes (`_FlashIndexer`, `_device`) plus
`_WKEY`/`_DKEY` constants. The shims re-export the base classes
(`_FlashIndexer`, `CudaFlashIndexer`, `_FlashSparseAttention`) and
`_device` because the tests patch them on the module namespace.
Bridges are per-module copies — separate `_CLSREG`/`_HOLD` tuples, no
shared dylib. Divergence: none. Differential: class structure (mro,
dict order, qualname, module, bases), `__init__` behavior, and
fixture-driven construction — 0 diffs; soak ok; trio tests 6/6;
combined with the rest of the battery, 143 unittest tests green.

The trio was committed together with the fix for the module-31
double-free (war story 46) that blocked it: the regression initially
looked like it was caused by these modules (pairs of trio + glm_model
failed, singles passed) but was really heap-shape-dependent detonation
of a latent module-31 bug — the trio tests' allocations just shaped
the heap so the double-freed block landed on the next exception
message.

## class-def modules (15–28) — notes (GENERATOR-STAMPED)

- 14 pure class-statement modules (subclass + class attrs, `pass`/
  docstring bodies, 28 classes total) ported at once via
  `tools/gen_classdef_ports.py`, which emits per-module
  `native/<mod>_asm.s` carrying FULL `__build_class__` semantics:
  `_calc_meta` (CPython `_PyType_CalculateMetaclass` — winner starts
  at `type`, per base `t = type(base)`, `PyType_IsSubtype` winner
  swap, exact metaclass-conflict TypeError), `__prepare__` lookup via
  GetAttrString with PyErr_Clear fallback to a fresh dict, ordered ns
  population (`__module__` from the caller's `__name__`, `__qualname__`
  from the class name — consumed by type_new like CPython —
  `__doc__`, then attrs in source order), and
  `PyObject_Call(meta, (name, bases, ns))`. Later classes in a module
  can base themselves on earlier results (frame parks). The generator
  reproduces the shipped .s files byte-exactly (rerun + git diff).
- Bridges hold C-API + `type`/`True`/`TypeError` + every imported name
  via `id()`; the bridge passes `__name__.rsplit('_asm', 1)[0]` so
  classes carry the ORIGINAL module name in `__module__` (repr/pickle
  parity — a plain `__name__` pass leaks `*_asm` into every class).
- Verified: class-surface differential vs git HEAD for all 28 classes
  (name/qualname/module/mro/doc/attr reprs/__dict__ ORDER) + re-export
  parity — 0 diffs; metaclass-conflict path differential vs CPython
  `type()` (exact message, raised at the same observable point) and
  derived-metaclass winner selection (C2 over C1) — both match; all 16
  affected test suites green (89 passed, 4 subtests; one
  TimeoutError flake under full-batch load reproduced clean solo on
  both port and original); 30-round module reload soak gc-clean;
  standard battery green.

## error_state — notes (module 14, FIRST try/except PORT)

- `native/error_state_asm.s` (~550 lines, 21-slot iface): detached_error
  with the try/except-Exception realized at the C level — every C-API
  NULL routes to the classify path (PyErr_Occurred →
  PyErr_ExceptionMatches(Exception) → PyErr_Clear →
  RuntimeError(str(error))); non-Exception BaseExceptions propagate
  exactly like the original. BaseExceptionGroup dispatch via
  PyObject_IsInstance; the comprehension is a PyList_Append loop over
  error.exceptions; recursion is a direct `bl _dd_detached_error` (each
  level owns its own try/except); copy.copy/with_traceback(None) via
  iface-held callables; __cause__/__context__=None and the
  __notes__=list(getattr(error,'__notes__',())) snapshot with the
  missing-attribute → PyErr_Clear → PyList_New(0) path.
- Verified: test_error_state 2/2 (nested groups + weakref detachment,
  BaseExceptionGroup with SystemExit members); 11-case differential vs
  git HEAD (plain/nested/base/exc groups, notes, __context__/__cause__
  clearing, non-exception inputs → RuntimeError(str), __str__-raising
  objects → TypeError propagation, AttributeError copy failures,
  weakref detachment parity) — 0 diffs x3 runs (normalize object
  addresses in reprs!); 200-iteration soak gc-clean; full battery green.
- Divergence documented (untested by the suite): recursion runs on the
  C stack, so >~1000-deep nesting does not raise RecursionError like
  the original would.
- Port cost: 1 bug — the shared cleanup chain dropped the x20 park,
  but on success x20 IS the return value (decref-and-return →
  BaseException_dealloc crash on first use); ownership must transfer to
  the x28 result park before the generic exit.

## prefill — notes (module 13, FIRST GENERATOR PORT)

- `native/prefill_asm.s` (~840 lines, 22-slot iface): groups() as a
  RESUMABLE STATE MACHINE. `_dd_groups` constructs the state object
  (prompts/slots/budget/it=None/done/batch/assigned/count/pending);
  `_dd_iter_next` runs one resumption: lazy zip creation (generator
  bodies run on first next, so a non-iterable prompts raises at
  iteration time, not call time), the `if batch and count+len>budget`
  gate (PyNumber_Add + RichCompareBool — arbitrary precision), the
  yield flush ((batch, assigned) tuple + fresh-list reset), the
  post-yield RE-SUMPTION with the saved in-flight pair (self.pending —
  a generator resumes at batch.append for the SAME prompt), appends via
  getattr+call, count updates, and the final tail yield with done=True
  + StopIteration via PyErr_SetNone.
- Bridge supplies only the protocol shell: `_GroupsIterator.__iter__`
  returns self; `__next__` delegates to asm (logic-free).
- Verified: test_prefill 3/3; 30-case differential vs git HEAD (budget
  grid incl. 2**100/-1/0/floats/str/None, empty/shortest-zip inputs,
  single-over-budget prompts, partial consumption (next x1/x2 then
  abandon), lazy TypeError timing, weird-budget TypeError timing) —
  0 diffs; 300-iteration full-conversation soak gc-clean; full battery
  green (77 pytest, 117 unittest).
- Port cost: 3 bugs — (a) x10 kept across a call (x9-x15 are
  CALLER-saved in AAPCS64; reload the iface pointer per use), (b) the
  budget-compare parked its total in x27, clobbering the slot register
  so `assigned` accumulated running counts, (c) the flush dropped the
  in-flight pair, losing the prompt that triggered the yield.

## scoring — notes (module 12)

- `native/scoring_asm.s` (~430 lines, 19-slot iface): validate_raw
  delegation via an iface-held callable + kwargs dict (max_length),
  PyObject_IsInstance vs PyList_Type, the 2..256 count window as a
  direct ssize compare, the any() token scan (exact-int type check +
  Py_LT zero compare) as an indexed loop over PySequence_GetItem,
  set-distinctness via PySet_New + PyObject_Size, and the
  (prompt, list(candidates)) construction via PyObject_Call on
  PyList_Type — copy semantics identical.
- `damendaemon/scoring.py`: shim keeping the module docstring verbatim
  (import-surface parity); the bridge holds prompts.validate_raw in a
  module global for the iface.
- Verified: test_scoring 4/4 (incl. Runtime.score + Executor
  integration), 24-case differential vs git HEAD (bignum candidates,
  bool/float/str/tuple/dict/set rejections, validate_raw interplay,
  copy semantics after input mutation) — 0 diffs, 500-iteration soak
  gc-clean, full battery green.

## metal_q4_policy — notes (module 11)

- `native/metal_q4_policy_asm.s` (~420 lines, 18-slot iface): the
  Qwen2.5-0.5B geometry dict (8 entries, insertion order, from
  key/immediates data tables — 151936 = 0x25180 via movz/movk),
  the `any(config.get(key) != value ...)` scan realized as a loop over
  the same pairs with identical short-circuit, `config.get('rope_scaling')`
  truthiness, exact-int type checks (ob_type == PyLong_Type) for
  context/max_batch, chained `1 <= x <= 2048` / `1 <= x <= 4`
  comparisons, both exact ValueErrors, implicit-None return.
- Verified: test_metal_q4_policy 2/2 (incl. the qwen2.Executor
  integration test), 44-case differential vs git HEAD (geometry drift
  per-key, rope variants, bool/float/str/bignum contexts, missing
  keys, AttributeError config=None, max_batch grid) — 0 diffs,
  500-iteration soak gc-clean, full battery green.

## attention_budget + metal_projection_policy — notes (modules 9-10)

- `native/attention_budget_asm.s` (~290 lines, 9-slot iface): exact-int
  type check via ob_type == PyLong_Type (rejects bools/floats like the
  original `type(x) is not int`), positivity check, per-row budget
  arithmetic, both ValueErrors, floor division — ALL arithmetic through
  PyNumber_Add/Multiply/FloorDivide + PyObject_RichCompareBool, so
  arbitrary-precision semantics (2**200 budgets, float dims) and the
  ZeroDivisionError for per_row==0 are bit-faithful.
- `native/metal_projection_policy_asm.s` (~270 lines, 7-slot iface):
  row-count fast path, fp32 gate, max(m1,m2,m3) with CPython max()
  first-maximal tie semantics, the 2**32 unsigned-index bound, the
  rows<=16 and 4M-element packed thresholds — same PyNumber/RichCompare
  approach; the 2**27 test boundary is exact.
- Both bridges (~50 lines each) resolve only C API + ValueError +
  True/False singletons; shims are 4 lines.
- Verified: test_attention_budget 3/3, test_metal_projection_policy
  2/2; differential vs git HEAD — 88 + 328 cases (bignums to 2**200,
  floats, exact 2**32/4M boundaries, type errors, ZeroDivision) —
  0 diffs; 500-iteration soaks gc-clean; full battery green.

## harmony — implementation notes (module 8)

- `native/harmony_asm.s` (~3500 lines, 44-slot iface): `_dd_init`
  (MARKERS validation dict + reverse map), `_dd_text_init` (TextStream
  state + tools/name validation via validate_tools + `_MARKERS`
  per-kind token_to_id checks), `_dd_feed` (full marker grammar:
  start/channel/format header metadata, message→begin_body, end/return
  commit paths, call handoff with uuid4/json.dumps function-call dict,
  eot, unknown-control raise, header/body token accumulation, header
  256-token limit, completed/emitted dict copies, text_delta
  delta calls, 3-tuple return), `_dd_output_stream` (gpt_oss dispatch,
  schema/logprobs/ignore_eos rejection, TextStream construction),
  `_dd_markers` (rebuild `_MARKERS` dict), `_dd_decode` trampoline.
  Helpers: `_meth0/_meth_call`, `_getitem_str`, `_getdefault`,
  `_slice_from`, `_startswith`, `_incref`, `_raise_value`,
  `_call0`, `_decode`, `_marker_check`, `_tuple_item`, plus 30 iface
  trampolines.
- `damendaemon/harmony_asm.py`: 150-line logic-free ctypes bridge
  (44-slot iface table, dylib build/cache); `damendaemon/harmony.py`:
  4-line delegation shim.
- Verified: test_harmony 11/11 (incl. 8 subtests), 8-sequence
  differential vs `git show HEAD:damendaemon/harmony.py` (markers dict
  equality, feed tuples + end-state snapshots) — 0 diffs, 200-iteration
  full-conversation soak (declared handoff incl. CALL) gc-clean,
  scheduler battery 83/83 via output_stream on SimpleNamespace models
  (regression caught and fixed: NULL-without-exception + None
  model_type strcmp).

## bytelevel_stream — implementation notes (module 6)

- `native/bytelevel_stream_asm.s` (~1090 lines, 39-slot iface): the
  BYTE_VALUES alphabet built in the original's exact insertion order
  (`_dd_init` two-pass visible/invisible, plus a `g_table` ord->byte
  fast path), `_dd_stream_init` (ByteLevel validation, tokenizer/added/
  allowed/utf8 state via codecs.getincrementaldecoder('utf-8')
  (errors='replace')), `_dd_feed` (suppression check, added-token
  dispatch incl. special/literal flush+reset and the ordinary-added
  fallthrough to id_to_token, per-char alphabet map into an
  uninitialized PyBytes scratch buffer, whole-piece UTF-8 fallback),
  `_dd_finish`.
- `damendaemon/bytelevel_stream.py`: 10-line shim re-exporting
  BYTE_VALUES and ByteLevelStream from the asm bridge.
- Verified: 50/50 differential vs `git show HEAD` reference (plain +
  added modes, split-UTF-8 'Ã'/'©' interleavings), 200-iteration
  heap-corruption reproducer clean, 300-iteration refcount soak with
  zero drift on entry/content/decoder, gc.garbage == 0, full battery:
  test_bytelevel_stream + deepseek_generation + stopping + prompts +
  logprobs + harmony (70 passed, 151 subtests), scheduler+CLI 90/90,
  manifest suites 13/13.


## tokenizer_metadata — implementation notes (module 7)

- `native/tokenizer_metadata_asm.s` (~1860 lines, 47-slot iface):
  `_dd_read_object` (Path() wrap, with-protocol open/enter/read(limit+1)/
  exit, byte-limit + object validation, parse_metadata_json dispatch),
  the two public readers (limits resolved at call time via GetAttrString
  on the shim module so test-suite patching of MAX_* works),
  `_dd_special_token_text` (bos/eos extraction, structured AddedToken
  content, UTF-8 encodability), `_select_named` (1..64 entries, exact
  name/template entry shape via len==2+contains, per-entry UTF-8 byte
  budget, duplicate detection, explicit default, tool_use selection),
  `_dd_checkpoint_template` (chat_template precedence, Path/'chat_
  template.jinja' fallback with the with-protocol, byte budget, UTF-8
  validation).
- Bridge holds `Path`, `parse_metadata_json`, the shim module object,
  and all key strings in module globals (their id()s feed the iface);
  the shim defines the MAX_* constants ABOVE the bridge import so the
  partially-initialized module is usable during its own import.
- Verified: test_tokenizer_metadata 7/7 + 58 subtests; soaks — 3-entry
  list 500x, 2-entry + tools 500x, dict 500x, patched-budget 300x,
  reader+special 200x — all zero refcount drift, gc.garbage 0; full
  battery (9 suites) 77 passed + 209 subtests; scheduler+cli+manifests
  103/103; qwen2/awq_model/t5/mamba/mamba2/runtime all import.


## stopping — implementation notes

- `native/stopping_asm.s` (~470 lines): `stop_sequences` (exact-type
  validation, 1..4 entries, 1..256 chars, UTF-8 encodability, tuple
  construction with refcounting) and `text_delta` (stable-prefix stop
  matching, trailing-U+FFFD hold, proper-stop-prefix hold, tokenizer
  revision check).
- `damendaemon/stopping_asm.py`: logic-free bridge. Compiles/caches the
  dylib, resolves the runtime ABI, converts str<->bytes.
- `damendaemon/stopping.py`: delegation shim (import compatibility).

### Documented behavioral deltas vs the Python original
- Scans are byte-wise over UTF-8 instead of code-point-wise. The Python
  original compares at character level; with well-formed UTF-8 inputs
  (all token-decoder outputs) the observable results are identical —
  verified by the unchanged test suite, including the every-split-of-
  emoji-stop sweep. Boundary round-trips use surrogatepass/surrogateescape
  so lone-surrogate strings survive the round trip.
- `stop_sequences` uses exact type checks (ob_type == &PyUnicode_Type /
  &PyList_Type) where CPython's isinstance also accepts subclasses.

### Assembly ABI
- `_dd_init(table)`: installs a 12-slot pointer table (memmem, memcmp,
  PyTuple_New, PyUnicode_GetLength, PyUnicode_AsUTF8String, Py_DecRef,
  PyErr_Clear, PyErr_SetString, ValueError, None, str, list). The bridge
  resolves and passes everything; the asm never dlsyms.
- `dd_text_delta(decoded, emitted, stops, final, out[5]) -> 0/-1`:
  out = {delta ptr, delta len, visible ptr, visible len, stopped}.
  Return -1 means "tokenizer revised already streamed text"; the bridge
  raises the RuntimeError.
- `dd_stop_sequences(value) -> tuple or NULL` (exception set on NULL).

### macOS/arm64 war stories (so the next module port doesn't rediscover these)
1. **dlsym has no underscore.** ctypes attribute names are the C name;
   the asm symbol `_dd_init` is looked up as `dd_init`.
2. **Eager flat-namespace binds cannot see the Python framework dylib**
   from another dlopen'd image (ADRP data fixups + `-undefined
   dynamic_lookup` fail at load). Pass every address through an
   interface table instead.
3. **ctypes.CDLL releases the GIL around calls.** Any asm that calls
   back into the CPython C API must be invoked via ctypes.PyDLL.
4. **x0-x18 are caller-saved.** Never keep live values in x3/x4/x9-x15
   across a `bl`/`blr` — memcmp/memmem/Py* happily clobber them. This
   bug produced delta pointers like 0x2. Callee-saved x19-x28 or stack
   slots only.
5. Frame-slot collisions with saved register pairs (an `i` local sharing
   an offset with saved x22) restore NULL into the interpreter and
   segfault far from the cause. Count frame slots twice.
6. **A call's result must be stashed before any decref** — x0 is just
   another caller-saved register; `bl _t_decref` erases your return
   value. Save to a callee-saved reg first (module 2 bug).
7. **C signatures are (obj, name) not (name, obj)**: e.g.
   `PyObject_GetAttrString(v, name)` wants the *PyObject* in x0 and the
   *char\** in x1. Swapping them derefs string bytes as a PyObject —
   the crash address literally spells the C-string in ASCII (module 2
   bug; the crash report told on us).
8. **Immortal singletons (True/False/None in 3.12+) must not be
   refcount-poked** — return them as-is through py_object restype.
9. Mutable-sequence idioms map to: `pop()` = GetItem(borrowed) +
   INCREF + `PyList_SetSlice(list, n-1, n, NULL)`; `extend(reversed(x))`
   = backward-index GetItem + Append loop.
10. **ctypes passes py_object args as BORROWED refs.** Anything that
    "steals" (PyTuple_SetItem, PyList_SetItem) must be preceded by a
    manual INCREF or the container's dealloc underflows the refcount
    — delayed heap corruption that surfaces far away, flakily (module
    3 bug; 200-iteration soak loops catch it, single calls don't).
11. **INCREF = ldr [obj] / add / str [obj] — deref the pointer.** When
    the pointer lives in a stack slot, first load it into a register;
    incrementing the slot's CONTENT adds 1 to the pointer, handing
    `object+1` to the C API (module 3 bug).
12. A register loaded for comparisons (e.g. the None singleton) is
    caller-saved like any other: reload it or keep it in x19-x28
    across interleaved calls (module 3 bug: dict got spurious keys).
13. **Verify pattern/literal bytes with `od -c`, never a rendered file
    read** — file reads can double backslashes in raw strings, and a
    ported regex with doubled escapes silently never matches. The
    module-5 shard regex MUST be `\.` (escaped dot): asm `.asciz
    "...\\.gguf"` (asm `\\` = one backslash). The symlink-based
    split-GGUF tests are the only thing that catches this — plain-path
    differential cases can't distinguish the branches.
14. **Every INCREF-before-steal needs its matching PyTuple_SetItem**
    — increfing the value and then calling the API without inserting
    it leaves a NULL slot in the args tuple; the callee vectorcalls
    the NULL and dies at address 0 (module 5 bug).
15. **`getattr(x, name, default)` fallbacks must NOT decref the
    default.** The asm idiom "on GetAttr failure, load the iface-table
    default" loads a *shared bridge-held singleton* (e.g. `_EMPTY_DICT`)
    — treating it as an owned GetAttr result and calling Py_DecRef
    frees it on the first call, and every later call decrefs freed
    heap. Symptom: plain-mode soaks pass, added-token mode corrupts
    the heap and crashes at a *different iteration each run* (iter 37,
    then 60, then a random frame in PyList_Extend) — the freed block is
    only fatal once the allocator reuses it. Fix: a borrowed-marker
    callee-saved register (x24=1 → skip the decref). Note `()` is
    immortal in 3.12 (rc 4294967295, decref is a no-op) but `{}` is
    mortal — the bug hid in the second default (module 6 bug, the
    nastiest so far).
16. **`PyBytes_FromStringAndSize(NULL, n)` returns UNINITIALIZED
    bytes** (`b'd\x00\x00...'` — verified). Any asm that allocates a
    scratch buffer this way must write every one of the n bytes before
    it escapes, or heap garbage leaks into output (and a length
    mismatch overflows into the neighboring heap object).
17. **id() of a bridge temporary dangles.** Every object whose address
    goes into the iface table (singletons like 'utf-8', defaults like
    `{}`) must first be held in a module global; `id({})` in the table
    constructor is the address of a dead dict by the first feed (module
    6 bug — the asm then read unrelated objects, e.g. a CDLL's dict,
    as the 'errors' kwarg dict).
18. Helper routines must preserve their argument register until the
    final `w0` exits (`mov w0, #0` *before* the comparisons returns 0
    unconditionally; `_ord_valid` bug), and dict-setstr call sites must
    not confuse the key register (x1) with the value register (x2) —
    the `final=True/False` kwarg bug loaded the bool into x1 and the
    key C-string into x2 (module 6 bugs).
19. **The return register x0 does not survive a cleanup epilogue.**
    A shared cleanup block that drops parks via `Py_DecRef` clobbers
    x0 with park pointers; ctypes' py_object restype then wraps the
    garbage. Park the result (free stack slot or callee-saved reg)
    BEFORE the decrefs and reload it at the end (module 7 bug — every
    error/early exit segfaulted; NULL returns are legal only when an
    exception is already set, so NULL-probing hid the real faults).
20. **Loading a park into a register without clearing the park is a
    double drop.** The cleanup later drops every nonzero park; the
    loaded copy and the park both "own" one ref (module 7 bug: the
    caller's chat-template LIST was freed while alive — its memory got
    reused, the list read back as empty, and everything downstream
    segfaulted nondeterministically).
21. **PyIter_Next returns NEW references.** Unlike PyDict_GetItem*
    (borrowed), each yielded item must be decref'd after use (module
    7 bug: leaked one ref per entry per call).
22. **`dd_init(ctypes.byref(iface()))` dangles if the array is a
    temporary** — the asm stores the pointer; the array is freed after
    the call; every later asm call reads freed memory. The iface array
    must live in a module global (this bit my *debugging probes*, not
    the shipped bridge — cost a full red-herring investigation).
23. Dynamic error messages: build the str (FromOrdinal/Concat), then
    `PyUnicode_AsUTF8AndSize(msg, NULL)` + `PyErr_SetString` (which
    copies); `PyErr_SetObject(ValueError, msg_str)` is WRONG — it
    wants an exception *instance*, not a message (module 7 bug:
    SystemError "not a BaseException subclass").
24. **Park slots must sit above the callee-saved save area — and
    helpers must not squat on callee-saved regs they don't save.**
    `_decode` parked its args tuple in x21 without saving it, silently
    corrupting every caller's x21 (channel in begin_body, `final` in
    feed); symptom: garbage late in ffi *after* the helper returned.
    Move helper temps into frame parks or save/restore the reg
    (module 8 bug — every `[6, MESSAGE]` feed crashed in GC).
25. **Attribute access vs method call: read the original object model,
    not the name.** `uuid.uuid4().hex` is an attribute read (UUID.hex
    is a str set in __init__), not a bound-method call; `_meth0` on it
    raises "'str' object is not callable". Same class as war story 18:
    dict `__getitem__` vs `PyObject_GetItem` (`_t_seq_getitem` on dicts
    raises "dict is not a sequence") — pick GetItemWithError for
    subscript/`.get` on mappings, and know which objects own which
    protocol slots.
26. **Returning NULL with no exception set is undefined under ctypes
    py_object.** `dict.get()`-absent → Los_drop_cfg returned `mov x0,#0`
    and segfaulted inside `_ctypes_callproc` instead of raising
    SystemError. NULL is legal ONLY when PyErr is set; for "return
    None", load the None singleton from the iface (borrowed is fine
    for a py_object restype).
27. **movz/movk immediates need arithmetic verification.** 199998 =
    0x30D3E, but the module shipped `movz #0xd9e; movk #3` = 0x30D9E =
    200094 — the control-token guard silently passed unknown markers.
    Compute the halves, don't eyeball hex; test the boundary value.
28. **Callee-saved registers used as parks must be zeroed at entry.**
    Stack parks get `stp xzr,xzr` init; register parks need the same:
    the module-9 raise path `cbz x24,3f` read a stale caller value and
    Py_DecRef'd it (crash on x0=6). Zero every owned-park register in
    the prologue, not just stack slots.
29. **PyObject_RichCompareBool op codes: Py_LT=0, Py_LE=1, Py_EQ=2,
    Py_NE=3, Py_GT=4, Py_GE=5.** Module 10 used #3 for "GE" — which is
    NE — and every gate inverted (40 >= 2**32 came out True). The small
    LE/EQ codes are adjacent to the GT/GE ones; write the enum comment
    beside every call site.
30. **Walk the original AST, not the summary.** Porting
    `heads*(splits*(dim+1)*4+dim*ob)` from memory dropped the `*4`
    (404 rows vs 118). Read the source line while writing each asm
    block and mirror the exact grouping — multiplication order also
    affects float rounding (bug-for-bug fidelity).
31. **AAPCS64: x9–x15 are CALLER-saved — only x19–x28 survive a call.**
    x10 held the iface pointer across a `bl _t_setattr_s` and came
    back as garbage (4); every earlier module reloaded the iface
    pointer per use, which is why the pattern held until one didn't.
    Never carry x9-x15 across a call, even for "one quick load".
32. **Don't multiplex a live value register as a scratch park.** The
    budget-compare parked its arithmetic total in x27 while x27 still
    held the (borrowed) slot needed by the fallthrough append — the
    `assigned` list silently accumulated running counts (values were
    plausible!). Park temps in a dedicated stack slot, and audit every
    register's full lifetime across ALL branches, not just the path
    that wrote it.
33. **Generator yields suspend mid-loop: the in-flight item survives
    the yield.** `yield` inside a for loop means resumption CONTINUES
    with the current element (batch.append(prompt) runs for the SAME
    prompt after the yield). A state-machine port must stash the
    pending pair (self.pending) at flush and process it FIRST on the
    next resumption — dropping it loses the element entirely.
    Also: generator bodies run at first next(), not at call time —
    construct lazily (it=None sentinel) so TypeErrors surface at
    iteration, exactly like the original.
34. **The result register must be excluded from the shared cleanup
    chain.** error_state parked the return value in x20 (same reg the
    error paths drop) — the generic exit decref'd and returned it;
    BaseException_dealloc crashed on first use. When a success path
    reuses a register that cleanup drops, explicitly transfer ownership
    (`mov x20, xzr` after moving to x0) before the generic exit.
35. **C-level try/except Exception: PyErr_Occurred →
    PyErr_ExceptionMatches(Exception) → PyErr_Clear.** Route every
    C-API NULL to one classifier; non-matching (BaseException)
    propagates with the exception still set. Recursion through the
    classifier preserves per-level semantics (each level owns its own
    try/except) — but note recursion now runs on the C stack
    (document the RecursionError divergence).
36. **Differential reprs leak object addresses** (`<Obj at 0x…>`) —
    two separately-constructed instances always diff. Normalize hex
    addresses (regex `0x[0-9a-f]+`) in snapshot comparators, and run
    the differential 2-3x to catch flaky passes.
37. **Class statements: `__module__` comes from the CALLER's frame.**
    The bridge's `__name__` is `<mod>_asm` — passing it straight puts
    `*_asm` into every class's `__module__` (reprs and pickles change).
    Strip the suffix (`__name__.rsplit('_asm', 1)[0]`) so classes
    carry the original module name.
38. **Generator-stamped ports need ground-truth probes FIRST.** The
    class-dict order (3.12: type_new consumes `__qualname__` from ns,
    appends `__dict__`/`__weakref__`, docless classes get `__doc__:
    None` appended LAST) and the exact metaclass-conflict message came
    from file-based dis()/repr probes of the live interpreter — memory
    of "how CPython does it" was wrong on two counts (qualname key,
    doc position). Also: when stamping N near-identical modules, build
    ONE algorithm (with a conflict-path differential) instead of N
    hand ports — and keep the generator in-repo so the shipped .s
    files are reproducible.
39. **PyUnicode_AsUTF8AndSize(u, &size) WRITES 8 bytes at *x1.** Passing
    a stale/garbage x1 (which held a live PyObject at both new call
    sites) corrupts that object's ob_refcount — symptoms surface much
    later, at interpreter shutdown (`weakref._select_for_exit`
    segfault) or as "random" refcount chaos. Always `mov x1, xzr`
    (NULL = no size out-param) before the call. (Modules 29/30 both
    had it; all committed modules audited clean — tokenizer_metadata
    got it right.)
40. **The generic exit's cleanup loads clobber x0.** reference_numerics'
    `Lrn_out` began `ldr x0, [sp, #104]` — the success result sitting
    in x0 was destroyed and NULL (no exception) returned; ctypes'
    py_object conversion of NULL then segfaults inside
    _ctypes_callproc, presenting as a "silent crash inside torch"
    because fault-PC symbolization lands in the victim frame. Fix:
    park the result in a dedicated stack slot across the decref chain
    (`str x0, [sp, #112]` … `ldr x0, [sp, #112]`), widening the frame
    if needed.
41. **Zero EVERY register park at entry, not just the ones this path
    uses.** glm_normalization zeroed x22/x23 but not x24–x26; an
    early-exit error path (getattr AttributeError) reached the
    generic cleanup, which read the callee-saved regs still holding
    CTYPES' leftovers — Py_DecRef(4), crash inside the framework.
    The cleanup chain is a garbage-collector for every register it
    names; all of them must be xzr at every entry path.
42. **`if not X.all()` compiles to `b.ne raise` on IsTrue's result**
    (0 = falsy = nonfinite = raise; 1 = truthy = fine). The natural
    "b.eq" reading inverts the gate — module 31 and (earlier) module
    30 both shipped inverted finite-gates on first build. Write the
    polarity in a comment at every IsTrue branch.
43. **PyDict_SetItem returns an INT (0 = success, -1 = failure), not
    a PyObject.** `cbz x0, error` after DictSetItem branches on the
    SUCCESS path — and the "error" cleanup then decref'd a register
    holding the RichCompareBool int 0/1, i.e. Py_DecRef(1) → segfault
    inside the framework (module 32, first build). Int-returning C-API
    calls (DictSetItem, DictSetString, SetItem, Print) test nonzero
    for error; pointer-returning calls test NULL. Also: never park an
    int result (RCB 0/1/-1, Contains) in a register a drop-chain
    decrefs — use a register outside every cleanup chain.
44. **lru_cache(maxsize=None) ports cleanly as a bridge-held dict**:
    lookup via PyDict_GetItemWithError (hit: incref for the py_object
    restype; miss: PyErr_Occurred to separate error from miss), then
    compute, then store before returning the NEW bool. Exceptions are
    never cached (error path exits before the store) — lru fidelity.
    The decorated function must ALSO resolve patchable globals per
    call: `torch.cuda.get_device_properties` is patched by tests, so
    resolve torch → .cuda → .get_device_properties via GetAttrString
    on every miss, never a bridge-held id(). Bridge exposes a
    protocol shell (cache_clear → asm PyDict_Clear) — cache_info is a
    documented divergence (unused repo-wide). Two modules with the
    same decorated helper (awq_decode_policy, q4_prefill_policy) get
    SEPARATE caches — copy the .s, don't share the dylib.
45. **Class-def modules with __init__ bodies: the method's decision
    logic ports to asm; the bridge supplies only a signature shell.**
    Pattern (glm_flash_ffn, module 34): bridge defines
    `def _flash_init(self, weights, *, activation_limit=10., **options):
    return _lib.dd_flash_ffn_init(self, weights, activation_limit,
    options)` — zero logic; ns gets this function via an iface slot
    (dict SetItem does NOT steal — pass the borrowed iface ref, no
    incref). super() fidelity: don't fake the parent lookup — call the
    real `super` builtin with (defining_class, self), getattr
    '__init__', then Call. The defining class must come from a
    bridge-held registry dict filled AFTER dd_classes returns (iface
    id() is stable across the fill); asm resolves it per call via
    GetItemWithError. kwargs merge `f(a=k, **options)`: DictNew,
    SetItem a, then PyDict_Merge(kwargs, options, override=1) —
    options wins, exactly like CPython's call-site kwargs build.
    Pitfall found: building the finfo args tuple — the incref'd
    argument was left in x0 while x0 was reloaded with the callable;
    the SetItem never ran and finfo() received an EMPTY tuple
    ("missing 1 required positional argument"). Park the tuple, set
    the item, THEN load the callable.
46. **A ref fully consumed by a helper MUST be zeroed in the caller's
    register before any shared exit label drops it again** (module 31,
    glm_normalization — the two-week "layout-dependent heap corruption"
    regression). `_allcheck(obj)` takes obj NEW and decrefs it
    internally; the caller left the pointer in x24 and every path
    through the shared epilogue drop chain decref'd it a SECOND time —
    a double-free on every successful rms_norm call. The freed block
    got handed to two owners; symptom was a RuntimeError message str
    full of interleaved foreign strings/binary (two live objects
    sharing one block), plus GC segvs at shutdown, all
    ~layout-dependent (sometimes clean, sometimes not, pure-py bridges
    still "corrupting" because the trigger was the HEAP SHAPE of the
    test run, not the trio under suspicion). Rules: (a) zero a register
    the moment its ownership leaves it (helper consumes it / value
    stolen into a tuple/dict); (b) never trust "the trio/pairs fail"
    combinatorics when the corruption source is heap-shape-dependent —
    restore EACH suspect to originals in turn instead; (c) a str whose
    data is scrambled/zeroed with a valid header is a double-free
    victim, not an AsUTF8AndSize x1 bug — read the byte pattern first
    (UTF-16-looking escapes = two str bodies interleaved).
47. **A shim must re-export EVERY module-level name of the original —
    not just the ones its own tests use** (module 34 fallout, caught at
    module 39). The glm_flash_ffn shim dropped `_limit`; four serving
    suites (glm_host_generation, cuda_glm_generation,
    glm_native_host_command, glm_flash_host_generation) import it
    transitively but were never in the battery — latent ImportError
    until the tool-chain suites ran. Related trap: `_dd_limit` was
    missing from `.globl` — a perfectly implemented asm entry point is
    unusable if the symbol is local (nm lowercase `t`). After every
    port, run an export audit: parse the ORIGINAL module's AST
    (defs/classes/imports/assigns), import the shim, diff the names —
    and grep the repo for imports of every dropped name. The audit
    script pattern lives in this file's history; 0 missing after the
    fix.

48. **x9-x15 are CALLER-SAVED — a PyObject* parked there is dead after
    ANY `bl`** (module 41, three separate segfaults, all the same
    class). The dd_classes round already fixed this for the iface
    pointer (x10 across `bl _t_fromstring`); module 41's feed/finish
    bodies re-committed it for VALUE registers: (a) `mov x2, x9; bl
    _t_setattr_s; ...; mov x0, x9; bl _t_decref` — setattr clobbers x9
    to garbage (observed: 8) and Py_DecRef(8) segfaults while setattr
    itself survived by luck; (b) a loop counter held in x22 reaching an
    error-path register sweep that treats it as a NEW ref —
    Py_DecRef(3) on the plain int; (c) `PyDict_GetItemWithError` returns
    a BORROWED ref — parking it and decrefing it later drains the class
    object's refcount across calls until the next lookup dangles.
    Rules: values that must survive a call live in callee-saved regs
    (x19-x28, x27/x28 preferred for scratch that shouldn't be swept) or
    re-parked in a stack slot and RELOADED after every bl; registers
    used for C-sized scalars (loop counters, sizes) must NEVER be in a
    function's exception-sweep register list; borrowed refs
    (GetItemWithError, GetAttrString results already consumed) get an
    explicit _incref before being parked as owned. Also from this hunt:
    svc-write markers (safe macro: save x0/x1/x2/x16, `svc #0x80`,
    restore; `.p2align 2` after inline `.ascii`) are the fastest
    fault-localizer when lldb is SIP-blocked and macOS crash-report
    dedup throttles — put a marker at every semantic step and read the
    stderr byte stream. And: `ldr xN,[xN,#0]` disassembles as a bare
    `ldr xN,[xN]` — don't misread offset-0 iface slots as missing
    offsets.
