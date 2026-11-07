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

Ported: 504 LoC across 14 modules (1.14%). Remaining: 356 modules / 43,983 lines.
At the current module size (41 LoC median-of-smallest), the remaining
work is left as an exercise for future generations.

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
