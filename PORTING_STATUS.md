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

Ported: 304 LoC across 7 modules (0.69%). Remaining: 363 modules / 44,045 lines.
At the current module size (41 LoC median-of-smallest), the remaining
work is left as an exercise for future generations.

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
