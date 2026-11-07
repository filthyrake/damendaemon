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
| damendaemon (whole package) | 44,201 | **0.25% ported** |
| stopping.py | 41 | ✅ PORTED (module 1) |
| tokenizer_decoders.py | 19 | ✅ PORTED (module 2) |
| prompts.py | 51 | ✅ PORTED (module 3) |

Ported surface: 370 modules. Remaining: 367 modules / 44,090 lines.
At the current module size (41 LoC median-of-smallest), the remaining
work is left as an exercise for future generations.

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
