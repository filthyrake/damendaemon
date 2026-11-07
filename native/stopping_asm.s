// damendaemon/stopping.py — hand-ported to arm64 assembly.
// Part of the asm-rewrite branch: the goal is the entire package in asm.
// This file: 100% of stopping.py's logic (stop_sequences + text_delta),
// no C shim, no logic in the Python loader. arm64 (Apple silicon), AAPCS64.
//
// CPython layout facts used (frozen for many versions):
//   PyObject    : ob_refcnt@0, ob_type@8                       (16 bytes)
//   PyTupleObject: header 24, then PyObject *ob_item[]        (item i at 24+8i)
//   PyBytesObject: ob_size@16 (length), ob_sval@32 (data, NUL-terminated)
// Everything else goes through the stable CPython C API.

.globl _dd_text_delta
.globl _dd_stop_sequences

// replacement char U+FFFD, UTF-8 EF BF BD, referenced by adrp+add
.section __TEXT,__cstring
c_utf8:   .asciz "stop strings must be valid UTF-8"
c_invalid:.asciz "stop must contain one to four nonempty strings of at most 256 characters"
c_revised:.asciz "tokenizer revised already streamed text"

// GOT-style slots are dead: eager flat-namespace binds cannot see the
// Python framework dylib from another dlopen'd image. Instead the
// bridge resolves every function/object address at import time and
// passes a pointer table through _dd_init (explicit ABI, no dlsym from
// the asm side at all).
.section __DATA,__data
g_iface: .quad 0

.text

// void _dd_init(x0 = interface table) — install and forget.
.globl _dd_init
_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines into the interface table
_dd_call_tuple_new:                  // iface[16]
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_dd_call_uni_length:                 // iface[24]
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_dd_call_utf8string:                  // iface[32]
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_dd_call_decref:                      // iface[40]
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_dd_call_err_clear:                   // iface[48]
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9

// ============================================================
// private helpers (AAPCS64, only x0-x15 clobbered)
// ============================================================

// _dd_tail_rep(ptr x0, len x1) -> w0: buf ends with UTF-8 EF BF BD (U+FFFD)?
_dd_tail_rep:
    cmp   x1, #3
    b.lo  9f
    sub   x9, x1, #3
    ldrb  w10, [x0, x9]
    add   x9, x9, #1
    ldrb  w11, [x0, x9]
    add   x9, x9, #1
    ldrb  w12, [x0, x9]
    cmp   w10, #0xEF
    cset  w10, eq
    cmp   w11, #0xBF
    cset  w11, eq
    cmp   w12, #0xBD
    cset  w12, eq
    and   w0, w10, w11
    and   w0, w0, w12
    ret
9:  mov   w0, #0
    ret

// _dd_find(hay x0, hlen x1, nee x2, nlen x3) -> x0: offset of needle or -1
_dd_find:
    stp   x29, x30, [sp, #-32]!
    mov   x29, sp
    str   x0, [sp, #16]                // remember haystack base
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #0]                // memmem
    blr   x9
    cmp   x0, #0
    b.eq  1f
    ldr   x9, [sp, #16]
    sub   x0, x0, x9
    b     2f
1:  mov   x0, #-1
2:  ldp   x29, x30, [sp], #32
    ret

// _dd_endswith(hay x0, hlen x1, nee x2, nlen x3) -> w0: hay endswith nee
_dd_endswith:
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    cmp   x3, x1
    b.hi  1f
    sub   x9, x1, x3
    add   x0, x0, x9                  // hay + hlen - nlen
    mov   x1, x2                      // nee ptr
    mov   x2, x3                      // nlen
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]                // memcmp
    blr   x9
    cmp   w0, #0
    cset  w0, eq
    ldp   x29, x30, [sp], #16
    ret
1:  mov   w0, #0
    ldp   x29, x30, [sp], #16
    ret

// _dd_raise_value(x0 = C-string message) -> ValueError raised; returns
_dd_raise_value:
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #64]               // ValueError object
    ldr   x9, [x9, #56]               // PyErr_SetString
    blr   x9
    ldp   x29, x30, [sp], #16
    ret

// ============================================================
// int _dd_text_delta(
//   x0 = decoded (PyBytes), x1 = emitted (PyBytes),
//   x2 = stops (PyTuple[PyBytes]), w3 = final (bool),
//   x4 = out (int64[5]: delta_ptr, delta_len, visible_ptr,
//                                visible_len, stopped)
// ) -> 0 on success, -1 with RuntimeError set on revision
// 100% of stopping.text_delta's decision logic, in assembly.
// UTF-8 simplification vs the Python original: byte-wise scans
// instead of code-point-wise (documented in PORTING_STATUS.md).
// ============================================================
_dd_text_delta:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    // locals: [x29,#88] loop i, [x29,#96] final, [x29,#104] stopped,
    //         [x29,#112] hold-loop count (survives calls)
    // x19 d_ptr   x20 d_len   x21 e_ptr   x22 e_len
    // x23 s_ptr   x24 s_len   x25 stops   x26 out
    // x27 min_match   x28 phase register (i, then held)
    add   x19, x0, #32               // d data (inline ob_sval)
    ldr   x20, [x0, #16]             // d len (ob_size)
    add   x21, x1, #32               // e data (inline ob_sval)
    ldr   x22, [x1, #16]             // e len
    mov   x25, x2
    mov   x26, x4
    str   w3, [x29, #96]            // final
    str   xzr, [x29, #104]          // stopped = 0

    // ---- stable = final ? decoded : decoded.rstrip(U+FFFD) ----
    mov   x23, x19
    mov   x24, x20
    ldr   w8, [x29, #96]
    cbnz  w8, Lstable_done
Lrstrip:
    mov   x0, x23
    mov   x1, x24
    bl    _dd_tail_rep
    cbz   w0, Lstable_done
    sub   x24, x24, #3
    b     Lrstrip
Lstable_done:

    // ---- earliest stop match over the stable prefix ----
    // x27 = min_match, -1 = none yet
    mov   x27, #-1
    str   xzr, [x29, #88]           // i = 0
Lmatch_loop:
    ldr   x9, [x25, #16]           // n
    ldr   x10, [x29, #88]          // i
    cmp   x10, x9
    b.hs  Lmatch_done
    add   x11, x25, #24            // ob_item[]
    ldr   x12, [x11, x10, lsl #3]  // stop bytes object
    mov   x0, x23                  // hay = stable
    mov   x1, x24
    add   x2, x12, #32               // needle data (inline ob_sval)
    ldr   x3, [x12, #16]             // needle len
    bl    _dd_find                   // x0 = offset or -1
    cmp   x0, #0
    b.lt  Lmatch_next
    mov   x9, #1
    str   x9, [x29, #104]          // stopped = 1
    cmp   x27, #0
    b.lt  Lmatch_set
    cmp   x0, x27
    b.hs  Lmatch_next
Lmatch_set:
    mov   x27, x0
Lmatch_next:
    ldr   x10, [x29, #88]
    add   x10, x10, #1
    str   x10, [x29, #88]
    b     Lmatch_loop
Lmatch_done:
    // ---- dispatch ----
    ldr   x9, [x29, #104]          // stopped
    cbz   x9, Lno_stop
    mov   x3, x19                  // visible = decoded[:min_match]
    mov   x4, x27
    b     Lemit
Lno_stop:
    ldr   w8, [x29, #96]           // final
    cbnz  w8, Lhold
    mov   x0, x19
    mov   x1, x20
    bl    _dd_tail_rep             // trailing U+FFFD may still vanish
    cbz   w0, Lhold
    str   x21, [x26]               // delta: ptr borrowed, len 0
    str   xzr, [x26, #8]
    str   x21, [x26, #16]          // visible = emitted
    str   x22, [x26, #24]
    str   xzr, [x26, #32]          // stopped = 0
    mov   w0, #0
    b     Lret
Lhold:
    // ---- held: longest suffix of decoded that is a proper stop prefix ----
    mov   x28, #0                  // held = 0
    ldr   w8, [x29, #96]           // final skips the hold entirely
    cbnz  w8, Lhold_done
    str   xzr, [x29, #88]          // i = 0
Lhold_loop:
    ldr   x9, [x25, #16]           // n
    ldr   x10, [x29, #88]          // i
    cmp   x10, x9
    b.hs  Lhold_done
    add   x11, x25, #24
    ldr   x12, [x11, x10, lsl #3]  // stop bytes object
    ldr   x13, [x12, #16]            // tn
    add   x14, x12, #32              // tp (inline ob_sval)
    sub   x13, x13, #1             // count = tn - 1
    cmp   x13, x20                 // count = min(count, d_len)
    b.ls  1f
    mov   x13, x20
1:
Lhold_c:
    cmp   x13, x28                 // c <= held -> next stop
    b.ls  Lhold_next
    str   x13, [x29, #112]         // count must survive the call
    mov   x0, x19
    mov   x1, x20
    mov   x2, x14
    mov   x3, x13
    bl    _dd_endswith             // decoded endswith stop[:c]?
    ldr   x13, [x29, #112]
    cbz   w0, 2f
    mov   x28, x13                 // held = c
    b     Lhold_next
2:  sub   x13, x13, #1
    b     Lhold_c
Lhold_next:
    ldr   x10, [x29, #88]
    add   x10, x10, #1
    str   x10, [x29, #88]
    b     Lhold_loop
Lhold_done:
    mov   x3, x19                  // visible = decoded[:-held]
    sub   x4, x20, x28
Lemit:
    // ---- revision check + delta extraction ----
    str   x3, [x26, #16]           // visible ptr
    str   x4, [x26, #24]           // visible len
    cmp   x22, x4                  // len(emitted) > len(visible)
    b.hi  Lrevised
    cbz   x22, Lrev_ok             // startswith('') is always true
    mov   x0, x3
    mov   x1, x21
    mov   x2, x22
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]                // memcmp
    blr   x9
    cbnz  w0, Lrevised
Lrev_ok:
    ldr   x3, [x26, #16]           // reload: x3/x4 are caller-saved
    ldr   x4, [x26, #24]
    add   x9, x3, x22
    str   x9, [x26]                // delta ptr = visible + len(emitted)
    sub   x9, x4, x22
    str   x9, [x26, #8]            // delta len
    ldr   x9, [x29, #104]
    str   x9, [x26, #32]           // stopped
    mov   w0, #0
    b     Lret
Lrevised:
    // signal only; the bridge raises the Python RuntimeError
    mov   w0, #-1
Lret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_stop_sequences(x0 = value) -> new tuple or NULL
// Faithful port of stopping.stop_sequences. Exact type checks
// (str/list), 1..4 entries, 1..256 chars each, UTF-8 encodable.
// ============================================================
_dd_stop_sequences:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 value   x20 n   x22 result tuple   [x29,#48] loop i
    mov   x19, x0
    // None -> ()
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #72]                   // the None singleton
    cmp   x0, x10
    b.ne  1f
    mov   x0, #0
    bl    _dd_call_tuple_new
    b     Lss_ret
1:
    // str -> 1-tuple holding it
    ldr   x9, [x0, #8]
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #80]                   // &PyUnicode_Type
    cmp   x9, x10
    b.ne  2f
    mov   x0, #1
    bl    _dd_call_tuple_new
    cbz   x0, Lss_ret
    mov   x22, x0
    ldr   x9, [x19]                 // Py_INCREF (GIL held)
    add   x9, x9, #1
    str   x9, [x19]
    str   x19, [x22, #24]           // tuple[0] = value
    mov   x20, #1
    b     Lss_validate
2:
    // list -> fresh tuple of its items
    ldr   x9, [x19, #8]
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #88]                   // &PyList_Type
    cmp   x9, x10
    b.ne  Lss_invalid
    ldr   x20, [x19, #16]           // n
    cmp   x20, #1
    b.lt  Lss_invalid
    cmp   x20, #4
    b.gt  Lss_invalid
    mov   x0, x20
    bl    _dd_call_tuple_new
    cbz   x0, Lss_ret
    mov   x22, x0
    str   xzr, [x29, #48]           // i = 0
Lss_copy:
    ldr   x10, [x29, #48]
    cmp   x10, x20
    b.hs  Lss_validate
    ldr   x11, [x19, #24]           // list ob_item
    ldr   x12, [x11, x10, lsl #3]
    ldr   x13, [x12]                // incref item
    add   x13, x13, #1
    str   x13, [x12]
    add   x14, x22, #24
    str   x12, [x14, x10, lsl #3]   // tuple[i] = item
    add   x10, x10, #1
    str   x10, [x29, #48]
    b     Lss_copy
Lss_validate:
    str   xzr, [x29, #48]           // i = 0
Lss_val_loop:
    ldr   x10, [x29, #48]
    cmp   x10, x20
    b.hs  Lss_val_ok
    add   x11, x22, #24
    ldr   x21, [x11, x10, lsl #3]  // item (x21 survives the C calls)
    ldr   x9, [x21, #8]            // item type
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #80]                   // &PyUnicode_Type
    cmp   x9, x10
    b.ne  Lss_invalid
    mov   x0, x21
    bl    _dd_call_uni_length        // char count
    cmp   x0, #1
    b.lt  Lss_invalid
    cmp   x0, #256
    b.gt  Lss_invalid
    mov   x0, x21
    bl    _dd_call_utf8string        // NULL => UnicodeEncodeError
    cbz   x0, Lss_utf8_err
    bl    _dd_call_decref            // drop the validation bytes
    ldr   x10, [x29, #48]
    add   x10, x10, #1
    str   x10, [x29, #48]
    b     Lss_val_loop
Lss_val_ok:
    mov   x0, x22
    b     Lss_ret
Lss_utf8_err:
    bl    _dd_call_err_clear
    adrp  x0, c_utf8@PAGE
    add   x0, x0, c_utf8@PAGEOFF
    bl    _dd_raise_value
    mov   x0, #0
    b     Lss_ret
Lss_invalid:
    adrp  x0, c_invalid@PAGE
    add   x0, x0, c_invalid@PAGEOFF
    bl    _dd_raise_value
    mov   x0, #0
Lss_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret

