// damendaemon/error_state.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 14/370. 100% of detached_error:
//     try:
//         if isinstance(error, BaseExceptionGroup):
//             detached = error.derive([detached_error(c) for c in error.exceptions])
//         else:
//             detached = copy.copy(error)
//         detached = detached.with_traceback(None)
//     except Exception:
//         detached = RuntimeError(str(error))
//     detached.__cause__ = detached.__context__ = None
//     detached.__notes__ = list(getattr(error, '__notes__', ()))
//     return detached
// The try/except Exception is realized with PyErr_Occurred +
// PyErr_ExceptionMatches(Exception) + PyErr_Clear; non-Exception
// BaseExceptions propagate exactly like the original. The
// comprehension is a raw PyList_Append loop; recursion is a direct
// `bl _dd_detached_error` (each level owns its own try/except, so a
// child only ever propagates a non-Exception BaseException).
// Divergence note (documented, untested by the suite): recursion here
// runs on the C stack, so absurdly-nested groups (>~1000 deep) do not
// raise RecursionError like the original would.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString        8 SetAttrString    16 PyObject_Call
//    24 PyTuple_New         32 PyTuple_SetItem 40 Py_DecRef
//    48 PyObject_IsInstance 56 PyObject_Size    64 PySequence_GetItem
//    72 PyList_New          80 PyList_Append   88 PyErr_Occurred
//    96 PyErr_ExceptionMatches 104 PyErr_Clear  112 copy.copy (held)
//   120 BaseExceptionGroup (held) 128 RuntimeError (held)
//   136 Exception (held)   144 str (held)      152 None
//   160 list (held)

.globl _dd_init
.globl _dd_detached_error

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_exceptions: .asciz "exceptions"
a_derive:      .asciz "derive"
a_with_tb:     .asciz "with_traceback"
a_notes:       .asciz "__notes__"
a_cause:       .asciz "__cause__"
a_context:     .asciz "__context__"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines
_t_getattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #0]
    br    x9
_t_setattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_call:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_isinstance:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_seq_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_list_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_list_append:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_err_occurred:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_err_matches:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #96]
    br    x9
_t_err_clear:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #104]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_detached_error(x0 = error) -> NEW detached snapshot
// Register plan:
//   x19 error (borrowed)      x20 detached NEW     x21 excs NEW
//   x22 lst NEW               x23 method NEW       x24 n (raw)
//   x25 i (raw)               x26 short temp NEW
// Frame parks: [96] args tuple  [104] temp obj  [112] temp obj
// ============================================================
_dd_detached_error:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    mov   x19, x0
    mov   x20, xzr                          // detached park
    mov   x21, xzr                          // excs park
    mov   x22, xzr                          // lst park
    mov   x23, xzr                          // method park

    // ---- isinstance(error, BaseExceptionGroup) ----
    mov   x0, x19
    adrp  x9, g_iface@PAGE
    ldr   x1, [x9, g_iface@PAGEOFF]
    ldr   x1, [x1, #120]                    // BaseExceptionGroup
    bl    _t_isinstance
    cmn   w0, #1
    b.eq  Les_raise
    cbz   w0, Les_copy

// ---- group path: excs = error.exceptions; lst = [] ----
    mov   x0, x19
    adrp  x1, a_exceptions@PAGE
    add   x1, x1, a_exceptions@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Les_raise
    mov   x21, x0                           // excs NEW
    mov   x0, x21
    bl    _t_size
    cmp   x0, #0
    b.lt  Les_raise_drop21
    mov   x24, x0                           // n
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Les_raise_drop21
    mov   x22, x0                           // lst NEW
    mov   x25, #0                           // i = 0
Les_loop:
    cmp   x25, x24
    b.ge  Les_loop_done
    mov   x0, x21
    mov   x1, x25
    bl    _t_seq_getitem
    cbz   x0, Les_raise_drop2122
    mov   x26, x0                           // child NEW
    mov   x0, x26
    bl    _dd_detached_error                // RECURSION
    mov   x27, x0                           // dchild NEW
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    cbz   x27, Les_raise_drop2122           // non-Exception propagated
    mov   x0, x22
    mov   x1, x27
    bl    _t_list_append                    // append increfs
    mov   w28, w0
    mov   x0, x27
    bl    _t_decref
    cmn   w28, #1
    b.eq  Les_raise_drop2122
    add   x25, x25, #1
    b     Les_loop
Les_loop_done:
    // ---- derive_m = error.derive; detached = derive_m(lst) ----
    mov   x0, x19
    adrp  x1, a_derive@PAGE
    add   x1, x1, a_derive@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Les_raise_drop2122
    mov   x23, x0                           // derive method NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_raise_drop2123
    str   x0, [sp, #96]                     // args NEW
    mov   x0, x22
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    mov   x0, x23
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                           // detached NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x26, Les_raise_drop2123
    mov   x20, x26
    mov   x26, xzr
    // drop derive method + lst + excs
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    b     Les_with_tb

// ---- non-group path: detached = copy.copy(error) ----
Les_copy:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_raise
    str   x0, [sp, #96]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]                   // copy.copy
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x20, x0                           // detached NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x20, Les_raise

// ---- detached = detached.with_traceback(None) ----
Les_with_tb:
    mov   x0, x20
    adrp  x1, a_with_tb@PAGE
    add   x1, x1, a_with_tb@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Les_raise                      // AttributeError -> except
    mov   x23, x0                           // wt NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_raise_drop23
    str   x0, [sp, #96]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #152]                   // None
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]                   // None
    bl    _t_tuple_setitem
    mov   x0, x23
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                           // detached2 NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x26, Les_raise_drop23
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x20
    bl    _t_decref
    mov   x20, x26
    mov   x26, xzr
    b     Les_try_done

// ================= try succeeded: cause/context/notes, return =================
Les_try_done:
    // ---- detached.__cause__ = None ----
    mov   x0, x20
    adrp  x1, a_cause@PAGE
    add   x1, x1, a_cause@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]                   // None (borrowed)
    bl    _t_setattr_s
    cbnz  w0, Les_prop_drop20
    // ---- detached.__context__ = None ----
    mov   x0, x20
    adrp  x1, a_context@PAGE
    add   x1, x1, a_context@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]
    bl    _t_setattr_s
    cbnz  w0, Les_prop_drop20
    b     Les_notes

// ================= except path =================
// Drop owned temps, then classify the pending exception.
Les_raise_drop2123:
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Les_raise_drop2122:
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Les_raise_drop21:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Les_raise_drop23:
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Les_raise:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    // ---- pending exception must be an Exception, else propagate ----
    bl    _t_err_occurred
    cbz   x0, Les_propagate                 // defensive: no error set
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // Exception
    bl    _t_err_matches
    cbz   w0, Les_propagate                 // BaseException -> re-raise
    bl    _t_err_clear
    // ---- s = str(error) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_propagate
    str   x0, [sp, #96]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                   // str
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                           // s NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x26, Les_propagate
    // ---- detached = RuntimeError(s) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_prop_drop26
    str   x0, [sp, #96]
    mov   x0, x26
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x26
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // RuntimeError
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                           // detached NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    cbz   x27, Les_propagate
    mov   x20, x27
    mov   x27, xzr
    // ---- __cause__ / __context__ = None on the replacement ----
    mov   x0, x20
    adrp  x1, a_cause@PAGE
    add   x1, x1, a_cause@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]
    bl    _t_setattr_s
    cbnz  w0, Les_prop_drop20
    mov   x0, x20
    adrp  x1, a_context@PAGE
    add   x1, x1, a_context@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]
    bl    _t_setattr_s
    cbnz  w0, Les_prop_drop20

// ================= __notes__ = list(getattr(error, '__notes__', ())) =================
Les_notes:
    mov   x0, x19
    adrp  x1, a_notes@PAGE
    add   x1, x1, a_notes@PAGEOFF
    bl    _t_getattr_s
    str   x0, [sp, #104]                     // notes NEW or NULL
    cbnz  x0, Les_notes_have
    // missing -> PyErr_Clear; result = [] (list(()) == [])
    bl    _t_err_clear
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Les_prop_drop20
    str   x0, [sp, #112]                     // notes_list NEW
    b     Les_notes_set
Les_notes_have:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Les_prop_drop2104
    str   x0, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    ldr   x2, [sp, #104]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // list
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                     // notes_list NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Les_prop_drop20
Les_notes_set:
    // ---- detached.__notes__ = notes_list ----
    mov   x0, x20
    adrp  x1, a_notes@PAGE
    add   x1, x1, a_notes@PAGEOFF
    ldr   x2, [sp, #112]
    bl    _t_setattr_s
    mov   w28, w0
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    cbnz  w28, Les_prop_drop20
    ldr   x0, [sp, #104]
    cbz   x0, Les_out20
    bl    _t_decref
    str   xzr, [sp, #104]
    b     Les_out20
Les_prop_drop2104:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    b     Les_prop_drop20

// ---- return detached (x20 owned) ----
Les_out20:
    mov   x0, x20                           // result NEW
    mov   x20, xzr                          // ownership moves to the x28 park
    b     Les_out

// ---- propagate: exception already set (or defensive NULL) ----
Les_prop_drop26:
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Les_prop_drop20:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Les_propagate:
    mov   x0, xzr                            // NULL, exception set

// ---- common exit: x0 = result ----
Les_out:
    mov   x28, x0                          // park result across decrefs
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #104]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #104]
2:  ldr   x0, [sp, #96]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #96]
3:  cbz   x26, 4f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
4:  cbz   x23, 5f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
5:  cbz   x22, 6f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
6:  cbz   x21, 7f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
7:  cbz   x20, 8f
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
8:  mov   x0, x28

    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

