// damendaemon/evidence_environment.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 43/370. 100% of the module logic:
//     def report_environment(environment):
//         return {name: environment[name] for name in sorted(REPORT_KEYS)
//                 if name in environment}
// REPORT_KEYS (the frozenset literal) is bridge-held data; sorted() is
// the real builtin called through the iface table.
// Fidelity notes:
//   - dict comprehension: for name in sorted(REPORT_KEYS) -> GetIter/
//     IterNext loop; membership+subscript fused into one
//     PyDict_GetItemWithError (dict __contains__ and [k] are the same
//     hash lookup — value-identical, and the filter runs before the
//     value fetch in both). DictSetItem takes the borrowed value (no
//     steal) and returns int 0/-1.
//   - PyIter_Next NULL + PyErr_Occurred split exhaustion from error.
//
// ABI table (byte offsets), installed via dd_init:
//     0 PyObject_Call     8 PyDict_New     16 PyDict_SetItem
//    24 Py_DecRef        32 PyObject_GetIter 40 PyIter_Next
//    48 PyErr_Occurred   56 sorted         64 REPORT_KEYS
//    72 PyDict_GetItemWithError  80 PyTuple_SetItem
//    88 PyTuple_New

.globl _dd_init
.globl _dd_report

.section __DATA,__data
g_iface: .quad 0

.macro IFACE off
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, \off]
    br    x9
.endm

.section __TEXT,__cstring
e_report: .asciz "report_environment"

.text
_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret
_t_call:           IFACE #0
_t_dict_new:       IFACE #8
_t_dict_setitem:   IFACE #16
_t_decref:         IFACE #24
_t_tuple_setitem:  IFACE #80
_t_getiter:        IFACE #32
_t_iternext:       IFACE #40
_t_err_occurred:   IFACE #48
_t_dict_getwe:     IFACE #72
_t_tuple_new:      IFACE #88

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_report(x0=environment BORROWED) -> dict (NEW)
// x19 env b  x21 result NEW  x22 name NEW
// parks: [96] args  [104] it NEW  [112] park
// ============================================================
_dd_report:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #104]
    stp   xzr, xzr, [sp, #112]
    mov   x19, x0
    mov   x21, xzr
    mov   x22, xzr

    // ---- args = (REPORT_KEYS,) ; sorted_keys = sorted(*args) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lrep_err
    str   x0, [sp, #96]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #64]                   // REPORT_KEYS
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #64]                   // REPORT_KEYS
    bl    _t_tuple_setitem
    // (helper defined below — alias to slot 24)
    ldr   x0, [x10, #56]                   // sorted
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                   // park across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #112]
    cbz   x0, Lrep_err
    str   x0, [sp, #112]                   // sorted_keys stays parked

    // ---- result = {} ----
    bl    _t_dict_new
    cbz   x0, Lrep_drop112_err
    mov   x21, x0                          // result NEW

    // ---- it = iter(sorted_keys) ----
    ldr   x0, [sp, #112]
    bl    _t_getiter
    cbz   x0, Lrep_drop112_err
    str   x0, [sp, #104]                   // it NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

Lrep_loop:
    ldr   x0, [sp, #104]
    bl    _t_iternext
    cbz   x0, Lrep_iter_end
    mov   x22, x0                          // name NEW
    // v = environment[name] (fused contains+get; NULL = missing or error)
    mov   x0, x19
    mov   x1, x22
    bl    _t_dict_getwe
    cbnz  x0, Lrep_have
    bl    _t_err_occurred
    cbnz  x0, Lrep_drop_all_err             // real error -> propagate
    // missing key: skip (comprehension filter)
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    b     Lrep_loop
Lrep_have:
    // v (borrowed) is in x0; move to callee-saved x23 for the store
    mov   x23, x0                          // v borrowed
    // result[name] = v  (DictSetItem returns int 0/-1)
    mov   x0, x21
    mov   x1, x22
    mov   x2, x23
    bl    _t_dict_setitem
    cmn   w0, #1
    b.eq  Lrep_drop_all_err
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    b     Lrep_loop

Lrep_iter_end:
    // iterator exhausted: IterNext NULL with no error pending
    bl    _t_err_occurred
    cbnz  x0, Lrep_drop_all_err             // paranoid: error at exhaustion
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x21
    b     Lrep_out

Lrep_drop_all_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #104]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #104]
2:  cbz   x22, 3f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
3:  cbz   x21, 4f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
4:  mov   x0, xzr                           // NULL + pending exception
    b     Lrep_out

Lrep_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, Lrep_err
    bl    _t_decref
    str   xzr, [sp, #112]
    b     Lrep_err
Lrep_err:
    mov   x0, xzr
    b     Lrep_out
Lrep_out:
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
