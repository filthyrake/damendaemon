// damendaemon/scoring.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 12/370. 100% of options:
//     prompt = validate_raw(prompt, max_length=max_length)
//     if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 256
//             or any(type(token) is not int or token < 0 for token in candidates)
//             or len(set(candidates)) != len(candidates)):
//         raise ValueError('candidate_token_ids must contain 2 to 256 '
//                          'distinct nonnegative integers')
//     return prompt, list(candidates)
// The any() generator is realized as an indexed scan with the same
// per-token checks and short-circuit; set-size equality is a direct
// ssize compare (int elements cannot mutate the list mid-scan).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString        8 PyObject_Call    16 PyObject_IsInstance
//    24 PyList_Type         32 PyLong_Type     40 PyObject_Size
//    48 PyObject_RichCompareBool  56 PySet_New  64 Py_DecRef
//    72 PyTuple_New         80 PyTuple_SetItem 88 PyDict_New
//    96 PyDict_SetItem     104 PyUnicode_FromString 112 PyLong_FromLongLong
//   120 PyErr_SetString    128 ValueError      136 PySequence_GetItem
//   144 validate_raw callable (bridge-held)

.globl _dd_init
.globl _dd_options

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
k_maxlen: .asciz "max_length"
m_cand:   .asciz "candidate_token_ids must contain 2 to 256 distinct nonnegative integers"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines
_t_call:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_isinstance:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_rcb:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_set_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_dict_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_dict_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #96]
    br    x9
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #104]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #112]
    br    x9
_t_seterr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #120]
    br    x9
_t_seq_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #136]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_options(x0 = prompt, x1 = candidates, x2 = max_length)
// Register plan: x24 prompt  x25 candidates  x26 max_length (borrowed)
//   x19 prompt2 NEW   x20 zero NEW    x21 set NEW
//   x22 newlist NEW   x23 result NEW   x27 temp
// Frame parks (above the save area): [96] args  [104] kwargs
// ============================================================
_dd_options:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]               // args / kwargs parks
    mov   x24, x0                           // prompt (borrowed)
    mov   x25, x1                           // candidates (borrowed)
    mov   x26, x2                           // max_length (borrowed)
    mov   x19, xzr                          // prompt2 park
    mov   x20, xzr                          // zero park
    mov   x21, xzr                          // set park
    mov   x22, xzr                          // newlist park

    // ---- kwargs = {'max_length': max_length} ----
    bl    _t_dict_new
    cbz   x0, Lsc_err
    str   x0, [sp, #104]
    adrp  x0, k_maxlen@PAGE
    add   x0, x0, k_maxlen@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lsc_err
    mov   x27, x0
    ldr   x0, [sp, #104]
    mov   x1, x27
    mov   x2, x26
    bl    _t_dict_setitem
    mov   x0, x27
    bl    _t_decref

    // ---- args = (prompt,) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsc_err
    str   x0, [sp, #96]
    mov   x0, x24
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x24
    bl    _t_tuple_setitem

    // ---- prompt2 = validate_raw(prompt, max_length=max_length) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                   // validate_raw
    ldr   x1, [sp, #96]
    ldr   x2, [sp, #104]
    bl    _t_call
    cbz   x0, Lsc_err
    mov   x19, x0                          // prompt2 NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]

    // ---- not isinstance(candidates, list) ----
    mov   x0, x25
    adrp  x9, g_iface@PAGE
    ldr   x1, [x9, g_iface@PAGEOFF]
    ldr   x1, [x1, #24]                     // PyList_Type
    bl    _t_isinstance
    cmn   w0, #1
    b.eq  Lsc_err
    cmp   w0, #1
    b.ne  Lsc_raise

    // ---- n = len(candidates); not 2 <= n <= 256 ----
    mov   x0, x25
    bl    _t_size
    cmp   x0, #0
    b.lt  Lsc_err
    cmp   x0, #2
    b.lt  Lsc_raise
    cmp   x0, #256
    b.gt  Lsc_raise
    mov   x28, x0                          // n (int, no refs)

    // ---- zero = long(0) ----
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lsc_err
    mov   x20, x0

    // ---- any(type(token) is not int or token < 0 ...) ----
    mov   x27, xzr                          // i
Lscan_loop:
    cmp   x27, x28
    b.ge  Lscan_done
    mov   x0, x25
    mov   x1, x27
    bl    _t_seq_getitem
    cbz   x0, Lsc_err
    str   x0, [sp, #96]                     // token NEW (borrow slot)
    ldr   x9, [sp, #96]
    ldr   x9, [x9, #8]                      // ob_type
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #32]                   // PyLong_Type
    cmp   x9, x10
    b.ne  1f
    // token < 0 ?
    ldr   x0, [sp, #96]
    mov   x1, x20
    mov   x2, #0                           // Py_LT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lsc_err_drop_tok
    cmp   w0, #1
    b.eq  Lsc_err_drop_tok_raise
    b     2f
1:  // type(token) is not int -> raise
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lsc_raise
2:  ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    add   x27, x27, #1
    b     Lscan_loop
Lscan_done:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr

    // ---- len(set(candidates)) != n ----
    mov   x0, x25
    bl    _t_set_new
    cbz   x0, Lsc_err
    mov   x21, x0                          // set NEW
    mov   x0, x21
    bl    _t_size
    cmp   x0, #0
    b.lt  Lsc_err
    cmp   x0, x28
    b.ne  Lsc_raise

    // ---- newlist = list(candidates) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsc_err
    str   x0, [sp, #96]
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x25
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #24]                    // PyList_Type
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                          // newlist NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x22, Lsc_err

    // ---- result = (prompt2, newlist) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lsc_err
    mov   x23, x0                          // result NEW
    mov   x0, x19
    bl    _incref
    mov   x0, x23
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    mov   x0, x22
    bl    _incref
    mov   x0, x23
    mov   x1, #1
    mov   x2, x22
    bl    _t_tuple_setitem
    // drop owned temps; result keeps them
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x23
    b     Lsc_out

// ---- token<0 raise (drop the parked token first) ----
Lsc_err_drop_tok_raise:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lsc_raise

Lsc_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // ValueError
    adrp  x1, m_cand@PAGE
    add   x1, x1, m_cand@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lsc_cleanup

// ---- error path: exception set; drop parks ----
Lsc_err_drop_tok:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
Lsc_err:
    mov   x0, xzr

Lsc_cleanup:
    mov   x28, x0                          // park result across decrefs
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #96]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #96]
2:  cbz   x22, 3f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
3:  cbz   x21, 4f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
4:  cbz   x20, 5f
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
5:  cbz   x19, 6f
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
6:  mov   x0, x28

Lsc_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret
