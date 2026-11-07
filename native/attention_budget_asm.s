// damendaemon/attention_budget.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 9/370. 100% of query_rows:
//     if type(scratch_bytes) is not int or scratch_bytes <= 0:
//         raise ValueError('attention scratch budget must be a positive byte count')
//     per_row = heads*(splits*(dim+1)*4 + dim*output_bytes)
//     if scratch_bytes < per_row:
//         raise ValueError('attention scratch budget cannot fit one query row')
//     return scratch_bytes // per_row
// All arithmetic goes through PyNumber_* (arbitrary precision — no int64
// boundary); the ZeroDivisionError for per_row==0 comes straight from
// PyNumber_FloorDivide with CPython's exact message.
//
// ABI table (byte offsets), installed via dd_init:
//     0 PyLong_Type              8 PyLong_FromLongLong
//    16 PyNumber_Add            24 PyNumber_Multiply
//    32 PyNumber_FloorDivide     40 PyObject_RichCompareBool
//    48 Py_DecRef               56 ValueError
//    64 PyErr_SetString

.globl _dd_init
.globl _dd_query_rows

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
m_positive: .asciz "attention scratch budget must be a positive byte count"
m_fit:      .asciz "attention scratch budget cannot fit one query row"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_num_add:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_num_mul:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_floordiv:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_rcb:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_seterr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_query_rows(x0 = heads, x1 = dim, x2 = splits,
//                          x3 = scratch_bytes, x4 = output_bytes)
// Register plan: x19 heads  x20 dim  x21 splits  x22 output_bytes
//                x23 scratch  x24 per_row  x25 temp NEW  x26 zero NEW
// Frame: [0] x29/x30, [16..80] saved x19-x26, [88] spare,
//        [96] one NEW, [104..] free (parks above save area).
// ============================================================
_dd_query_rows:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]               // one / u parks
    mov   x19, x0                           // heads (borrowed)
    mov   x20, x1                           // dim (borrowed)
    mov   x21, x2                           // splits (borrowed)
    mov   x22, x4                           // output_bytes (borrowed)
    mov   x23, x3                           // scratch_bytes (borrowed)
    mov   x24, xzr                          // per_row park (not yet owned)
    mov   x25, xzr                          // temp park (not yet owned)
    mov   x26, xzr                          // zero park (not yet owned)

    // ---- type(scratch_bytes) is not int -> raise ----
    ldr   x9, [x23, #8]                     // ob_type
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #0]                    // PyLong_Type
    cmp   x9, x10
    b.ne  Lab_raise_positive

    // ---- zero = long(0) ----
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lab_err
    mov   x26, x0                          // zero NEW

    // ---- scratch_bytes <= 0 -> raise ----
    mov   x0, x23
    mov   x1, x26
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lab_err
    cmp   w0, #1
    b.eq  Lab_raise_positive

    // ---- one = long(1) ----
    mov   x0, #1
    bl    _t_fromlong
    cbz   x0, Lab_err
    str   x0, [sp, #96]                     // one NEW

    // ---- t = dim + one ----
    mov   x0, x20
    ldr   x1, [sp, #96]
    bl    _t_num_add
    cbz   x0, Lab_err
    mov   x25, x0                          // t NEW

    // ---- t = splits * t ----
    mov   x0, x21
    mov   x1, x25
    bl    _t_num_mul
    mov   x27, x0
    mov   x0, x25
    bl    _t_decref
    cbz   x27, Lab_err
    mov   x25, x27                         // t = splits*(dim+1) NEW

    // ---- four = long(4); t = t * four ----
    mov   x0, #4
    bl    _t_fromlong
    cbz   x0, Lab_err
    str   x0, [sp, #104]                    // four NEW
    mov   x0, x25
    ldr   x1, [sp, #104]
    bl    _t_num_mul
    mov   x27, x0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x25
    bl    _t_decref
    cbz   x27, Lab_err
    mov   x25, x27                         // t = splits*(dim+1)*4 NEW

    // ---- u = dim * output_bytes ----
    mov   x0, x20
    mov   x1, x22
    bl    _t_num_mul
    cbz   x0, Lab_err
    str   x0, [sp, #104]                    // u NEW

    // ---- t = t + u ----
    mov   x0, x25
    ldr   x1, [sp, #104]
    bl    _t_num_add
    mov   x27, x0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x25
    bl    _t_decref
    cbz   x27, Lab_err
    mov   x25, x27                         // t = splits*(dim+1)+dim*ob NEW

    // ---- per_row = heads * t ----
    mov   x0, x19
    mov   x1, x25
    bl    _t_num_mul
    mov   x27, x0
    mov   x0, x25
    bl    _t_decref
    cbz   x27, Lab_err
    mov   x24, x27                         // per_row NEW

    // ---- scratch_bytes < per_row -> raise ----
    mov   x0, x23
    mov   x1, x24
    mov   x2, #0                           // Py_LT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lab_err
    cmp   w0, #1
    b.eq  Lab_raise_fit

    // ---- result = scratch_bytes // per_row ----
    mov   x0, x23
    mov   x1, x24
    bl    _t_floordiv
    mov   x27, x0                          // result NEW
    // drop parks, then return result
    mov   x0, x24
    bl    _t_decref
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x27
    b     Lab_out

// ---- raise: drop parks, set ValueError, return NULL ----
Lab_raise_positive:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  cbz   x26, 2f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
2:  cbz   x24, 3f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
3:  cbz   x25, 4f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
4:  adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #56]                    // ValueError
    adrp  x1, m_positive@PAGE
    add   x1, x1, m_positive@PAGEOFF
    bl    _t_seterr
    mov   x0, #0
    b     Lab_out

Lab_raise_fit:
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #56]                    // ValueError
    adrp  x1, m_fit@PAGE
    add   x1, x1, m_fit@PAGEOFF
    bl    _t_seterr
    mov   x0, #0
    b     Lab_out

// ---- error path: exception already set; drop parks ----
Lab_err:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  cbz   x26, 2f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
2:  cbz   x24, 3f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
3:  cbz   x25, 4f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
4:  mov   x0, #0

Lab_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret
