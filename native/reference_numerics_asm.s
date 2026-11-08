// damendaemon/reference_numerics.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 30/370. 100% of finite:
//     if not torch.isfinite(value).all():
//         raise ValueError(f'nonfinite reference {stage}')
//     return value
// The f-string is PyUnicode_Concat('nonfinite reference ', stage).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString    8 PyObject_Call   16 PyTuple_New
//    24 PyTuple_SetItem  32 Py_DecRef       40 PyObject_IsTrue
//    48 PyUnicode_Concat 56 ValueError      64 PyErr_SetString
//    72 AsUTF8AndSize    80 torch.isfinite  88 prefix str

.globl _dd_init
.globl _dd_finite

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_all: .asciz "all"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

_t_getattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #0]
    br    x9
_t_call:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_seterr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_asutf8:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_finite(x0 = value, x1 = stage)
// x19 value (borrowed)  x20 stage (borrowed)
// x21 isfinite result NEW  x22 all-meth NEW  x23 all result NEW
// [96] args tuple  [104] msg NEW
// ============================================================
_dd_finite:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]              // [96] args  [104] msg  [112] result
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr

    // ---- t = torch.isfinite(value) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lrn_out
    str   x0, [sp, #96]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #80]                   // torch.isfinite
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x21, x0                           // t NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x21, Lrn_out

    // ---- m = t.all(); r = m() ----
    mov   x0, x21
    adrp  x1, a_all@PAGE
    add   x1, x1, a_all@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lrn_out_drop21
    mov   x22, x0                           // all method NEW
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lrn_out_drop2122
    str   x0, [sp, #96]
    mov   x0, x22
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x23, x0                           // all result NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    cbz   x23, Lrn_out_drop21

    // ---- not r ----
    mov   x0, x23
    bl    _t_istrue
    mov   x27, x0                           // park w0
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    cmn   w27, #1
    b.eq  Lrn_out_drop21
    cmp   w27, #1
    b.ne  Lrn_raise
    // finite -> return value (borrowed arg as NEW)
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x19
    bl    _incref
    b     Lrn_out

    // ---- raise ValueError('nonfinite reference ' + stage) ----
Lrn_raise:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #88]                   // prefix str (borrowed)
    mov   x1, x20                          // stage (borrowed)
    bl    _t_concat
    str   x0, [sp, #104]                    // msg NEW
    cbz   x0, Lrn_out
    ldr   x0, [sp, #104]
    mov   x1, xzr                          // NULL size ptr — AsUTF8AndSize WRITES
    bl    _t_asutf8                        // 8 bytes at *x1; never pass garbage
    cbz   x0, Lrn_out_drop104
    mov   x27, x0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #56]                   // ValueError
    mov   x1, x27
    bl    _t_seterr
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, xzr
    b     Lrn_out

Lrn_out_drop104:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, xzr
    b     Lrn_out
Lrn_out_drop2122:
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lrn_out_drop21:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, xzr

Lrn_out:
    str   x0, [sp, #112]                    // park result across decrefs
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #96]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #96]
2:  cbz   x23, 3f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
3:  cbz   x22, 4f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
4:  cbz   x21, 5f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
5:  ldr   x0, [sp, #112]
    str   xzr, [sp, #112]
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
