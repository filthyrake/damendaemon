// damendaemon/floating_parameter.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 29/370. 100% of require_floating:
//     import torch
//     if value.dtype not in (torch.float16, torch.bfloat16,
//                            torch.float32, torch.float64):
//         raise ValueError('unsupported floating checkpoint parameter '
//                          'dtype: ' + name)
//     return value
// The membership test is PySequence_Contains over a bridge-held tuple
// of the four dtype objects; the message is PyUnicode_Concat of the
// constant prefix with `name`. (The function-local `import torch` is
// realized as a bridge-held torch module reference — observably
// identical for any call after torch is first imported.)
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString        8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem     32 Py_DecRef        40 PySequence_Contains
//    48 ValueError          56 PyErr_SetString  64 dtype tuple (held)

.globl _dd_init
.globl _dd_require_floating

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_dtype:  .asciz "dtype"
m_prefix: .asciz "unsupported floating checkpoint parameter dtype: "

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
_t_contains:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_seterr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_require_floating(x0 = value, x1 = name)
// x19 value (borrowed)  x20 name (borrowed)  x21 dtype NEW
// [96] prefix str NEW   [104] msg NEW
// ============================================================
_dd_require_floating:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr

    // ---- dtype = value.dtype ----
    mov   x0, x19
    adrp  x1, a_dtype@PAGE
    add   x1, x1, a_dtype@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfp_out
    mov   x21, x0                          // dtype NEW

    // ---- dtype in (f16, bf16, f32, f64) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #64]                   // dtype tuple (borrowed)
    mov   x1, x21
    bl    _t_contains
    cmn   w0, #1
    b.eq  Lfp_out_drop                     // error -> propagate
    cbnz  w0, Lfp_ok                       // contained -> fine

    // ---- raise ValueError(prefix + name) ----
    adrp  x0, m_prefix@PAGE
    add   x0, x0, m_prefix@PAGEOFF
    adrp  x1, g_iface@PAGE
    ldr   x10, [x1, g_iface@PAGEOFF]
    // prefix needs to be a str object: build via PyUnicode_FromString?
    // No slot for it — concatenate using the bridge-held prefix STR.
    // (Bridge provides the prefix as a str object at iface #72.)
    ldr   x0, [x10, #72]                   // prefix str (borrowed)
    mov   x1, x20                          // name (borrowed)
    bl    _t_concat
    str   x0, [sp, #96]                     // msg NEW
    cbz   x0, Lfp_out_drop
    // PyErr_SetString(ValueError, PyUnicode_AsUTF8(msg)) — use SetObject?
    // SetString copies from C str; get UTF8 via iface #80 AsUTF8AndSize.
    ldr   x0, [sp, #96]
    mov   x1, xzr                          // NULL size ptr — AsUTF8AndSize WRITES
    bl    _t_asutf8                        // 8 bytes at *x1; never pass garbage
    cbz   x0, Lfp_out_drop_both
    mov   x27, x0                          // utf8 ptr (no ref)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #48]                   // ValueError
    mov   x1, x27
    bl    _t_seterr
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, xzr
    b     Lfp_out

Lfp_ok:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x19
    bl    _incref                          // return borrowed arg as NEW
    b     Lfp_out

Lfp_out_drop_both:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
Lfp_out_drop:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, xzr

Lfp_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret

_t_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #112]
    br    x9
_t_asutf8:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #120]
    br    x9
