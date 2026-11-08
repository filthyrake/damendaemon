// damendaemon/glm_normalization.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 31/370. 100% of rms_norm:
//     variance = value.square().mean(-1, keepdim=True)
//     if not torch.isfinite(variance).all(): raise RuntimeError('nonfinite GLM normalization variance')
//     result = value*torch.rsqrt(variance+epsilon)*weight
//     if not torch.isfinite(result).all(): raise RuntimeError('nonfinite GLM normalization output')
//     return result
// AST order: left-associative (value*rsqrt)*weight; RuntimeError; static C messages.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString      8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem   32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem    56 PyLong_FromLongLong  64 PyObject_IsTrue
//    72 RuntimeError      80 PyErr_SetString  88 PyNumber_Add
//    96 torch.isfinite   104 torch.rsqrt    112 True
//   120 PyNumber_Multiply 128 PyUnicode_FromString

.globl _dd_init
.globl _dd_rms_norm

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_square:  .asciz "square"
a_mean:    .asciz "mean"
a_all:     .asciz "all"
k_keepdim: .asciz "keepdim"
m_var:     .asciz "nonfinite GLM normalization variance"
m_out:     .asciz "nonfinite GLM normalization output"

.macro IFACE off
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, \off]
    br    x9
.endm

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

_t_getattr_s:  IFACE #0
_t_call:       IFACE #8
_t_tuple_new:  IFACE #16
_t_tuple_setitem: IFACE #24
_t_decref:     IFACE #32
_t_dict_new:   IFACE #40
_t_dict_setitem: IFACE #48
_t_fromlong:   IFACE #56
_t_istrue:     IFACE #64
_t_seterr:     IFACE #80
_t_num_add:    IFACE #88
_t_num_mul:    IFACE #120
_t_fromstring: IFACE #128

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// helper: obj.all() -> IsTrue in w0 (obj NEW, consumed)
_allcheck:
    stp   x29, x30, [sp, #-48]!
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    mov   x19, x0                          // obj NEW
    mov   x0, x19
    adrp  x1, a_all@PAGE
    add   x1, x1, a_all@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lall_err
    mov   x20, x0                          // all method NEW
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lall_drop20
    mov   x21, x0                          // args NEW
    mov   x0, x20
    mov   x1, x21
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                          // r NEW
    mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref
    mov   x0, x19
    bl    _t_decref
    cbz   x22, Lall_err
    mov   x0, x22
    bl    _t_istrue
    mov   w21, w0
    mov   x0, x22
    bl    _t_decref
    mov   w0, w21
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret
Lall_drop20:
    mov   x0, x20
    bl    _t_decref
    mov   x0, x19
    bl    _t_decref
Lall_err:
    mov   w0, #-1                          // error signal
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// PyObject* _dd_rms_norm(x0 = value, x1 = weight, x2 = epsilon)
// x19 value (borrowed)  x20 weight (borrowed)  x21 epsilon (borrowed)
// x22 variance NEW      x23 result NEW        x24-x26 temps
// parks: [96] key/args  [104] kwargs  [112] result
// ============================================================
_dd_rms_norm:
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
    mov   x20, x1
    mov   x21, x2
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr
    mov   x26, xzr

    // ---- s = value.square() ----
    mov   x0, x19
    adrp  x1, a_square@PAGE
    add   x1, x1, a_square@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lgn_out
    str   x0, [sp, #96]                     // square method NEW
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lgn_drop96
    str   x0, [sp, #104]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // s NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x24, Lgn_out

    // ---- m = s.mean (method NEW) ----
    mov   x0, x24
    adrp  x1, a_mean@PAGE
    add   x1, x1, a_mean@PAGEOFF
    bl    _t_getattr_s
    mov   x25, x0                          // mean method NEW
    cbz   x25, Lgn_drop24

    // ---- kwargs = {'keepdim': True} ----
    bl    _t_dict_new
    cbz   x0, Lgn_drop25
    str   x0, [sp, #104]                    // kwargs NEW
    adrp  x0, k_keepdim@PAGE
    add   x0, x0, k_keepdim@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lgn_drop104
    str   x0, [sp, #96]                     // key NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #112]                   // True (borrowed)
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #96]
    bl    _t_dict_setitem
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]

    // ---- args = (-1,) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lgn_drop104
    str   x0, [sp, #96]                     // args NEW
    mov   x0, #-1
    bl    _t_fromlong
    cbz   x0, Lgn_drop96_104
    mov   x26, x0                          // -1 NEW
    mov   x0, x26
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x26
    bl    _t_tuple_setitem

    // ---- variance = m(args, kwargs) ----
    mov   x0, x25
    ldr   x1, [sp, #96]
    ldr   x2, [sp, #104]
    bl    _t_call
    mov   x22, x0                          // variance NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    cbz   x22, Lgn_out

    // ---- t = torch.isfinite(variance); not t.all() ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lgn_drop22
    str   x0, [sp, #96]
    mov   x0, x22
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #96]                   // torch.isfinite
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // t NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x24, Lgn_drop22
    mov   x0, x24
    bl    _allcheck
    cmn   w0, #1
    b.eq  Lgn_drop22                        // error -> propagate
    cmp   w0, #1
    b.ne  Lgn_raise_var                      // all() falsy -> nonfinite -> raise

    // ---- t = variance + epsilon; r = torch.rsqrt(t) ----
    mov   x0, x22
    mov   x1, x21
    bl    _t_num_add
    mov   x25, x0                          // t NEW
    cbz   x25, Lgn_drop22
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lgn_drop25
    str   x0, [sp, #96]
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x25
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #104]                  // torch.rsqrt
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                          // r NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    cbz   x26, Lgn_drop22

    // ---- result = (value*r)*weight — left-assoc ----
    mov   x0, x19
    mov   x1, x26
    bl    _t_num_mul
    mov   x25, x0                          // p NEW
    cbz   x25, Lgn_drop26
    mov   x0, x25
    mov   x1, x20
    bl    _t_num_mul
    mov   x23, x0                          // result NEW
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    cbz   x23, Lgn_drop22

    // ---- not torch.isfinite(result).all() ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lgn_drop23
    str   x0, [sp, #96]
    mov   x0, x23
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x23
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #96]                   // torch.isfinite
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // t NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x24, Lgn_drop23
    mov   x0, x24
    bl    _allcheck
    cmn   w0, #1
    b.eq  Lgn_drop23
    cmp   w0, #1
    b.ne  Lgn_raise_out

    // ---- success ----
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x23
    bl    _incref
    mov   x23, xzr                          // ownership transferred to x0
    b     Lgn_out

Lgn_raise_var:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #72]                   // RuntimeError
    adrp  x1, m_var@PAGE
    add   x1, x1, m_var@PAGEOFF
    bl    _t_seterr
    b     Lgn_drop22

Lgn_raise_out:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #72]                   // RuntimeError
    adrp  x1, m_out@PAGE
    add   x1, x1, m_out@PAGEOFF
    bl    _t_seterr
    b     Lgn_drop23

Lgn_drop96_104:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
Lgn_drop104:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cbz   x25, Lgn_drop24
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
Lgn_drop24:
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    b     Lgn_out
Lgn_drop96:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lgn_out
Lgn_drop26:
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lgn_drop25:
    cbz   x25, Lgn_drop22
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    b     Lgn_drop22
Lgn_drop23:
    cbz   x23, Lgn_drop22
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    b     Lgn_drop22
Lgn_drop22:
    cbz   x22, Lgn_out
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr

Lgn_out:
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #96]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #96]
2:  cbz   x26, 3f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
3:  cbz   x25, 4f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
4:  cbz   x24, 5f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
5:  cbz   x23, 6f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
6:  cbz   x22, 7f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
7:  ldr   x0, [sp, #112]
    str   xzr, [sp, #112]
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
