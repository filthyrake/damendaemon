// damendaemon/awq_decode_policy.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 32/370. 100% of the module:
//   _ROWS = {(896,1152):2, (896,896):4, (896,9728):3, (4864,896):4}
//   @lru_cache(maxsize=None)
//   def _is_a2(device):
//       return torch.cuda.get_device_properties(device).name == 'NVIDIA A2'
//   def use_small_rows(rows, inputs, outputs, device):
//       if not 1 <= rows <= _ROWS.get((inputs, outputs), 0): return False
//       return device.type == 'cuda' and _is_a2(device)
// Fidelity notes:
//   - get_device_properties is resolved PER CALL via getattr chain on the
//     bridge-held torch module (tests patch torch.cuda.get_device_properties).
//   - the lru_cache is a bridge-held dict; lookup/store logic lives here.
//     Exceptions are never cached (error path returns before store).
//   - chained comparison short-circuits: `1 <= rows` false -> max not evaluated.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef        40 PyDict_GetItemWithError
//    48 PyDict_SetItem     56 RichCompareBool  64 PyBool_FromLong
//    72 PyDict_Clear       80 torch module     88 _ROWS dict
//    96 _CACHE dict       104 'NVIDIA A2'     112 'cuda'
//   120 int 1             128 int 0           136 False
//   144 PyErr_Occurred

.globl _dd_init
.globl _dd_is_a2
.globl _dd_cache_clear
.globl _dd_use_small_rows

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_cuda:   .asciz "cuda"
a_props:  .asciz "get_device_properties"
a_name:   .asciz "name"
a_type:   .asciz "type"
a_a2:     .asciz "NVIDIA A2"

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

_t_getattr_s:      IFACE #0
_t_call:           IFACE #8
_t_tuple_new:      IFACE #16
_t_tuple_setitem:  IFACE #24
_t_decref:         IFACE #32
_t_dict_getwe:     IFACE #40
_t_dict_setitem:   IFACE #48
_t_rcb:            IFACE #56
_t_boolfromlong:   IFACE #64
_t_dict_clear:     IFACE #72
_t_err_occurred:   IFACE #144

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// void dd_cache_clear() — lru_cache.cache_clear()
_dd_cache_clear:
    stp   x29, x30, [sp, #-16]!
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #96]                   // _CACHE
    bl    _t_dict_clear
    ldp   x29, x30, [sp], #16
    ret

// ============================================================
// PyObject* _dd_is_a2(x0 = device)  [lru_cache(maxsize=None)]
// x19 device borrowed   x20-x22 temps NEW   parks: [sp,#80]
// ============================================================
_dd_is_a2:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   xzr, xzr, [sp, #80]
    mov   x19, x0
    mov   x20, xzr
    mov   x21, xzr
    mov   x22, xzr

    // ---- cache hit? v = _CACHE[device] (borrowed) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #96]                   // _CACHE
    mov   x1, x19
    bl    _t_dict_getwe
    cbz   x0, La2_miss
    bl    _incref                           // hit: NEW ref for restype
    b     La2_out

La2_miss:
    // PyErr from unhashable key? then propagate
    bl    _t_err_occurred
    cbz   x0, 9f                           // no error -> real cache miss
    mov   x0, xzr                          // error set: return NULL
    b     La2_out
9:

    // ---- cuda_mod = torch.cuda ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #80]                   // torch
    adrp  x1, a_cuda@PAGE
    add   x1, x1, a_cuda@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, La2_out
    mov   x20, x0                          // cuda_mod NEW

    // ---- props = cuda_mod.get_device_properties (PER CALL) ----
    mov   x0, x20
    adrp  x1, a_props@PAGE
    add   x1, x1, a_props@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                          // props NEW
    cbz   x21, La2_drop20

    // ---- p = props(device) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, La2_drop21
    str   x0, [sp, #80]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    mov   x0, x21
    ldr   x1, [sp, #80]
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                          // p NEW
    ldr   x0, [sp, #80]
    bl    _t_decref
    str   xzr, [sp, #80]
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    cbz   x22, La2_out

    // ---- name = p.name ----
    mov   x0, x22
    adrp  x1, a_name@PAGE
    add   x1, x1, a_name@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                          // name NEW
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    cbz   x21, La2_out

    // ---- r = (name == 'NVIDIA A2') ----
    mov   x0, x21
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #104]                  // 'NVIDIA A2'
    mov   x2, #2                           // Py_EQ
    bl    _t_rcb
    mov   x23, x0                          // rcb int 0/1/-1 — NEVER a drop-chain reg
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    cmn   w23, #1
    b.eq  La2_out                          // -1 -> error, exception set

    // ---- b = bool(r) ----
    mov   x0, x23
    bl    _t_boolfromlong
    cbz   x0, La2_out
    mov   x21, x0                          // b NEW

    // ---- _CACHE[device] = b (store does not steal) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #96]                   // _CACHE
    mov   x1, x19
    mov   x2, x21
    bl    _t_dict_setitem
    cbnz  x0, 1f                           // nonzero (-1) = error
    mov   x0, x21                          // return b (transfer NEW ref)
    mov   x21, xzr
    b     La2_out
1:  b     La2_drop21

La2_drop21:
    cbz   x21, La2_drop20
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
La2_drop20:
    cbz   x20, 8f
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
8:  mov   x0, xzr                          // error path: NULL + exception

La2_out:
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// PyObject* _dd_use_small_rows(x0=rows, x1=inputs, x2=outputs, x3=device)
// x19 rows b  x20 inputs b  x21 outputs b  x22 device b
// x23 max b   x24 getwe result b   parks: [sp,#80] tuple/type
// Evaluation order mirrors CPython: `1 <= rows` FIRST, then the
// tuple build + _ROWS.get (only if the first compare holds).
// All parks borrowed — epilogue never decrefs.
// ============================================================
_dd_use_small_rows:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   xzr, xzr, [sp, #80]
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr
    mov   x26, xzr

    // ---- 1 <= rows ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]                   // int 1
    mov   x1, x19
    mov   x2, #1                            // Py_LE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lw_err_null
    cbz   w0, Lw_false                      // chain false -> return False

    // ---- t = (inputs, outputs) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lw_err_null
    str   x0, [sp, #80]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #1
    mov   x2, x21
    bl    _t_tuple_setitem

    // ---- v = _ROWS.get(t) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #88]                    // _ROWS
    ldr   x1, [sp, #80]
    bl    _t_dict_getwe
    mov   x24, x0                          // borrowed result (NULL=miss/err)
    ldr   x0, [sp, #80]
    bl    _t_decref
    str   xzr, [sp, #80]
    cbnz  x24, Lw_have_max
    bl    _t_err_occurred
    cbz   x0, Lw_max0                       // clean miss -> default 0
    b     Lw_err_null                        // error from lookup
Lw_max0:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x23, [x10, #128]                  // int 0 borrowed
    b     Lw_cmp2
Lw_have_max:
    mov   x23, x24

Lw_cmp2:
    // ---- rows <= max ----
    mov   x0, x19
    mov   x1, x23
    mov   x2, #1                            // Py_LE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lw_err_null
    cbz   w0, Lw_false

    // ---- device.type == 'cuda' ----
    mov   x0, x22
    adrp  x1, a_type@PAGE
    add   x1, x1, a_type@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lw_err_null
    mov   x24, x0                          // type NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #112]                  // 'cuda'
    mov   x0, x24
    mov   x2, #2                           // Py_EQ
    bl    _t_rcb
    mov   x25, x0
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    cmn   w25, #1
    b.eq  Lw_err_null
    cbz   w25, Lw_false

    // ---- return _is_a2(device) ----
    mov   x0, x22
    bl    _dd_is_a2
    b     Lw_out

Lw_false:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                  // False singleton (immortal)
    b     Lw_out
Lw_err_null:
    mov   x0, xzr                          // error path: NULL + exception
    b     Lw_out
Lw_out:
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret

