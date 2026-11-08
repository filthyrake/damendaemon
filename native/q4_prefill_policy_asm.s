// damendaemon/q4_prefill_policy.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 33/370. 100% of the module:
//   @lru_cache(maxsize=None)
//   def _is_a2(device):
//       return torch.cuda.get_device_properties(device).name == 'NVIDIA A2'
//   def use_small_tile(rows, outputs, width, device):
//       if not 32 < rows <= 64 or (width, outputs) not in (
//               (4096, 6144), (4096, 4096), (4096, 28672),
//               (14336, 4096), (4096, 32000)):
//           return False
//       return device.type == 'cuda' and _is_a2(device)
// Fidelity notes:
//   - get_device_properties resolved PER CALL via getattr chain on the
//     bridge-held torch module (tests patch torch.cuda.get_device_properties).
//   - the lru_cache is a bridge-held dict; lookup/store logic lives here.
//   - `or` short-circuit: the membership test runs only when 32 < rows <= 64
//     is False... no — when it HOLDS. CPython evaluates the chained compare,
//     then the tuple+membership only if the first disjunct is False.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef        40 PyDict_GetItemWithError
//    48 PyDict_SetItem     56 RichCompareBool  64 PyBool_FromLong
//    72 PyDict_Clear       80 torch module     88 _TILES tuple
//    96 _CACHE dict       104 'NVIDIA A2'     112 'cuda'
//   120 int 32            128 int 64          136 False
//   144 PyErr_Occurred    152 PySequence_Contains

.globl _dd_init
.globl _dd_is_a2
.globl _dd_cache_clear
.globl _dd_use_small_tile

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
_t_contains:       IFACE #152

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
// PyObject* _dd_use_small_tile(x0=rows, x1=outputs, x2=width, x3=device)
// x19 rows b  x20 outputs b  x21 width b  x22 device b
// x23/x24 temps b   parks: [sp,#80] tuple/type
// Evaluation order mirrors CPython: chained compare `32 < rows <= 64`
// first; the tuple build + membership runs only if it HOLDS (or-
// short-circuit); then device.type, then _is_a2.
// All parks borrowed — epilogue never decrefs.
// ============================================================
_dd_use_small_tile:
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

    // ---- 32 < rows ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]                   // int 32
    mov   x1, x19
    mov   x2, #0                            // Py_LT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lq_err_null
    cbz   w0, Lq_false                       // 32 < rows false -> `not` true

    // ---- rows <= 64 ----
    mov   x0, x19
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #128]                   // int 64
    mov   x2, #1                            // Py_LE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lq_err_null
    cbz   w0, Lq_false

    // ---- t = (width, outputs) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lq_err_null
    str   x0, [sp, #80]
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #0
    mov   x2, x21
    bl    _t_tuple_setitem
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #1
    mov   x2, x20
    bl    _t_tuple_setitem

    // ---- c = (width, outputs) in _TILES ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #88]                    // _TILES
    ldr   x1, [sp, #80]
    bl    _t_contains
    mov   x23, x0                           // 1 in / 0 not-in / -1 err
    ldr   x0, [sp, #80]
    bl    _t_decref
    str   xzr, [sp, #80]
    cmn   w23, #1
    b.eq  Lq_err_null
    cbz   w23, Lq_false                     // not in -> return False

    // ---- device.type == 'cuda' ----
    mov   x0, x22
    adrp  x1, a_type@PAGE
    add   x1, x1, a_type@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lq_err_null
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
    b.eq  Lq_err_null
    cbz   w25, Lq_false

    // ---- return _is_a2(device) ----
    mov   x0, x22
    bl    _dd_is_a2
    b     Lq_out

Lq_false:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                  // False singleton (immortal)
    b     Lq_out
Lq_err_null:
    mov   x0, xzr                          // error path: NULL + exception
    b     Lq_out
Lq_out:
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret

