// damendaemon/metal_projection_policy.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 10/370. 100% of use_packed_projection:
//     if rows <= 4: return True
//     if not fp32 or max(outputs*width, rows*width, rows*outputs) >= 2**32:
//         return False
//     return rows <= 8 or (rows <= 16 and outputs*width >= 4_000_000)
// All comparisons/arithmetic via PyObject_RichCompareBool/PyNumber_Multiply
// (arbitrary precision — the 2**27 test boundary is exact).
//
// ABI table (byte offsets), installed via dd_init:
//     0 PyObject_IsTrue         8 PyLong_FromLongLong
//    16 PyNumber_Multiply      24 PyObject_RichCompareBool
//    32 Py_DecRef              40 True
//    48 False

.globl _dd_init
.globl _dd_use_packed_projection

.section __DATA,__data
g_iface: .quad 0

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #0]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_num_mul:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_rcb:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_use_packed_projection(x0 = rows, x1 = outputs,
//                                     x2 = width, x3 = fp32)
// Register plan: x19 rows  x20 outputs  x21 width  x22 fp32-flag(w)
//   x23 m1 = outputs*width NEW   x24 m2 = rows*width NEW
//   x25 m3 = rows*outputs NEW    x26 mx NEW (only within step 6-7)
//   x27 const temp NEW           x28 result park across cleanup
// ============================================================
_dd_use_packed_projection:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    mov   x19, x0                           // rows (borrowed)
    mov   x20, x1                           // outputs (borrowed)
    mov   x21, x2                           // width (borrowed)
    mov   x23, xzr                          // m1 park
    mov   x24, xzr                          // m2 park
    mov   x25, xzr                          // m3 park
    mov   x26, xzr                          // mx park

    // ---- fp32 flag (convert now: x3 dies at the first C call) ----
    mov   x0, x3
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_ret_null
    mov   w22, w0

    // ---- rows <= 4 -> True ----
    mov   x0, #4
    bl    _t_fromlong
    cbz   x0, Lmp_ret_null
    mov   x27, x0
    mov   x0, x19
    mov   x1, x27
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0                          // cmp result (w) across decref
    mov   x0, x27
    bl    _t_decref
    cmn   w28, #1
    b.eq  Lmp_ret_null
    cmp   w28, #1
    b.eq  Lmp_ret_true

    // ---- not fp32 -> False ----
    cbz   w22, Lmp_ret_false

    // ---- m1 = outputs*width; m2 = rows*width; m3 = rows*outputs ----
    mov   x0, x20
    mov   x1, x21
    bl    _t_num_mul
    cbz   x0, Lmp_ret_null
    mov   x23, x0
    mov   x0, x19
    mov   x1, x21
    bl    _t_num_mul
    cbz   x0, Lmp_err
    mov   x24, x0
    mov   x0, x19
    mov   x1, x20
    bl    _t_num_mul
    cbz   x0, Lmp_err
    mov   x25, x0

    // ---- max(m1, m2): cand = m1 if m1 >= m2 else m2 ----
    mov   x0, x23
    mov   x1, x24
    mov   x2, #5                           // Py_GE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lmp_err
    cmp   w0, #1
    b.ne  1f
    mov   x0, x23
    bl    _incref
    mov   x26, x0
    b     2f
1:  mov   x0, x24
    bl    _incref
    mov   x26, x0
2:  // ---- max(cand, m3): mx = cand if cand >= m3 else m3 ----
    mov   x0, x26
    mov   x1, x25
    mov   x2, #5                           // Py_GE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lmp_err
    cmp   w0, #1
    b.eq  3f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x25
    bl    _incref
    mov   x26, x0
3:  // drop m1/m2/m3; x26 owns the chosen max
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    // KEEP x25 = m3? NO — m1 (outputs*width) is needed later, not m3.
    // Rebuild strategy: m1 was dropped; see note below.
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr

    // ---- mx >= 2**32 -> False ----
    movz  x0, #1, lsl #32
    bl    _t_fromlong
    cbz   x0, Lmp_ret_null
    mov   x27, x0
    mov   x0, x26
    mov   x1, x27
    mov   x2, #5                           // Py_GE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    cmn   w28, #1
    b.eq  Lmp_ret_null
    cmp   w28, #1
    b.eq  Lmp_ret_false

    // ---- rows <= 8 -> True ----
    mov   x0, #8
    bl    _t_fromlong
    cbz   x0, Lmp_ret_null
    mov   x27, x0
    mov   x0, x19
    mov   x1, x27
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    cmn   w28, #1
    b.eq  Lmp_ret_null
    cmp   w28, #1
    b.eq  Lmp_ret_true

    // ---- rows <= 16 (else False) ----
    mov   x0, #16
    bl    _t_fromlong
    cbz   x0, Lmp_ret_null
    mov   x27, x0
    mov   x0, x19
    mov   x1, x27
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    cmn   w28, #1
    b.eq  Lmp_ret_null
    cmp   w28, #1
    b.ne  Lmp_ret_false

    // ---- outputs*width >= 4_000_000 -> True else False ----
    // m1 was consumed; recompute outputs*width
    mov   x0, x20
    mov   x1, x21
    bl    _t_num_mul
    cbz   x0, Lmp_ret_null
    mov   x23, x0                          // m1 NEW
    movz  x0, #0x0900
    movk  x0, #0x3d, lsl #16                // 4_000_000 = 0x3D0900
    bl    _t_fromlong
    cbz   x0, Lmp_err
    mov   x27, x0
    mov   x0, x23
    mov   x1, x27
    mov   x2, #5                           // Py_GE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    cmn   w28, #1
    b.eq  Lmp_ret_null
    cmp   w28, #1
    b.eq  Lmp_ret_true
    b     Lmp_ret_false

// ---- shared exits: park result in x28-slot? x28 holds w-flags at
//      Lret time only for err; load the real result into x27? x27 is
//      dead here. Use x27 as the result carrier across cleanup. ----
Lmp_ret_true:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x27, [x9, #40]                    // True (borrowed, immortal)
    b     Lmp_cleanup
Lmp_ret_false:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x27, [x9, #48]                    // False (borrowed, immortal)
    b     Lmp_cleanup
Lmp_err:
Lmp_ret_null:
    mov   x27, xzr                          // NULL (exception may be set)

Lmp_cleanup:
    cbz   x26, 1f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
1:  cbz   x25, 2f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
2:  cbz   x24, 3f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
3:  cbz   x23, 4f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
4:  mov   x0, x27

    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret
