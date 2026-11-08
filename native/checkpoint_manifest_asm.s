// damendaemon/checkpoint_manifest.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 44/370. 100% of the module logic:
//     def manifest_digest(checkpoint):
//         manifest = Path(checkpoint) / MANIFEST_NAME
//         if not manifest.is_file():
//             return None
//         return hashlib.sha256(MANIFEST_NAME.encode() + b'\0'
//                               + manifest.read_bytes()).digest()
// MANIFEST_NAME, Path, b'\0' and hashlib.sha256 are bridge-held;
// `/` = PyNumber_TrueDivide (Path.__truediv__), `+` = PyNumber_Add.
//
// ABI table (byte offsets), installed via dd_init:
//     0 PyObject_Call    8 Py_DecRef   16 PyObject_GetAttrString
//    24 PyNumber_TrueDivide  32 PyNumber_Add  40 PyObject_IsTrue
//    48 None            56 Path        64 MANIFEST_NAME
//    72 b'\0'           80 sha256      88 PyTuple_New
//    96 PyTuple_SetItem

.globl _dd_init
.globl _dd_manifest_digest

.section __DATA,__data
g_iface: .quad 0

.macro IFACE off
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, \off]
    br    x9
.endm

.section __TEXT,__cstring
a_isfile:    .asciz "is_file"
a_encode:    .asciz "encode"
a_readbytes: .asciz "read_bytes"
a_digest:    .asciz "digest"

.text
_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret
_t_call:          IFACE #0
_t_decref:        IFACE #8
_t_getattr_s:     IFACE #16
_t_truediv:       IFACE #24
_t_num_add:       IFACE #32
_t_istrue:        IFACE #40
_t_tuple_new:      IFACE #88
_t_tuple_setitem:  IFACE #96

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// helper: call fn (x0-held via park) with the single arg parked at [sp,#104]
// call1 layout below uses parks [96]=args [104]=value [112]=park

// ============================================================
// PyObject* _dd_manifest_digest(x0=checkpoint BORROWED) -> bytes/None
// x19 ckpt b  x21 m NEW  x22 c NEW
// parks: [96] args  [104] value/it  [112] park
// ============================================================
_dd_manifest_digest:
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

    // ---- p = Path(checkpoint) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmd_err
    str   x0, [sp, #96]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #56]                   // Path
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                   // park
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #112]
    cbz   x0, Lmd_err
    str   x0, [sp, #112]                   // p stays parked (used once)

    // ---- m = p / MANIFEST_NAME ----
    ldr   x0, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #64]                   // MANIFEST_NAME
    bl    _t_truediv
    str   x0, [sp, #104]                   // park m
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    cbz   x0, Lmd_err
    ldr   x21, [sp, #104]                  // m NEW
    str   xzr, [sp, #104]

    // ---- f = m.is_file() ; if not f: return None ----
    mov   x0, x21
    adrp  x1, a_isfile@PAGE
    add   x1, x1, a_isfile@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmd_drop21_err
    str   x0, [sp, #96]                    // is_file m
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmd_drop96_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                   // f park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x9, [sp, #112]
    str   xzr, [sp, #112]
    cbz   x9, Lmd_drop21_err
    mov   x0, x9
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmd_drop21_err
    cmp   w0, #1
    b.eq  Lmd_isfile
    // not a file: drop m, return None
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #48]                   // None
    b     Lmd_out
Lmd_isfile:

    // ---- b1 = MANIFEST_NAME.encode() ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #64]                   // MANIFEST_NAME
    adrp  x1, a_encode@PAGE
    add   x1, x1, a_encode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmd_drop21_err
    str   x0, [sp, #96]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmd_drop96_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                   // b1 park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #112]
    cbz   x0, Lmd_drop21_err
    // z = b1 + b'\0'
    ldr   x0, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #72]                   // b'\0'
    bl    _t_num_add
    str   x0, [sp, #104]                   // z park
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    cbz   x0, Lmd_drop21_err

    // ---- r = m.read_bytes() ----
    mov   x0, x21
    adrp  x1, a_readbytes@PAGE
    add   x1, x1, a_readbytes@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmd_drop104_err
    str   x0, [sp, #96]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmd_drop96_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                   // r park (m freed below)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x9, [sp, #112]
    cbz   x9, Lmd_drop104_err
    // m no longer needed; free it now
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr

    // ---- c = z + r  (z parked at [104], r parked at [112]) ----
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #112]
    bl    _t_num_add
    str   x0, [sp, #120]                   // c park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #120]
    cbz   x0, Lmd_err
    ldr   x22, [sp, #120]                  // c NEW
    str   xzr, [sp, #120]

    // ---- h = sha256(c) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmd_drop22_err
    str   x0, [sp, #96]
    mov   x0, x22
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #80]                   // sha256
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                   // h park
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    cbz   x0, Lmd_drop22_err

    // ---- d = h.digest() ----
    ldr   x0, [sp, #104]                   // h
    adrp  x1, a_digest@PAGE
    add   x1, x1, a_digest@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmd_drop104_err
    str   x0, [sp, #96]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmd_drop96_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]                   // d park
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]                   // drop h
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    ldr   x0, [sp, #120]
    str   xzr, [sp, #120]
    b     Lmd_out

// ---- error sweeps ----
Lmd_drop96_err:
    ldr   x0, [sp, #96]
    cbz   x0, Lmd_drop104_err
    bl    _t_decref
    str   xzr, [sp, #96]
Lmd_drop104_err:
    ldr   x0, [sp, #104]
    cbz   x0, Lmd_drop22_err
    bl    _t_decref
    str   xzr, [sp, #104]
Lmd_drop22_err:
    cbz   x22, Lmd_drop21_err
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lmd_drop21_err:
    cbz   x21, Lmd_err
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lmd_err:
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
3:  cbz   x22, 4f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
4:  cbz   x21, 5f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
5:  mov   x0, xzr                           // NULL + pending exception
    b     Lmd_out
Lmd_out:
    ldr   x9, [sp, #120]                    // stray park safety (normally 0)
    cbz   x9, 6f
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #120]
6:  ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
