// damendaemon/model_path.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 5/370. 100% of canonical_model_path:
//   p = Path(value).expanduser()
//   if re.fullmatch(r'.+-00001-of-[0-9]{5}\\.gguf', p.name, re.IGNORECASE):
//       return p.parent.resolve() / p.name
//   return p.resolve()
// The pattern bytes (including BOTH literal backslashes of the raw
// string) are reproduced exactly — see PORTING_STATUS.md note.
//
// ABI table (offsets in bytes), installed via dd_init:
//     0 PyObject_GetAttrString     8 PyObject_Call
//    16 PyUnicode_FromString      24 PyNumber_TrueDivide
//    32 Py_DecRef                 40 PyTuple_New
//    48 PyTuple_SetItem           56 re module
//    64 Path class                72 None singleton

.globl _dd_init
.globl _dd_canonical_model_path

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
// r'.+-00001-of-[0-9]{5}\.gguf' — single backslash + escaped dot
// (verified byte-exact against the source via od -c; the file-read
// rendering doubles backslashes — always od the pattern bytes)
c_pattern:  .asciz ".+-00001-of-[0-9]{5}\\.gguf"
a_expand:   .asciz "expanduser"
a_name:     .asciz "name"
a_fullmatch:.asciz "fullmatch"
a_ignore:   .asciz "IGNORECASE"
a_parent:   .asciz "parent"
a_resolve:  .asciz "resolve"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines
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
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_truediv:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_canonical_model_path(x0 = value) -> Path
// Register plan (all callee-saved, all owned unless noted):
//   x19 value (borrowed arg)   x20 p (Path)      x21 temp/name
//   x22 fm                     x23 pattern      x24 flags
//   x25 match                  x26 temp
// ============================================================
_dd_canonical_model_path:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    // [x29,#80] args tuple parking
    mov   x19, x0                          // value (borrowed)

    // ---- p = Path(value) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmp_ret0
    str   x0, [x29, #80]
    mov   x0, x19
    bl    _incref                          // tuple steals
    ldr   x0, [x29, #80]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #64]                    // Path callable
    ldr   x1, [x29, #80]
    mov   x2, #0
    bl    _t_call
    mov   x20, x0                          // p = Path(value)
    ldr   x0, [x29, #80]
    bl    _t_decref
    cbz   x20, Lmp_ret0

    // ---- p = p.expanduser() ----
    mov   x0, x20
    adrp  x1, a_expand@PAGE
    add   x1, x1, a_expand@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_p
    mov   x21, x0                          // expanduser bound method
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmp_drop_meth
    str   x0, [x29, #80]
    mov   x0, x21
    ldr   x1, [x29, #80]
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                          // p2 = expanduser()
    ldr   x0, [x29, #80]
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref                        // drop pre-expand p
    cbz   x22, Lmp_ret0
    mov   x20, x22                          // p = expanded

    // ---- name = p.name ----
    mov   x0, x20
    adrp  x1, a_name@PAGE
    add   x1, x1, a_name@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_p
    mov   x21, x0                          // name (owned)

    // ---- fm = re.fullmatch ----
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #56]                    // re module
    adrp  x1, a_fullmatch@PAGE
    add   x1, x1, a_fullmatch@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_name
    mov   x22, x0                          // fm (owned)

    // ---- pattern str ----
    adrp  x0, c_pattern@PAGE
    add   x0, x0, c_pattern@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lmp_drop_fm
    mov   x23, x0                          // pattern (owned)

    // ---- flags = re.IGNORECASE ----
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #56]
    adrp  x1, a_ignore@PAGE
    add   x1, x1, a_ignore@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_pat
    mov   x24, x0                          // flags (owned)

    // ---- args = (pattern, name, flags) ----
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lmp_drop_flags
    str   x0, [x29, #80]
    mov   x0, x23
    bl    _incref
    ldr   x0, [x29, #80]
    mov   x1, #0
    mov   x2, x23
    bl    _t_tuple_setitem
    mov   x0, x21
    bl    _incref
    ldr   x0, [x29, #80]
    mov   x1, #1
    mov   x2, x21
    bl    _t_tuple_setitem
    mov   x0, x24
    bl    _incref
    ldr   x0, [x29, #80]
    mov   x1, #2
    mov   x2, x24
    bl    _t_tuple_setitem

    // ---- m = fm(pattern, name, flags) ----
    mov   x0, x22
    ldr   x1, [x29, #80]
    mov   x2, #0
    bl    _t_call
    mov   x25, x0                          // m (owned; None if no match)
    ldr   x0, [x29, #80]
    bl    _t_decref                        // drop args
    mov   x0, x23
    bl    _t_decref                        // drop pattern
    mov   x0, x24
    bl    _t_decref                        // drop flags
    mov   x0, x22
    bl    _t_decref                        // drop fm
    cbz   x25, Lmp_drop_name

    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x10, [x9, #72]                   // None
    cmp   x25, x10
    b.eq  Lmp_plain

    // ---- match: return p.parent.resolve() / name ----
    mov   x0, x20
    adrp  x1, a_parent@PAGE
    add   x1, x1, a_parent@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_all
    mov   x26, x0                          // parent (owned)
    mov   x0, x26
    adrp  x1, a_resolve@PAGE
    add   x1, x1, a_resolve@PAGEOFF
    bl    _t_getattr_s
    mov   x23, x0                          // resolve method (owned)
    cbz   x23, Lmp_drop_parent
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmp_drop_resmeth
    str   x0, [x29, #80]
    mov   x0, x23
    ldr   x1, [x29, #80]
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // resolved parent (owned)
    ldr   x0, [x29, #80]
    bl    _t_decref
    mov   x0, x23
    bl    _t_decref
    mov   x0, x26
    bl    _t_decref
    cbz   x24, Lmp_drop_all
    // result = resolved / name   (TrueDivide: no steal, new ref)
    mov   x0, x24
    mov   x1, x21
    bl    _t_truediv
    mov   x26, x0                          // result
    mov   x0, x24
    bl    _t_decref
    mov   x0, x25
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref
    mov   x0, x26
    b     Lmp_ret
    b     Lmp_ret0                          // unreachable guard

    // ---- no match: return p.resolve() ----
Lmp_plain:
    mov   x0, x20
    adrp  x1, a_resolve@PAGE
    add   x1, x1, a_resolve@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_all
    mov   x26, x0                          // resolve method
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmp_drop_resmeth2
    str   x0, [x29, #80]
    mov   x0, x26
    ldr   x1, [x29, #80]
    mov   x2, #0
    bl    _t_call
    mov   x23, x0                          // result
    ldr   x0, [x29, #80]
    bl    _t_decref
    mov   x0, x26
    bl    _t_decref
    mov   x0, x25
    bl    _t_decref                        // drop the None match
    mov   x0, x21
    bl    _t_decref                        // drop name
    mov   x0, x20
    bl    _t_decref                        // drop p
    mov   x0, x23
    b     Lmp_ret

    // ---- error-path drops ----
Lmp_drop_resmeth:
    mov   x0, x23
    bl    _t_decref
    b     Lmp_drop_parent
Lmp_drop_parent:
    mov   x0, x26
    bl    _t_decref
    b     Lmp_drop_all
Lmp_drop_resmeth2:
    mov   x0, x26
    bl    _t_decref
    b     Lmp_drop_all
Lmp_drop_all:
    mov   x0, x25
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    b     Lmp_drop_p
Lmp_drop_flags:
    mov   x0, x24
    bl    _t_decref
Lmp_drop_pat:
    mov   x0, x23
    bl    _t_decref
Lmp_drop_fm:
    mov   x0, x22
    bl    _t_decref
Lmp_drop_name:
    mov   x0, x21
    bl    _t_decref
Lmp_drop_meth:
    mov   x0, x21
    bl    _t_decref
Lmp_drop_p:
    mov   x0, x20
    bl    _t_decref
Lmp_ret0:
    mov   x0, #0
Lmp_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x29, x30, [sp], #96
    ret
