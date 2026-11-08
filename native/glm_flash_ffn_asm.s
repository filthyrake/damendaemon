// damendaemon/glm_flash_ffn.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 34/370. First class-def module WITH method
// bodies. 100% of the module:
//   def _limit(value):
//       if type(value) not in (int,float) or not 0<value<=torch.finfo(torch.float32).max:
//           raise ValueError('Flash requires a finite positive activation limit')
//       return value
//   class FlashFeedForward(GlmFeedForward):
//       def __init__(self,weights,*,activation_limit=10.,**options):
//           super().__init__(weights,activation_limit=_limit(activation_limit),**options)
//   class PackedFlashFeedForward(PackedGlmFeedForward):
//       def __init__(self,experts,*,activation_limit=10.,**options):
//           super().__init__(experts,activation_limit=_limit(activation_limit),**options)
// Fidelity notes:
//   - classes are docless: ns gets __module__/__qualname__/__init__ only;
//     type_new appends __dict__/__weakref__/__doc__:None last.
//   - _limit: EXACT type check (ob_type vs PyLong/PyFloat) — bool rejected.
//   - torch.finfo(torch.float32).max recomputed per call (module-global torch).
//   - kwargs merge: explicit activation_limit set first, then **options
//     overrides (PyDict_Merge override=1) — matches CPython call kwargs.
//   - super(): resolved via the real super builtin with (defining_class,
//     self) — identical to zero-arg super(); the class is fetched from the
//     bridge registry at call time (registry filled after class creation).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear       104 type            112 PyLong_Type
//   120 TypeError         128 GlmFeedForward  136 PackedGlmFeedForward
//   144 ValueError        152 PyFloat_Type    160 torch.finfo
//   168 torch.float32     176 int 0           184 PyDict_Merge
//   192 super             200 _CLSREG dict    208 key 'FlashFeedForward'
//   216 key 'PackedFlashFeedForward'  224 'activation_limit'
//   232 _flash_init fn    240 _packed_flash_init fn
//   248 RichCompareBool   256 PyDict_GetItemWithError  264 PyErr_Occurred

.globl _dd_init
.globl _dd_classes
.globl _dd_limit
.globl _dd_flash_ffn_init
.globl _dd_packed_flash_ffn_init

.section __DATA,__data
g_iface: .quad 0

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
_t_dict_new:       IFACE #40
_t_dict_setitem:   IFACE #48
_t_fromstring:     IFACE #56
_t_issub:          IFACE #64
_t_size:           IFACE #72
_t_seq_getitem:    IFACE #80
_t_seterr:         IFACE #88
_t_err_clear:      IFACE #96
_t_merge:          IFACE #184
_t_rcb:            IFACE #248
_t_dict_getwe:     IFACE #256
_t_err_occurred:   IFACE #264

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// _calc_meta(x0 = bases tuple BORROWED) -> x0 meta OWNED (NULL+exc)
// CPython _PyType_CalculateMetaclass: winner = type; per base:
// t = type(base); t==winner -> next; IsSubtype(winner,t) -> winner=t;
// !IsSubtype(t,winner) -> TypeError metaclass conflict.
// x19 bases  x20 winner(owned)  x21 base(NEW)  x22 i  x23 t  x24 n
// ============================================================
_calc_meta:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    mov   x19, x0                           // bases borrowed
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #104]
    bl    _incref
    mov   x20, x0                           // winner owned
    mov   x0, x19
    bl    _t_size
    cmp   x0, #1
    b.lt  Lcm_ret_winner                    // empty bases -> type
    mov   x24, x0                           // n
    mov   x22, #0
Lcm_loop:
    cmp   x22, x24
    b.ge  Lcm_ret_winner
    mov   x0, x19
    mov   x1, x22
    bl    _t_seq_getitem
    cbz   x0, Lcm_err
    mov   x21, x0                           // base NEW
    ldr   x23, [x21, #8]                    // t = type(base) borrowed
    cmp   x23, x20
    b.eq  Lcm_next
    mov   x0, x20                           // IsSubtype(winner, t)?
    mov   x1, x23
    bl    _t_issub
    cmp   w0, #1
    b.ne  Lcm_check_b
    mov   x0, x20                          // winner = t (more derived)
    bl    _t_decref
    mov   x0, x23
    bl    _incref
    mov   x20, x0
    b     Lcm_next
Lcm_check_b:
    mov   x0, x23                          // IsSubtype(t, winner)?
    mov   x1, x20
    bl    _t_issub
    cmp   w0, #1
    b.eq  Lcm_next
    // ---- metaclass conflict ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]
    adrp  x1, s_conflict@PAGE
    add   x1, x1, s_conflict@PAGEOFF
    bl    _t_seterr
    mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref
    mov   x0, xzr
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret
Lcm_next:
    mov   x0, x21
    bl    _t_decref
    add   x22, x22, #1
    b     Lcm_loop
Lcm_ret_winner:
    mov   x0, x20                           // meta owned
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret
Lcm_err:
    mov   x0, x20                           // drop winner; exc set
    bl    _t_decref
    mov   x0, xzr
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_limit(x0 = value)
//   type(value) not in (int, float) -> ValueError
//   not 0 < value <= torch.finfo(torch.float32).max -> ValueError
//   return value
// x19 value borrowed  x20 finfo NEW  x21 max NEW
// parks: [sp,#80] finfo  [sp,#88] max
// ============================================================
_dd_limit:
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

    // ---- exact type check: int or float (bool rejected) ----
    ldr   x9, [x19, #8]                      // ob_type
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x11, [x10, #112]                   // PyLong_Type
    cmp   x9, x11
    b.eq  Lli_type_ok
    ldr   x11, [x10, #152]                   // PyFloat_Type
    cmp   x9, x11
    b.eq  Lli_type_ok
    b     Lli_raise

Lli_type_ok:
    // ---- finfo = torch.finfo(torch.float32) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lli_out
    str   x0, [sp, #80]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #168]                   // torch.float32
    bl    _incref
    ldr   x0, [sp, #80]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #168]                   // torch.float32
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // torch.finfo
    ldr   x1, [sp, #80]
    mov   x2, #0
    bl    _t_call
    mov   x20, x0                           // finfo NEW
    ldr   x0, [sp, #80]
    bl    _t_decref
    str   xzr, [sp, #80]
    cbz   x20, Lli_out

    // ---- max = finfo.max ----
    mov   x0, x20
    adrp  x1, a_max@PAGE
    add   x1, x1, a_max@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                           // max NEW
    cbz   x21, Lli_drop20

    // ---- 0 < value ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #176]                   // int 0
    mov   x1, x19
    mov   x2, #0                            // Py_LT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lli_drop21
    cbz   w0, Lli_raise_drop                // 0 < value false -> raise

    // ---- value <= max ----
    mov   x0, x19
    mov   x1, x21
    mov   x2, #1                            // Py_LE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lli_drop21
    cbz   w0, Lli_raise_drop                // value <= max false -> raise

    // ---- success: return value (NEW ref) ----
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    mov   x0, x19
    bl    _incref
    b     Lli_out

Lli_raise_drop:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lli_drop20_21:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Lli_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                   // ValueError
    adrp  x1, s_limitmsg@PAGE
    add   x1, x1, s_limitmsg@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lli_out
Lli_drop21:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    b     Lli_drop20_21
Lli_drop20:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    mov   x0, xzr
Lli_out:
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// _ffn_init_impl(x0=self, x1=weights, x2=activation_limit,
//                x3=options dict, x4=cls key str BORROWED)
// super(cls, self).__init__(weights, activation_limit=_limit(limit),
//                           **options)
// x19 self b  x20 weights b  x21 limit b  x22 options b  x23 key b
// x24 cls b    x25 r NEW     x26 kwargs NEW
// parks: [80] super_obj NEW  [88] init_m NEW  [96] args NEW
// ============================================================
_ffn_init_impl:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]              // [96] super/init  [112] result
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, xzr
    mov   x25, xzr
    mov   x26, xzr
    mov   x27, xzr

    // ---- cls = _CLSREG[key] (borrowed) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #200]                   // _CLSREG
    mov   x1, x23
    bl    _t_dict_getwe
    cbnz  x0, 1f
    bl    _t_err_occurred
    cbnz  x0, Lfi_err_null                  // lookup error -> propagate
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                   // ValueError (unused path)
    adrp  x1, s_regmsg@PAGE
    add   x1, x1, s_regmsg@PAGEOFF
    bl    _t_seterr
    b     Lfi_err_null
1:  mov   x24, x0

    // ---- r = _limit(activation_limit) ----
    mov   x0, x21
    bl    _dd_limit
    cbz   x0, Lfi_err_null
    mov   x25, x0                           // r NEW

    // ---- kwargs = {'activation_limit': r}; merge **options ----
    bl    _t_dict_new
    cbz   x0, Lfi_drop25
    mov   x26, x0                           // kwargs NEW
    mov   x0, x26
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #224]                   // 'activation_limit' key
    mov   x2, x25
    bl    _t_dict_setitem
    cbnz  x0, Lfi_drop26
    mov   x0, x26
    mov   x1, x22
    mov   x2, #1                            // override: options wins
    bl    _t_merge
    cbnz  x0, Lfi_drop26

    // ---- super_obj = super(cls, self) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lfi_drop26
    str   x0, [sp, #96]
    mov   x0, x24
    bl    _incref
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x24
    bl    _t_tuple_setitem
    ldr   x0, [sp, #96]
    mov   x1, #1
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #192]                   // super builtin
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                           // super_obj NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x27, Lfi_drop26

    // ---- init_m = super_obj.__init__ ----
    mov   x0, x27
    adrp  x1, a_init@PAGE
    add   x1, x1, a_init@PAGEOFF
    bl    _t_getattr_s
    str   x0, [sp, #88]                     // init_m NEW
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x0, [sp, #88]
    cbz   x0, Lfi_drop26
    // ---- args = (weights,) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfi_drop88
    str   x0, [sp, #96]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem

    // ---- result = init_m(weights, **kwargs) ----
    ldr   x0, [sp, #88]
    ldr   x1, [sp, #96]
    mov   x2, x26
    bl    _t_call
    str   x0, [sp, #112]                    // result NEW

    // ---- cleanup ----
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #88]
    bl    _t_decref
    str   xzr, [sp, #88]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x0, [sp, #112]
    str   xzr, [sp, #112]
    b     Lfi_out

Lfi_drop88:
    ldr   x0, [sp, #88]
    bl    _t_decref
    str   xzr, [sp, #88]
Lfi_drop26:
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lfi_drop25:
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
Lfi_err_null:
    mov   x0, xzr
Lfi_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ---- entry points: pick the registry key, delegate ----
_dd_flash_ffn_init:
    stp   x29, x30, [sp, #-16]!
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x4, [x10, #208]                   // 'FlashFeedForward'
    bl    _ffn_init_impl
    ldp   x29, x30, [sp], #16
    ret
_dd_packed_flash_ffn_init:
    stp   x29, x30, [sp, #-16]!
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x4, [x10, #216]                   // 'PackedFlashFeedForward'
    bl    _ffn_init_impl
    ldp   x29, x30, [sp], #16
    ret

// ============================================================
// PyObject* _dd_classes(x0 = modname BORROWED) -> tuple of 2 classes
// Per class (docless): bases -> _calc_meta -> __prepare__ -> ns
// {__module__, __qualname__, __init__} -> meta(name, bases, ns).
// x19 meta(owned, per-class)  x28 result park across cleanup
// parks: [96] cls0  [104] cls1  [128] bases  [136] name
// [144] ns  [152] prep-args  [160] prepare  [176] meta-args/cls
// [184] key-temp
// ============================================================
_dd_classes:
    stp   x29, x30, [sp, #-224]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    str   xzr, [sp, #96]
    str   xzr, [sp, #104]
    str   xzr, [sp, #112]
    str   xzr, [sp, #120]
    str   xzr, [sp, #128]
    str   xzr, [sp, #136]
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #160]
    str   xzr, [sp, #176]
    str   xzr, [sp, #184]
    mov   x19, xzr
    str   x0, [sp, #168]

// ===== class 0: FlashFeedForward(GlmFeedForward) =====
Lcls_0:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // GlmFeedForward
    bl    _incref
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #128]
    mov   x1, #0
    ldr   x2, [x10, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _calc_meta
    cbz   x0, Lerr
    mov   x19, x0
    adrp  x0, cn_0@PAGE
    add   x0, x0, cn_0@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, s_prepare@PAGE
    add   x1, x1, s_prepare@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lc0_prep_fallback
    str   x0, [sp, #160]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #152]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #1
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #160]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #144]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    ldr   x9, [sp, #144]
    cbnz  x9, Lc0_ns_have
    b     Lerr
Lc0_prep_fallback:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Lerr
    str   x0, [sp, #144]
Lc0_ns_have:
    // __module__
    adrp  x0, k_module@PAGE
    add   x0, x0, k_module@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #168]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __qualname__
    adrp  x0, k_qualname@PAGE
    add   x0, x0, k_qualname@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __init__ (wrapper fn; docless — no __doc__)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]                    // ns
    ldr   x1, [sp, #176]                    // '__init__' key NEW
    ldr   x2, [x10, #232]                   // _flash_init (borrowed)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // meta(name, bases, ns)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #152]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #1
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #2
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    mov   x0, x19
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #176]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x9, [sp, #176]
    cbnz  x9, Lc0_park
    b     Lerr
Lc0_park:
    ldr   x0, [sp, #176]
    str   x0, [sp, #96]
    str   xzr, [sp, #176]
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr

// ===== class 1: PackedFlashFeedForward(PackedGlmFeedForward) =====
Lcls_1:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // PackedGlmFeedForward
    bl    _incref
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #128]
    mov   x1, #0
    ldr   x2, [x10, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _calc_meta
    cbz   x0, Lerr
    mov   x19, x0
    adrp  x0, cn_1@PAGE
    add   x0, x0, cn_1@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, s_prepare@PAGE
    add   x1, x1, s_prepare@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lc1_prep_fallback
    str   x0, [sp, #160]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #152]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #1
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #160]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #144]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    ldr   x9, [sp, #144]
    cbnz  x9, Lc1_ns_have
    b     Lerr
Lc1_prep_fallback:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Lerr
    str   x0, [sp, #144]
Lc1_ns_have:
    // __module__
    adrp  x0, k_module@PAGE
    add   x0, x0, k_module@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #168]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __qualname__
    adrp  x0, k_qualname@PAGE
    add   x0, x0, k_qualname@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __init__ (wrapper fn)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]                    // ns
    ldr   x1, [sp, #176]                    // '__init__' key NEW
    ldr   x2, [x10, #240]                   // _packed_flash_init (borrowed)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // meta(name, bases, ns)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #152]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #1
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #2
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    mov   x0, x19
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #176]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x9, [sp, #176]
    cbnz  x9, Lc1_park
    b     Lerr
Lc1_park:
    ldr   x0, [sp, #176]
    str   x0, [sp, #104]
    str   xzr, [sp, #176]
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr

// ---- result tuple of 2 ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #176]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #176]
    mov   x1, #1
    ldr   x2, [sp, #104]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #176]
    str   xzr, [sp, #176]
    b     Lexit
Lerr:
    mov   x0, xzr
Lexit:
    mov   x28, x0
    ldr   x0, [sp, #176]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #176]
1:  ldr   x0, [sp, #184]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #184]
1:  ldr   x0, [sp, #160]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #160]
1:  ldr   x0, [sp, #152]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #152]
1:  ldr   x0, [sp, #144]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #144]
1:  ldr   x0, [sp, #136]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #136]
1:  ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #96]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #96]
2:  cbz   x19, 3f
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
3:  mov   x0, x28
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #224
    ret

.section __TEXT,__cstring
s_prepare:  .asciz "__prepare__"
s_conflict: .asciz "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases"
k_module:   .asciz "__module__"
k_qualname: .asciz "__qualname__"
k_init:     .asciz "__init__"
a_max:      .asciz "max"
a_init:     .asciz "__init__"
s_limitmsg: .asciz "Flash requires a finite positive activation limit"
s_regmsg:   .asciz "asm bridge class registry not initialized"
cn_0:       .asciz "FlashFeedForward"
cn_1:       .asciz "PackedFlashFeedForward"


