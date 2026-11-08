// damendaemon/cuda_glm_flash_indexer.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 35/370. Class-def with __init__ body:
//   class CudaFlashIndexer(_FlashIndexer):
//       def __init__(self,weights,**options):
//           device=_device(weights.get('wk.weight') if isinstance(weights,dict) else None)
//           super().__init__(weights,device=device,**options)
// _device is bridge-imported (cuda_glm_flash_primitives); the conditional
// expr, .get call, kwargs merge, and real super() all live here.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear       104 type            112 dict class
//   120 TypeError         128 _FlashIndexer  136 _device fn
//   144 'wk.weight'       152 None            160 'device' key
//   168 _CLSREG dict      176 key 'CudaFlashIndexer'  184 _cuda_init fn
//   192 PyDict_GetItemWithError  200 super    208 PyDict_Merge
//   216 PyObject_IsInstance  224 PyErr_Occurred

.globl _dd_init
.globl _dd_classes
.globl _dd_cuda_init

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
_t_dict_getwe:     IFACE #192
_t_super:          IFACE #200
_t_merge:          IFACE #208
_t_isinst:         IFACE #216
_t_err_occurred:   IFACE #224

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// _calc_meta(x0 = bases tuple BORROWED) -> x0 meta OWNED (NULL+exc)
// ============================================================
_calc_meta:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    mov   x19, x0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #104]
    bl    _incref
    mov   x20, x0
    mov   x0, x19
    bl    _t_size
    cmp   x0, #1
    b.lt  Lcm_ret_winner
    mov   x24, x0
    mov   x22, #0
Lcm_loop:
    cmp   x22, x24
    b.ge  Lcm_ret_winner
    mov   x0, x19
    mov   x1, x22
    bl    _t_seq_getitem
    cbz   x0, Lcm_err
    mov   x21, x0
    ldr   x23, [x21, #8]
    cmp   x23, x20
    b.eq  Lcm_next
    mov   x0, x20
    mov   x1, x23
    bl    _t_issub
    cmp   w0, #1
    b.ne  Lcm_check_b
    mov   x0, x20
    bl    _t_decref
    mov   x0, x23
    bl    _incref
    mov   x20, x0
    b     Lcm_next
Lcm_check_b:
    mov   x0, x23
    mov   x1, x20
    bl    _t_issub
    cmp   w0, #1
    b.eq  Lcm_next
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
    mov   x0, x20
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret
Lcm_err:
    mov   x0, x20
    bl    _t_decref
    mov   x0, xzr
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_cuda_init(x0=self, x1=weights, x2=options dict)
// x19 self b  x20 weights b  x21 options b  x22 dev NEW
// x23 kwargs NEW  x24 cls b  x25 value b
// parks: [96] super_obj NEW  [104] init_m NEW  [112] args NEW  [120] result
// ============================================================
_dd_cuda_init:
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
    mov   x27, xzr

    // ---- cls = _CLSREG['CudaFlashIndexer'] ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #168]
    ldr   x1, [x10, #176]
    bl    _t_dict_getwe
    cbnz  x0, 1f
    bl    _t_err_occurred
    cbnz  x0, Lci_err_null
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]
    adrp  x1, s_regmsg@PAGE
    add   x1, x1, s_regmsg@PAGEOFF
    bl    _t_seterr
    b     Lci_err_null
1:  mov   x24, x0

    // ---- value = weights.get('wk.weight') if isinstance(weights,dict) else None ----
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #112]                   // dict
    bl    _t_isinst
    cmn   w0, #1
    b.eq  Lci_err_null
    cbz   w0, Lci_value_none
    mov   x0, x20
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lci_err_null
    mov   x26, x0                          // .get NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lci_drop26
    str   x0, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                  // 'wk.weight'
    bl    _incref
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #112]
    mov   x1, #0
    ldr   x2, [x10, #144]
    bl    _t_tuple_setitem
    mov   x0, x26
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    mov   x25, x0                          // value NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    cbz   x25, Lci_err_null
    b     Lci_have_value
Lci_value_none:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x25, [x10, #152]                  // None borrowed
Lci_have_value:
    mov   x0, x25
    bl    _incref                           // uniform ownership

    // ---- dev = _device(value) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lci_drop25
    str   x0, [sp, #112]
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x25
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // _device
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                          // dev NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    cbz   x22, Lci_drop25

    // ---- kwargs = {'device': dev}; merge **options ----
    bl    _t_dict_new
    cbz   x0, Lci_drop22
    mov   x23, x0
    mov   x0, x23
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #160]                   // 'device' key
    mov   x2, x22
    bl    _t_dict_setitem
    cbnz  x0, Lci_drop23
    mov   x0, x23
    mov   x1, x21
    mov   x2, #1
    bl    _t_merge
    cbnz  x0, Lci_drop23

    // ---- super_obj = super(cls, self) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lci_drop23
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
    ldr   x0, [x10, #200]                   // super builtin
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                          // super_obj NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x27, Lci_drop23

    // ---- init_m = super_obj.__init__ ----
    mov   x0, x27
    adrp  x1, a_init@PAGE
    add   x1, x1, a_init@PAGEOFF
    bl    _t_getattr_s
    str   x0, [sp, #104]
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x0, [sp, #104]
    cbz   x0, Lci_drop23

    // ---- args = (weights,) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lci_drop104
    str   x0, [sp, #112]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem

    // ---- result = init_m(weights, **kwargs) ----
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #112]
    mov   x2, x23
    bl    _t_call
    str   x0, [sp, #120]

    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
Lci_drop104:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
Lci_drop23:
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Lci_drop22:
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lci_drop25:
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x0, [sp, #120]
    str   xzr, [sp, #120]
    b     Lci_out
Lci_drop26:
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lci_err_null:
    mov   x0, xzr
Lci_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// === PART 3: dd_classes — single docless class with __init__ ===
// PyObject* _dd_classes(x0 = modname BORROWED) -> 1-tuple (cls,)
// parks: [96] cls  [128] bases  [136] name  [144] ns
// [152] prep-args  [160] prepare  [176] meta-args/cls
_dd_classes:
    stp   x29, x30, [sp, #-224]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    str   xzr, [sp, #96]
    str   xzr, [sp, #128]
    str   xzr, [sp, #136]
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #160]
    str   xzr, [sp, #176]
    mov   x19, xzr
    str   x0, [sp, #168]

    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // _FlashIndexer
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
    // __init__ (docless — no __doc__)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [x10, #184]                   // _cuda_init (borrowed)
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

    // result 1-tuple
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #176]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #176]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
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
a_get:      .asciz "get"
a_init:     .asciz "__init__"
s_regmsg:   .asciz "asm bridge class registry not initialized"
cn_0:       .asciz "CudaFlashIndexer"

