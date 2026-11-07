// damendaemon/metal_q4_policy.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 11/370. 100% of validate:
//     geometry = dict(model_type='qwen2', hidden_size=896, num_hidden_layers=24,
//                     num_attention_heads=14, num_key_value_heads=2,
//                     intermediate_size=4864, vocab_size=151936, head_dim=64)
//     if (any(config.get(key) != value for key,value in geometry.items())
//             or config.get('rope_scaling')
//             or type(context) is not int or not 1 <= context <= 2048):
//         raise ValueError('Metal decoder quantization Q4 requires Qwen2.5-0.5B '
//                           'geometry, standard RoPE and context <=2048')
//     if max_batch is not None and (type(max_batch) is not int
//                                   or not 1 <= max_batch <= 4):
//         raise ValueError('Metal Q4 requires 1..4 request slots')
// The any() generator is realized as a loop over the same (key, value)
// pairs in the same insertion order with the same short-circuit.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyDict_New
//    24 PyDict_SetItem     32 PyUnicode_FromString  40 PyLong_FromLongLong
//    48 PyLong_Type        56 PyObject_RichCompareBool
//    64 PyObject_IsTrue    72 Py_DecRef
//    80 PyTuple_New       88 PyTuple_SetItem  96 PyObject_GetIter
//   104 PyIter_Next       112 PySequence_GetItem  120 None
//   128 PyErr_SetString   136 ValueError

.globl _dd_init
.globl _dd_validate

.section __DATA,__data
g_iface: .quad 0

// geometry key C-strings in insertion order; values (0 marks 'qwen2')
geo_keys:
    .quad k_model_type, k_hidden, k_layers, k_heads, k_kv_heads
    .quad k_inter, k_vocab, k_head_dim
geo_vals:
    .quad 0, 896, 24, 14, 2, 4864, 151936, 64

.section __TEXT,__cstring
k_model_type: .asciz "model_type"
k_hidden:     .asciz "hidden_size"
k_layers:     .asciz "num_hidden_layers"
k_heads:      .asciz "num_attention_heads"
k_kv_heads:   .asciz "num_key_value_heads"
k_inter:      .asciz "intermediate_size"
k_vocab:      .asciz "vocab_size"
k_head_dim:   .asciz "head_dim"
k_rope:       .asciz "rope_scaling"
a_get:        .asciz "get"
c_qwen2:      .asciz "qwen2"
m_geometry:   .asciz "Metal decoder quantization Q4 requires Qwen2.5-0.5B geometry, standard RoPE and context <=2048"
m_slots:      .asciz "Metal Q4 requires 1..4 request slots"

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
_t_dict_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_dict_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_rcb:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_seterr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #128]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_validate(x0 = config, x1 = context, x2 = max_batch)
// Register plan: x19 config  x20 context  x21 max_batch (borrowed)
//   x22 getm NEW      x23 geometry NEW   x24 loop index / temp
//   x25 key NEW       x26 val NEW        x27 got/const NEW
// ============================================================
_dd_validate:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    mov   x19, x0                           // config (borrowed)
    mov   x20, x1                           // context (borrowed)
    mov   x21, x2                           // max_batch (borrowed)
    mov   x22, xzr                          // getm park
    mov   x23, xzr                          // geometry park
    mov   x25, xzr                          // key park
    mov   x26, xzr                          // val park
    mov   x27, xzr                          // got/const park

    // ---- geometry dict (8 entries, insertion order) ----
    bl    _t_dict_new
    cbz   x0, Lmq_err
    mov   x23, x0
    mov   x24, xzr                          // i = 0
Lgeo_loop:
    cmp   x24, #8
    b.ge  Lgeo_done
    // key = PyUnicode_FromString(geo_keys[i])
    adrp  x9, geo_keys@PAGE
    add   x9, x9, geo_keys@PAGEOFF
    ldr   x0, [x9, x24, lsl #3]
    bl    _t_fromstring
    cbz   x0, Lmq_err
    mov   x25, x0                          // key NEW
    // val = i==0 ? 'qwen2' : long(geo_vals[i])
    adrp  x9, geo_vals@PAGE
    add   x9, x9, geo_vals@PAGEOFF
    ldr   x28, [x9, x24, lsl #3]
    cbnz  x24, 1f
    adrp  x0, c_qwen2@PAGE
    add   x0, x0, c_qwen2@PAGEOFF
    bl    _t_fromstring
    b     2f
1:  mov   x0, x28
    bl    _t_fromlong
2:  cbz   x0, Lmq_err
    mov   x26, x0                          // val NEW
    // geometry[key] = val
    mov   x0, x23
    mov   x1, x25
    mov   x2, x26
    bl    _t_dict_setitem
    // drop key/val
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    add   x24, x24, #1
    b     Lgeo_loop
Lgeo_done:

    // ---- getm = config.get ----
    mov   x0, x19
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmq_err
    mov   x22, x0                          // getm NEW

    // ---- any(config.get(key) != value for ...) ----
    mov   x24, xzr
Lany_loop:
    cmp   x24, #8
    b.ge  Lany_done
    adrp  x9, geo_keys@PAGE
    add   x9, x9, geo_keys@PAGEOFF
    ldr   x0, [x9, x24, lsl #3]
    bl    _t_fromstring
    cbz   x0, Lmq_err
    mov   x25, x0                          // key NEW
    adrp  x9, geo_vals@PAGE
    add   x9, x9, geo_vals@PAGEOFF
    ldr   x28, [x9, x24, lsl #3]
    cbnz  x24, 1f
    adrp  x0, c_qwen2@PAGE
    add   x0, x0, c_qwen2@PAGEOFF
    bl    _t_fromstring
    b     2f
1:  mov   x0, x28
    bl    _t_fromlong
2:  cbz   x0, Lmq_err
    mov   x26, x0                          // val NEW
    // args = (key,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmq_err
    str   x0, [sp, #96]                     // args tuple (park, above saves)
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x25
    bl    _t_tuple_setitem
    // got = config.get(key)
    mov   x0, x22
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                          // got NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x27, Lmq_err
    // got != val ?
    mov   x0, x27
    mov   x1, x26
    mov   x2, #3                           // Py_NE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lmq_err
    cmp   w0, #1
    b.eq  Lmq_raise_geometry
    // drop got/key/val; i++
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    add   x24, x24, #1
    b     Lany_loop
Lany_done:

    // ---- config.get('rope_scaling') truthy ----
    adrp  x0, k_rope@PAGE
    add   x0, x0, k_rope@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lmq_err
    mov   x25, x0
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmq_err
    str   x0, [sp, #96]
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x25
    bl    _t_tuple_setitem
    mov   x0, x22
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                          // rope value NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x27, Lmq_err
    mov   x0, x27
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmq_err
    cmp   w0, #1
    b.eq  Lmq_raise_geometry
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr

    // ---- type(context) is not int ----
    ldr   x9, [x20, #8]                     // ob_type(context)
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #48]                   // PyLong_Type
    cmp   x9, x10
    b.ne  Lmq_raise_geometry

    // ---- not 1 <= context <= 2048 ----
    mov   x0, #1
    bl    _t_fromlong
    cbz   x0, Lmq_err
    mov   x27, x0                          // one NEW
    mov   x0, x27
    mov   x1, x20
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0                          // park w-result
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    cmn   w28, #1
    b.eq  Lmq_err
    cmp   w28, #1
    b.ne  Lmq_raise_geometry
    movz  x0, #0x800                        // 2048
    bl    _t_fromlong
    cbz   x0, Lmq_err
    mov   x27, x0
    mov   x0, x20
    mov   x1, x27
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    cmn   w28, #1
    b.eq  Lmq_err
    cmp   w28, #1
    b.ne  Lmq_raise_geometry

    // ---- max_batch is not None -> check ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x10, [x10, #120]                  // None singleton
    cmp   x21, x10
    b.eq  Lmq_ret_none
    // type(max_batch) is not int
    ldr   x9, [x21, #8]
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x10, [x10, #48]
    cmp   x9, x10
    b.ne  Lmq_raise_slots
    // not 1 <= max_batch <= 4
    mov   x0, #1
    bl    _t_fromlong
    cbz   x0, Lmq_err
    mov   x27, x0
    mov   x0, x27
    mov   x1, x21
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    cmn   w28, #1
    b.eq  Lmq_err
    cmp   w28, #1
    b.ne  Lmq_raise_slots
    mov   x0, #4
    bl    _t_fromlong
    cbz   x0, Lmq_err
    mov   x27, x0
    mov   x0, x21
    mov   x1, x27
    mov   x2, #1                           // Py_LE
    bl    _t_rcb
    mov   x28, x0
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    cmn   w28, #1
    b.eq  Lmq_err
    cmp   w28, #1
    b.ne  Lmq_raise_slots

Lmq_ret_none:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]                   // None (borrowed)
    b     Lmq_cleanup

// ---- ValueError: geometry/RoPE/context ----
Lmq_raise_geometry:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // ValueError
    adrp  x1, m_geometry@PAGE
    add   x1, x1, m_geometry@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lmq_cleanup
// ---- ValueError: request slots ----
Lmq_raise_slots:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // ValueError
    adrp  x1, m_slots@PAGE
    add   x1, x1, m_slots@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lmq_cleanup

// ---- error path: exception set by C API; drop parks ----
Lmq_err:
    mov   x0, xzr

Lmq_cleanup:
    mov   x28, x0                          // park result across decrefs
    cbz   x27, 1f
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
1:  cbz   x26, 2f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
2:  cbz   x25, 3f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
3:  cbz   x23, 4f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
4:  cbz   x22, 5f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
5:  mov   x0, x28

    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #112
    ret
