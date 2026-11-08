// damendaemon/moe_prefill.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 45/370. 100% of the module:
//     def qualified_scope(config, device_name, quantization, context, max_batch):
//         geometry = dict(model_type='olmoe', hidden_size=2048, intermediate_size=1024,
//                         num_hidden_layers=16, num_attention_heads=16,
//                         num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
//                         vocab_size=50304, norm_topk_prob=False)
//         return (device_name == 'NVIDIA A2' and quantization == 'q8_0'
//                 and 1 <= context <= 2048 and 1 <= max_batch <= 4
//                 and all(config.get(key) == value for key, value in geometry.items())
//                 and config.get('clip_qkv') is None)
//     def use_grouped_prefill(qualified, tokens):
//         return qualified and 1 <= tokens <= 128
//     def use_packed_prefill(qualified, lengths):
//         return (qualified and 2 <= len(lengths) <= 4
//                 and all(length > 0 for length in lengths) and sum(lengths) <= 128)
// Fidelity notes:
//   - `and` chains return the actual operand (CPython semantics): a falsy
//     operand is returned as-is (borrowed args get an incref; RCB results
//     are NEW); the last operand is returned when all are truthy.
//   - chained comparisons a <= b <= c = two RCB calls (EQ=2 LE=1 GT=4);
//     comparisons via PyObject_RichCompare (object results, CPython
//     COMPARE_OP), truth via PyObject_IsTrue.
//   - all(genexp) and sum(genexp) are evaluated as explicit GetIter/
//     IterNext loops (value-identical; all() always returns exactly the
//     True/False singletons, sum() = PyNumber_Add chain from int 0).
//   - geometry dict is rebuilt fresh per call (CPython builds the call
//     kwargs dict per evaluation); keys via PyUnicode_FromString.
//   - `is None` = pointer compare against the None singleton.
//
// ABI table (byte offsets), installed via dd_init:
//     0 PyObject_Call     8 Py_DecRef   16 PyObject_GetAttrString
//    24 PyObject_RichCompare  32 PyObject_IsTrue  40 PyDict_New
//    48 PyDict_SetItem   56 PyIter_Next  64 PyObject_GetIter
//    72 PyNumber_Add     80 True    88 False    96 None
//   104 'olmoe'         112 2048    120 1024    128 16
//   136 64             144 8       152 50304   160 'NVIDIA A2'
//   168 'q8_0'          176 'clip_qkv'  184 int 1  192 int 2
//   200 int 4           208 int 0   216 int 128  224 PyTuple_New
//   232 PyTuple_SetItem  240 PyUnicode_FromString
//   248 PySequence_GetItem  256 PyObject_Size  264 PyLong_FromSsize_t

.globl _dd_init
.globl _dd_qualified
.globl _dd_grouped
.globl _dd_packed

.section __DATA,__data
g_iface: .quad 0

.macro IFACE off
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, \off]
    br    x9
.endm

.section __TEXT,__cstring
k_model_type:    .asciz "model_type"
k_hidden:       .asciz "hidden_size"
k_inter:        .asciz "intermediate_size"
k_layers:       .asciz "num_hidden_layers"
k_heads:        .asciz "num_attention_heads"
k_kvheads:      .asciz "num_key_value_heads"
k_experts:      .asciz "num_experts"
k_epertok:      .asciz "num_experts_per_tok"
k_vocab:        .asciz "vocab_size"
k_normtop:      .asciz "norm_topk_prob"
a_get:          .asciz "get"
a_items:        .asciz "items"

.text
_t_fromstring:     IFACE #240
_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret
_t_call:          IFACE #0
_t_decref:        IFACE #8
_t_getattr_s:     IFACE #16
_t_rcb:           IFACE #24
_t_istrue:        IFACE #32
_t_dict_new:      IFACE #40
_t_dict_setitem:  IFACE #48
_t_iternext:      IFACE #56
_t_getiter:       IFACE #64
_t_num_add:       IFACE #72
_t_tuple_new:      IFACE #224
_t_tuple_setitem:  IFACE #232
_t_seq_getitem:    IFACE #248
_t_size:           IFACE #256
_t_fromssize:      IFACE #264

_incref:
    // immortal singletons (True/False/None in 3.12, PEP 683) live in
    // read-only pages — a raw refcount write segfaults on them. Route
    // through the real Py_IncRef, which checks immortality first.
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x11, [x10, #272]                  // Py_IncRef
    br    x11

// helper macro: r = iface[off] (obj) — load into x0 with a fresh adrp
.macro ILOAD off
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, \off]
.endm

// helper: store one (cstring-key, iface-val) pair into the dict at [sp,#96]
// usage: DSET kname, valoff  (dict park must be at [sp,#96], key park [sp,#104])
.macro DSET kname, valoff
    adrp  x0, \kname@PAGE
    add   x0, x0, \kname@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lmp_err
    str   x0, [sp, #104]                     // key park
    ldr   x1, [sp, #104]
    ILOAD \valoff                           // x0 = value
    mov   x2, x0
    ldr   x0, [sp, #96]                      // x0 = dict (reload AFTER val load)
    bl    _t_dict_setitem
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
.endm

// ============================================================
// PyObject* _dd_qualified(x0=config, x1=device_name, x2=quantization,
//                          x3=context, x4=max_batch) — all BORROWED
// x19 config b  x20 device b  x21 quant b  x22 context b  x23 max_batch b
// x24 cur operand NEW  [96] geometry NEW  [104] key/temp  [112] it/items
// [120] temp park  [128] item tuple  [136] val NEW  [152] get-m
// ============================================================
_dd_qualified:
    stp   x29, x30, [sp, #-192]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    stp   xzr, xzr, [sp, #120]
    stp   xzr, xzr, [sp, #128]
    str   xzr, [sp, #136]
    str   xzr, [sp, #152]
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, xzr
    mov   x25, xzr

    // ---- geometry = dict(...) ----
    bl    _t_dict_new
    cbz   x0, Lmp_err
    str   x0, [sp, #96]
    DSET k_model_type, #104
    DSET k_hidden, #112
    DSET k_inter, #120
    DSET k_layers, #128
    DSET k_heads, #128
    DSET k_kvheads, #128
    DSET k_experts, #136
    DSET k_epertok, #144
    DSET k_vocab, #152
    DSET k_normtop, #88

    // ---- op1: device_name == 'NVIDIA A2' (RCB EQ=2) ----
    mov   x0, x20
    ILOAD #160
    mov   x1, x0
    mov   x0, x20
    mov   x2, #2
    bl    _t_rcb
    cbz   x0, Lmp_err
    str   x0, [sp, #120]                   // park op result
    ldr   x0, [sp, #120]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop120_err
    cmp   w0, #1
    b.ne  Lq_ret120                          // falsy -> return the operand

    // ---- op2: quantization == 'q8_0' ----
    mov   x0, x21
    ILOAD #168
    mov   x1, x0
    mov   x0, x21
    mov   x2, #2
    bl    _t_rcb
    cbz   x0, Lmp_drop120_err
    ldr   x9, [sp, #120]
    str   x0, [sp, #120]                    // new operand parked
    mov   x0, x9
    bl    _t_decref                         // drop old truthy operand
    ldr   x0, [sp, #120]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop120_err
    cmp   w0, #1
    b.ne  Lq_ret120

    // ---- op3: 1 <= context <= 2048 ----
    ILOAD #184
    mov   x1, x22
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lmp_drop120_err
    str   x0, [sp, #104]                   // park left result
    ldr   x0, [sp, #104]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop_all_err
    cmp   w0, #1
    b.ne  Lq_ret_all
    mov   x0, x22
    ILOAD #112
    mov   x1, x0
    mov   x0, x22
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lmp_drop_all_err
    ldr   x9, [sp, #104]
    str   x0, [sp, #120]                    // park new operand
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop120_err
    cmp   w0, #1
    b.ne  Lq_ret120

    // ---- op4: 1 <= max_batch <= 4 ----
    ILOAD #184
    mov   x1, x23
    mov   x2, #1
    bl    _t_rcb
    cbz   x0, Lmp_drop120_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #104]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop_all_err
    cmp   w0, #1
    b.ne  Lq_ret_all
    mov   x0, x23
    ILOAD #200
    mov   x1, x0
    mov   x0, x23
    mov   x2, #1
    bl    _t_rcb
    cbz   x0, Lmp_drop_all_err
    ldr   x9, [sp, #104]
    str   x0, [sp, #120]
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop120_err
    cmp   w0, #1
    b.ne  Lq_ret120

    // ---- op5: all(config.get(key) == value for key, value in geometry.items()) ----
    // items = geometry.items()
    ldr   x0, [sp, #96]
    adrp  x1, a_items@PAGE
    add   x1, x1, a_items@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop_all_err
    str   x0, [sp, #112]                    // items m
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lmp_drop112_err
    str   x0, [sp, #136]
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    cbz   x0, Lmp_drop136_err
    mov   x24, x0                          // view → callee-saved temp
    ldr   x0, [sp, #136]
    bl    _t_decref                        // drop the empty args tuple
    str   xzr, [sp, #136]
    ldr   x0, [sp, #112]
    bl    _t_decref                        // drop the bound method m
    str   xzr, [sp, #112]
    str   x24, [sp, #112]                  // items view
    ldr   x0, [sp, #112]
    // it = iter(items view)  — views are NOT iterators
    ldr   x0, [sp, #112]
    bl    _t_getiter
    cbz   x0, Lmp_drop112_err
    str   x0, [sp, #128]                    // iterator NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
Lq_items_loop:
    ldr   x0, [sp, #128]
    bl    _t_iternext
    cbz   x0, Lq_items_end
    str   x0, [sp, #136]                    // item tuple park
    // key = item[0]; val = item[1]
    ldr   x0, [sp, #136]
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lmp_drop136_err
    str   x0, [sp, #104]                    // key park
    ldr   x0, [sp, #136]
    mov   x1, #1
    bl    _t_seq_getitem
    cbz   x0, Lmp_drop104_err
    str   x0, [sp, #112]                    // val park
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // g = config.get (fetched per iteration, as the genexp body does)
    mov   x0, x19
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop104_err
    str   x0, [sp, #152]                    // get-m
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmp_drop152_err
    str   x0, [sp, #136]
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    ldr   x2, [sp, #104]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #152]
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]                    // g result park
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x0, [sp, #120]
    cbz   x0, Lmp_drop104_err
    // compare g == val
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #112]
    mov   x2, #2                            // EQ
    bl    _t_rcb
    cbz   x0, Lmp_drop_all_err
    str   x0, [sp, #136]                    // cmp result park
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #136]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lmp_drop136_err
    cmp   w0, #1
    b.eq  Lq_items_loop                     // keep going
    // falsy member: all(...) -> False
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ILOAD #88                               // False singleton
    bl    _incref
    b     Lmp_out
Lq_items_end:
    // iterator exhausted with no error -> all(...) is True
    ldr   x0, [sp, #128]
    cbz   x0, 8f
    bl    _t_decref
    str   xzr, [sp, #128]
8:  ldr   x0, [sp, #120]
    cbz   x0, 9f
    bl    _t_decref
    str   xzr, [sp, #120]
9:  nop

    // ---- op6: config.get('clip_qkv') is None ----
    mov   x0, x19
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmp_drop96_err
    str   x0, [sp, #112]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmp_drop112_err
    str   x0, [sp, #104]
    ILOAD #176                              // 'clip_qkv'
    bl    _incref
    mov   x2, x0                           // x2 = value
    ldr   x0, [sp, #104]                    // x0 = args tuple (AFTER value load)
    mov   x1, #0
    bl    _t_tuple_setitem
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]                    // g result
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    // g (config.get('clip_qkv')) — park in x24: the ILOAD macro's adrp
    // scratch uses x9, so holding g in a caller-saved reg across ILOAD
    // silently replaces it with &g_iface and Py_DecRef then corrupts the
    // interface table pointer itself (war story 49)
    ldr   x24, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x24, Lmp_drop96_err
    ILOAD #96                               // None
    cmp   x24, x0
    b.eq  Lq_ret_true
    // not None: result is the False singleton
    mov   x0, x24
    bl    _t_decref
    ILOAD #88                               // False
    bl    _incref
    mov   x24, x0                           // park result (callee-saved)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x24
    b     Lmp_out

Lq_ret_true:
    ILOAD #80                               // True
    bl    _incref
    mov   x24, x0                           // park result (callee-saved)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x24
    b     Lmp_out

// return the parked operand at [sp,#120] (falsy and-chain exit)
Lq_ret120:
    ldr   x24, [sp, #120]                   // park operand in x24 (callee-saved)
    str   xzr, [sp, #120]
    // drop geometry + everything else still parked
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x0, x24
    b     Lmp_out
Lq_ret_all:
    // falsy at a chained-compare exit: [104] holds the falsy operand
    ldr   x24, [sp, #104]                   // park operand in x24 (callee-saved)
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x0, x24
    b     Lmp_out

// ---- error sweeps ----
Lmp_drop136_err:
    ldr   x0, [sp, #136]
    cbz   x0, Lmp_drop_all_err
    bl    _t_decref
    str   xzr, [sp, #136]
    b     Lmp_drop_all_err
Lmp_drop152_err:
    ldr   x0, [sp, #152]
    cbz   x0, Lmp_drop_all_err
    bl    _t_decref
    str   xzr, [sp, #152]
    b     Lmp_drop_all_err
Lmp_drop104_err:
    ldr   x0, [sp, #104]
    cbz   x0, Lmp_drop112_err
    bl    _t_decref
    str   xzr, [sp, #104]
Lmp_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, Lmp_drop120_err
    bl    _t_decref
    str   xzr, [sp, #112]
Lmp_drop120_err:
    ldr   x0, [sp, #120]
    cbz   x0, Lmp_drop_all_err
    bl    _t_decref
    str   xzr, [sp, #120]
Lmp_drop_all_err:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  ldr   x0, [sp, #104]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #104]
2:  ldr   x0, [sp, #112]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #112]
3:  ldr   x0, [sp, #120]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #120]
4:  ldr   x0, [sp, #128]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #128]
5:  ldr   x0, [sp, #136]
    cbz   x0, 6f
    bl    _t_decref
    str   xzr, [sp, #136]
6:  ldr   x0, [sp, #152]
    cbz   x0, 7f
    bl    _t_decref
    str   xzr, [sp, #152]
7:  mov   x0, xzr
    b     Lmp_out
Lmp_drop96_err:
    ldr   x0, [sp, #96]
    cbz   x0, Lmp_err
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lmp_err
Lmp_err:
    mov   x0, xzr                           // NULL + pending exception
Lmp_out:
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #192
    ret

// ============================================================
// PyObject* _dd_grouped(x0=qualified, x1=tokens) — BORROWED
// return qualified and (1 <= tokens <= 128)
// ============================================================
_dd_grouped:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   xzr, xzr, [sp, #64]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr

    // if not qualified: return qualified
    mov   x0, x19
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lg_err
    cmp   w0, #1
    b.ne  Lg_ret_qual
    // 1 <= tokens
    ILOAD #184
    mov   x1, x20
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lg_err
    str   x0, [sp, #64]                     // park
    ldr   x0, [sp, #64]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lg_drop64_err
    cmp   w0, #1
    b.ne  Lg_ret64
    // tokens <= 128
    mov   x0, x20
    ILOAD #216
    mov   x1, x0
    mov   x0, x20
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lg_drop64_err
    ldr   x9, [sp, #64]
    str   x0, [sp, #64]
    mov   x0, x9
    bl    _t_decref
    ldr   x0, [sp, #64]
    b     Lg_out

Lg_ret_qual:
    mov   x0, x19
    bl    _incref
    b     Lg_out
Lg_ret64:
    ldr   x0, [sp, #64]
    str   xzr, [sp, #64]
    b     Lg_out
Lg_drop64_err:
    ldr   x0, [sp, #64]
    cbz   x0, Lg_err
    bl    _t_decref
    str   xzr, [sp, #64]
Lg_err:
    mov   x0, xzr
Lg_out:
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// PyObject* _dd_packed(x0=qualified, x1=lengths) — BORROWED
// return qualified and 2 <= len(lengths) <= 4
//        and all(length > 0 for length in lengths) and sum(lengths) <= 128
// x19 qualified b  x20 lengths b  x21 acc NEW  x22 item NEW
// parks: [96] it NEW  [104] park  [112] park
// ============================================================
_dd_packed:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #104]
    stp   xzr, xzr, [sp, #112]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr

    // if not qualified: return qualified
    mov   x0, x19
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lp_err
    cmp   w0, #1
    b.ne  Lp_ret_qual

    // n = len(lengths) — scalar kept in x23 (NOT in any drop chain)
    mov   x0, x20
    bl    _t_size
    cmn   x0, #1
    b.eq  Lp_err
    mov   x23, x0                           // n raw
    // nobj = int(n)
    mov   x0, x23
    bl    _t_fromssize
    cbz   x0, Lp_err
    str   x0, [sp, #104]                    // nobj NEW

    // r1 = 2 <= nobj
    ILOAD #192
    ldr   x1, [sp, #104]
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lp_drop104_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lp_drop112_err
    cmp   w0, #1
    b.ne  Lp_ret112                          // falsy operand

    // r2 = nobj <= 4
    ldr   x0, [sp, #104]
    ILOAD #200
    mov   x1, x0
    ldr   x0, [sp, #104]
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lp_drop_all_err
    ldr   x9, [sp, #112]
    str   x0, [sp, #112]
    mov   x0, x9
    bl    _t_decref
    ldr   x0, [sp, #112]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lp_drop_all_err
    cmp   w0, #1
    b.ne  Lp_ret112

    // drop nobj
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]

    // ---- all(length > 0 for length in lengths) ----
    mov   x0, x20
    bl    _t_getiter
    cbz   x0, Lp_err
    str   x0, [sp, #96]
Lp_all_loop:
    ldr   x0, [sp, #96]
    bl    _t_iternext
    cbz   x0, Lp_all_end
    mov   x22, x0                           // item NEW
    // r = item > 0
    mov   x0, x22
    ILOAD #208
    mov   x1, x0
    mov   x0, x22
    mov   x2, #4                            // GT
    bl    _t_rcb
    cbz   x0, Lp_drop22_err
    ldr   x9, [sp, #112]
    str   x0, [sp, #112]
    mov   x0, x9
    bl    _t_decref
    ldr   x0, [sp, #112]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lp_drop_all_err
    cmp   w0, #1
    b.eq  Lp_all_loop                       // truthy -> continue
    // falsy member: all() -> False; return the False singleton
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ILOAD #88                               // False
    bl    _incref
    b     Lp_out
Lp_all_end:
    // exhausted -> all() is True
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]

    // ---- sum(lengths) ----
    ILOAD #208                              // int 0
    bl    _incref
    mov   x21, x0                           // acc NEW
    mov   x0, x20
    bl    _t_getiter
    cbz   x0, Lp_drop21_err
    str   x0, [sp, #96]
Lp_sum_loop:
    ldr   x0, [sp, #96]
    bl    _t_iternext
    cbz   x0, Lp_sum_end
    mov   x22, x0                           // item NEW
    mov   x0, x21
    mov   x1, x22
    bl    _t_num_add
    str   x0, [sp, #112]
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    ldr   x0, [sp, #112]
    cbz   x0, Lp_err
    ldr   x21, [sp, #112]
    str   xzr, [sp, #112]
    b     Lp_sum_loop
Lp_sum_end:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]

    // ---- acc <= 128 ----
    mov   x0, x21
    ILOAD #216
    mov   x1, x0
    mov   x0, x21
    mov   x2, #1                            // LE
    bl    _t_rcb
    cbz   x0, Lp_drop21_err
    str   x0, [sp, #112]
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    ldr   x0, [sp, #112]
    b     Lp_out

// ---- returns / sweeps ----
Lp_ret_qual:
    mov   x0, x19
    bl    _incref
    b     Lp_out
Lp_ret112:
    ldr   x0, [sp, #112]
    str   xzr, [sp, #112]
    ldr   x9, [sp, #96]
    cbz   x9, 1f
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #96]
1:  b     Lp_out
Lp_drop22_err:
    cbz   x22, Lp_drop_all_err
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lp_drop21_err:
    cbz   x21, Lp_drop_all_err
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lp_drop104_err:
    ldr   x0, [sp, #104]
    cbz   x0, Lp_drop112_err
    bl    _t_decref
    str   xzr, [sp, #104]
Lp_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, Lp_drop_all_err
    bl    _t_decref
    str   xzr, [sp, #112]
Lp_drop_all_err:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  ldr   x0, [sp, #104]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #104]
2:  ldr   x0, [sp, #112]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #112]
3:  cbz   x22, 4f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
4:  cbz   x21, 5f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
5:  mov   x0, xzr                           // NULL + pending exception
    b     Lp_out
Lp_err:
    mov   x0, xzr
Lp_out:
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret
