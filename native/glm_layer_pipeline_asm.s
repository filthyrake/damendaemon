// damendaemon/glm_layer_pipeline.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 42/370. 100% of the module:
//     from .layer_pipeline import CpuLayerPipeline, LayerPipelineState as GlmPipelineState
//     from .glm_layer_stage import GlmLayerStage
//     class GlmLayerPipeline(CpuLayerPipeline):
//         stage_type = GlmLayerStage
//         def _stage_batch(self,state): return state.layers[0].attention.latents.shape[0]
//         def cache_bytes_per_row(self,capacity):
//             return 4*capacity*sum(block.attention.rank+block.attention.rope_dim
//                 +(block.attention.indexer.width if block.attention.indexer is not None else 0)
//                 for stage in self.stages for block in stage.blocks)
// Fidelity notes:
//   - real class machinery: _calc_meta + __prepare__ + ns population
//     (__module__, __qualname__, stage_type, _stage_batch, cache_bytes_per_row)
//     then meta(name, bases, ns). No super()/__init__ (inherited).
//   - the sum(...) generator expression is evaluated with explicit
//     GetIter/IterNext loops and PyNumber_Add accumulation from int 0
//     (value-identical to CPython's sum builtin over the generator).
//   - the ternary fetches block.attention.indexer TWICE, as CPython does
//     (condition + value expressions are separate loads).
//   - 4*capacity first, then *total: PyNumber_Multiply in source order.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear        104 type           112 CpuLayerPipeline
//   120 GlmLayerStage      128 None           136 int 0
//   144 int 4              152 PyNumber_Multiply 160 PyNumber_Add
//   168 PyObject_GetIter   176 PyIter_Next     184 _stage_batch shell
//   192 _cache_bytes shell 200 __prepare__ (string) 208 TypeError
//   216 PyErr_Occurred

.globl _dd_init
.globl _dd_classes
.globl _dd_stage_batch
.globl _dd_cache_bytes

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
s_prepare:  .asciz "__prepare__"
s_conflict: .asciz "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases"
k_module:   .asciz "__module__"
k_qualname: .asciz "__qualname__"
k_stage_batch: .asciz "_stage_batch"
k_cbpr:     .asciz "cache_bytes_per_row"
k_stage_type: .asciz "stage_type"
a_layers:   .asciz "layers"
a_attention: .asciz "attention"
a_latents:  .asciz "latents"
a_shape:    .asciz "shape"
a_blocks:   .asciz "blocks"
a_rank:     .asciz "rank"
a_rope_dim: .asciz "rope_dim"
a_indexer:  .asciz "indexer"
a_width:    .asciz "width"
a_stages:   .asciz "stages"
cn_0:       .asciz "GlmLayerPipeline"

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
_t_num_mul:        IFACE #152
_t_num_add:        IFACE #160
_t_getiter:        IFACE #168
_t_iternext:       IFACE #176
_t_err_occurred:    IFACE #216

_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// _calc_meta(x0 = bases tuple BORROWED) -> x0 meta OWNED (NULL+exc)
// CPython _PyType_CalculateMetaclass (see glm_flash_ffn_asm.s).
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
    ldr   x0, [x10, #208]                   // TypeError
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
// PyObject* _dd_stage_batch(x0=self, x1=state) — BORROWED
// return state.layers[0].attention.latents.shape[0]
// x19 self b  x20 state b  x21 cur NEW  x22 next NEW  x23 (unused)
// Single live chain: each step's result is the only owned ref.
// ============================================================
_dd_stage_batch:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   xzr, xzr, [sp, #64]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr

    // layers = state.layers
    mov   x0, x20
    adrp  x1, a_layers@PAGE
    add   x1, x1, a_layers@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsb_err
    mov   x21, x0                           // layers NEW
    // b0 = layers[0]
    mov   x0, x21
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lsb_err
    mov   x22, x0                           // b0 NEW
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    // att = b0.attention
    mov   x0, x22
    adrp  x1, a_attention@PAGE
    add   x1, x1, a_attention@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsb_drop22_err
    mov   x21, x0                           // att NEW
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    // lat = att.latents
    mov   x0, x21
    adrp  x1, a_latents@PAGE
    add   x1, x1, a_latents@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsb_drop21_err
    mov   x22, x0                           // lat NEW
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    // shp = lat.shape
    mov   x0, x22
    adrp  x1, a_shape@PAGE
    add   x1, x1, a_shape@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsb_drop22_err
    mov   x21, x0                           // shp NEW
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    // r = shp[0]
    mov   x0, x21
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lsb_drop21_err
    mov   x22, x0                           // result NEW
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x22
    b     Lsb_out

Lsb_drop22_err:
    cbz   x22, Lsb_drop21_err
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lsb_drop21_err:
    cbz   x21, Lsb_err
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lsb_err:
    mov   x0, xzr                           // NULL + pending exception
Lsb_out:
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// PyObject* _dd_cache_bytes(x0=self, x1=capacity) — BORROWED
// return 4*capacity*sum(genexp)
// x19 self b  x20 capacity b
// x21 cap4 NEW  x22 acc NEW  x23 stage NEW  x24 block NEW
// x25 att NEW  x26 terms/idx NEW  x27 s1 NEW
// parks: [96] stage_it NEW  [104] block_it NEW  [112] temp park
// ============================================================
_dd_cache_bytes:
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
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr
    mov   x26, xzr
    mov   x27, xzr
    mov   x28, xzr                           // no object use; free for any

    // ---- cap4 = 4 * capacity ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                    // int 4 (borrowed)
    bl    _incref
    str   x0, [sp, #112]                     // four NEW
    ldr   x0, [sp, #112]
    mov   x1, x20
    bl    _t_num_mul
    str   x0, [sp, #112]                     // park across cleanup
    ldr   x0, [sp, #112]
    cbz   x0, Lcb_err
    ldr   x21, [sp, #112]                    // cap4 NEW
    str   xzr, [sp, #112]

    // ---- acc = 0 (owned copy) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                    // int 0 (borrowed)
    bl    _incref
    mov   x22, x0                            // acc NEW

    // ---- stages = self.stages; it1 = iter(stages) ----
    mov   x0, x19
    adrp  x1, a_stages@PAGE
    add   x1, x1, a_stages@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_err
    str   x0, [sp, #112]                     // stages NEW
    ldr   x0, [sp, #112]
    bl    _t_getiter
    cbz   x0, Lcb_drop112_err
    str   x0, [sp, #96]                      // it1 NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

Lcb_stage_loop:
    ldr   x0, [sp, #96]
    bl    _t_iternext
    str   x0, [sp, #112]                     // park stage candidate
    cbnz  x0, Lcb_have_stage
    // NULL: exhausted or error?
    bl    _t_err_occurred
    cbnz  x0, Lcb_drop96_err                  // error -> propagate
    b     Lcb_stage_done                      // exhausted
Lcb_have_stage:
    ldr   x23, [sp, #112]                    // stage NEW
    str   xzr, [sp, #112]

    // blocks = stage.blocks; it2 = iter(blocks)
    mov   x0, x23
    adrp  x1, a_blocks@PAGE
    add   x1, x1, a_blocks@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop23_err
    str   x0, [sp, #112]                     // blocks NEW
    ldr   x0, [sp, #112]
    bl    _t_getiter
    cbz   x0, Lcb_drop112_err
    str   x0, [sp, #104]                     // it2 NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

Lcb_block_loop:
    ldr   x0, [sp, #104]
    bl    _t_iternext
    str   x0, [sp, #112]
    cbnz  x0, Lcb_have_block
    bl    _t_err_occurred
    cbnz  x0, Lcb_drop_all_err
    b     Lcb_block_done
Lcb_have_block:
    ldr   x24, [sp, #112]                    // block NEW
    str   xzr, [sp, #112]

    // att = block.attention
    mov   x0, x24
    adrp  x1, a_attention@PAGE
    add   x1, x1, a_attention@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop24_err
    mov   x25, x0                           // att NEW

    // rank = att.rank
    mov   x0, x25
    adrp  x1, a_rank@PAGE
    add   x1, x1, a_rank@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop25_err
    mov   x26, x0                           // rank NEW

    // rope = att.rope_dim
    mov   x0, x25
    adrp  x1, a_rope_dim@PAGE
    add   x1, x1, a_rope_dim@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop26_err
    mov   x27, x0                           // rope NEW

    // s = rank + rope   (Add borrows both, returns NEW)
    mov   x0, x26
    mov   x1, x27
    bl    _t_num_add
    str   x0, [sp, #112]                    // park s
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x26, [sp, #112]                   // s NEW (x26)
    str   xzr, [sp, #112]
    cbz   x26, Lcb_err

    // idx = att.indexer  (fetch #1: condition)
    mov   x0, x25
    adrp  x1, a_indexer@PAGE
    add   x1, x1, a_indexer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop26_err
    str   x0, [sp, #112]                    // idx1 NEW
    // idx1 is not None?
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #128]                    // None
    ldr   x10, [sp, #112]
    cmp   x10, x9
    b.eq  Lcb_no_indexer
    // drop idx1 (condition ref; CPython discards it after the test)
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

    // idx2 = att.indexer  (fetch #2: value, literal double-fetch)
    mov   x0, x25
    adrp  x1, a_indexer@PAGE
    add   x1, x1, a_indexer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop26_err
    mov   x27, x0                           // idx2 NEW

    // w = idx2.width
    mov   x0, x27
    adrp  x1, a_width@PAGE
    add   x1, x1, a_width@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop2627_err
    str   x0, [sp, #112]                    // w park
    mov   x0, x27
    bl    _t_decref                        // drop idx2
    mov   x27, xzr
    ldr   x27, [sp, #112]                   // w -> x27 (callee-saved)
    str   xzr, [sp, #112]

    // s = s + w
    mov   x0, x26
    mov   x1, x27
    bl    _t_num_add
    str   x0, [sp, #112]                    // park s2
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x26, [sp, #112]                   // s2 NEW (x26)
    str   xzr, [sp, #112]
    b     Lcb_add_acc

Lcb_no_indexer:
    // drop idx1 (the None); s stays in x26
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
Lcb_add_acc:
    // acc = acc + s1/s2 (x26)
    mov   x0, x22
    mov   x1, x26
    bl    _t_num_add
    str   x0, [sp, #112]                    // park acc2
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
    ldr   x0, [sp, #112]
    cbz   x0, Lcb_err
    ldr   x22, [sp, #112]                   // acc NEW
    str   xzr, [sp, #112]
    // drop att, block; next block
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    b     Lcb_block_loop

Lcb_block_done:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    b     Lcb_stage_loop
Lcb_stage_done:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    // result = cap4 * acc
    mov   x0, x21
    mov   x1, x22
    bl    _t_num_mul
    str   x0, [sp, #112]                    // park result
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    ldr   x0, [sp, #112]
    b     Lcb_out

// ---- error sweeps (drop everything that can be live) ----
Lcb_drop24_err:
    cbz   x24, Lcb_drop_all_err
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
Lcb_drop_all_err:
    // note: falls through after any drop-chain entry — full sweep
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
3:  cbz   x26, 4f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
4:  cbz   x27, 5f
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
5:  cbz   x25, 6f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
6:  cbz   x24, 7f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
7:  cbz   x23, 8f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
8:  cbz   x22, 9f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
9:  cbz   x21, 10f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
10: mov   x0, xzr                           // NULL + pending exception
    b     Lcb_out

Lcb_drop23_err:
    cbz   x23, Lcb_err
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    b     Lcb_err
Lcb_drop26_err:
    cbz   x26, Lcb_drop25_err
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lcb_drop2627_err:
    cbz   x27, Lcb_drop25_err
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    b     Lcb_drop25_err
Lcb_drop25_err:
    cbz   x25, Lcb_drop24_err
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    b     Lcb_drop24_err
Lcb_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, Lcb_drop_all_err
    bl    _t_decref
    str   xzr, [sp, #112]
    b     Lcb_drop_all_err
Lcb_drop96_err:
    ldr   x0, [sp, #96]
    cbz   x0, Lcb_err
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lcb_err
Lcb_err:
    mov   x0, xzr                           // NULL + pending exception
Lcb_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_classes(x0 = modname BORROWED) -> tuple of 1 class
// Per class (docless): bases -> _calc_meta -> __prepare__ -> ns
// {__module__, __qualname__, stage_type, _stage_batch,
//  cache_bytes_per_row} -> meta(name, bases, ns).
// x19 meta(owned)  x28 result park (zeroed at entry)
// parks: [96] cls0  [128] bases  [136] name  [144] ns
// [152] prep-args  [160] prepare  [168] modname  [176] meta-args/cls
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
    mov   x19, xzr
    mov   x28, xzr
    str   x0, [sp, #168]

// ===== class 0: GlmLayerPipeline(CpuLayerPipeline) =====
Lcls_0:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lc_err
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]                   // CpuLayerPipeline
    bl    _incref
    ldr   x0, [sp, #128]
    mov   x1, #0
    ldr   x2, [x10, #112]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    bl    _calc_meta
    cbz   x0, Lc_err
    mov   x19, x0
    adrp  x0, cn_0@PAGE
    add   x0, x0, cn_0@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, s_prepare@PAGE
    add   x1, x1, s_prepare@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lc0_prep_fallback
    str   x0, [sp, #160]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lc_err
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
    b     Lc_err
Lc0_prep_fallback:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Lc_err
    str   x0, [sp, #144]
Lc0_ns_have:
    // __module__
    adrp  x0, k_module@PAGE
    add   x0, x0, k_module@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
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
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // stage_type = GlmLayerStage
    adrp  x0, k_stage_type@PAGE
    add   x0, x0, k_stage_type@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [x10, #120]                   // GlmLayerStage (borrowed ok: no steal)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // _stage_batch (bridge shell)
    adrp  x0, k_stage_batch@PAGE
    add   x0, x0, k_stage_batch@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [x10, #184]                   // _stage_batch (borrowed)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // cache_bytes_per_row (bridge shell)
    adrp  x0, k_cbpr@PAGE
    add   x0, x0, k_cbpr@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [x10, #192]                   // _cache_bytes (borrowed)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // meta(name, bases, ns)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lc_err
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
    b     Lc_err
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

// ---- result tuple of 1 ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lc_err
    str   x0, [sp, #128]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #128]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    ldr   x28, [sp, #128]                   // result park; [128] stays 0
    str   xzr, [sp, #128]

Lc_out:
    ldr   x0, [sp, #176]
    cbz   x0, Lc_o1
    bl    _t_decref
    str   xzr, [sp, #176]
Lc_o1:
    ldr   x0, [sp, #152]
    cbz   x0, Lc_o2
    bl    _t_decref
    str   xzr, [sp, #152]
Lc_o2:
    ldr   x0, [sp, #160]
    cbz   x0, Lc_o3
    bl    _t_decref
    str   xzr, [sp, #160]
Lc_o3:
    ldr   x0, [sp, #144]
    cbz   x0, Lc_o4
    bl    _t_decref
    str   xzr, [sp, #144]
Lc_o4:
    ldr   x0, [sp, #136]
    cbz   x0, Lc_o5
    bl    _t_decref
    str   xzr, [sp, #136]
Lc_o5:
    ldr   x0, [sp, #128]
    cbz   x0, Lc_o6
    bl    _t_decref
    str   xzr, [sp, #128]
Lc_o6:
    cbz   x19, Lc_o7
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
Lc_o7:
    ldr   x0, [sp, #96]
    cbz   x0, Lc_o8
    bl    _t_decref
    str   xzr, [sp, #96]
Lc_o8:
    mov   x0, x28
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #224
    ret

Lc_err:
    ldr   x0, [sp, #176]
    cbz   x0, Lc_e1
    bl    _t_decref
    str   xzr, [sp, #176]
Lc_e1:
    ldr   x0, [sp, #152]
    cbz   x0, Lc_e2
    bl    _t_decref
    str   xzr, [sp, #152]
Lc_e2:
    ldr   x0, [sp, #160]
    cbz   x0, Lc_e3
    bl    _t_decref
    str   xzr, [sp, #160]
Lc_e3:
    ldr   x0, [sp, #144]
    cbz   x0, Lc_e4
    bl    _t_decref
    str   xzr, [sp, #144]
Lc_e4:
    ldr   x0, [sp, #136]
    cbz   x0, Lc_e5
    bl    _t_decref
    str   xzr, [sp, #136]
Lc_e5:
    ldr   x0, [sp, #128]
    cbz   x0, Lc_e6
    bl    _t_decref
    str   xzr, [sp, #128]
Lc_e6:
    cbz   x19, Lc_e7
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
Lc_e7:
    ldr   x0, [sp, #96]
    cbz   x0, Lc_e8
    bl    _t_decref
    str   xzr, [sp, #96]
Lc_e8:
    mov   x0, xzr                           // NULL + pending exception
    b     Lc_o7
