// damendaemon/bytelevel_stream.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 6/370. 100% of the module: the BYTE_VALUES
// alphabet (dict built in the original's exact insertion order), the
// ByteLevelStream constructor validation/state, feed (suppression,
// added-token dispatch, special/literal passthrough with incremental
// decoder flush, per-char alphabet mapping with UTF-8 fallback), and
// finish.
//
// ABI table (offsets in bytes), installed via dd_init:
//    0 PyObject_GetAttrString     8 PyObject_SetAttrString
//   16 PyObject_Call             24 PyTuple_New
//   32 PyTuple_SetItem           40 Py_DecRef
//   48 PyDict_New                56 PyDict_SetItem        (no steal)
//   64 PyDict_GetItemWithError   72 PyDict_Contains
//   80 PySequence_Contains       88 PyObject_IsTrue
//   96 PyErr_Clear              104 PyErr_SetString
//  112 PyExc_ValueError         120 None
//  128 PyUnicode_FromOrdinal    136 PyUnicode_AsUTF8String
//  144 PyBytes_FromStringAndSize 152 PyLong_AsLongLong
//  160 PyUnicode_Concat         168 &PyUnicode_Type
//  176 PyUnicode_GetLength      184 PyUnicode_ReadChar
//  192 PyLong_FromLongLong      200 codecs.getincrementaldecoder
//  208 frozenset                216 is_bytelevel (asm module 2)
//  224 'utf-8'                  232 'replace'
//  240 'errors'                 248 'final'
//  256 Py_True                  264 '' empty-string singleton
//  272 () empty-tuple default   280 {} empty-dict default
//  288 Py_False                 296 PyDict_SetItemString
//  304 PyObject_IsInstance
//
// Alphabet fast path: g_table[ord] maps each ByteLevel alphabet char's
// ordinal (0..255 visible, 256..323 invisible) to its byte value,
// filled at init alongside the dict; feed reads chars via
// PyUnicode_ReadChar with an arithmetic validity check.

.globl _dd_init
.globl _dd_byte_values
.globl _dd_stream_init
.globl _dd_feed
.globl _dd_finish

.section __DATA,__data
g_iface:  .quad 0
g_values: .quad 0
g_table:  .space 324          // ord -> byte value (0..323)

.section __TEXT,__cstring
a_decoder:  .asciz "decoder"
a_tokenizer:.asciz "tokenizer"
a_suppress: .asciz "suppressed_token_ids"
a_added:    .asciz "added"
a_allowed:  .asciz "allowed"
a_utf8:     .asciz "utf8"
a_getadded: .asciz "get_added_tokens_decoder"
a_content:  .asciz "content"
a_special:  .asciz "special"
a_literal:  .asciz "_literal_tokens"
a_id2tok:   .asciz "id_to_token"
a_decode:   .asciz "decode"
a_reset:    .asciz "reset"
m_notbl:    .asciz "incremental output requires a ByteLevel tokenizer"
m_control:  .asciz "unsupported generated added/control token"
m_vocab:    .asciz "invalid ByteLevel vocabulary token"
k_errors:   .asciz "errors"
k_final:    .asciz "final"
v_utf8:     .asciz "utf-8"
v_replace:  .asciz "replace"
c_empty:    .asciz ""

.text

// tail-call trampolines
_t_getattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #0]
    br    x9
_t_setattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_call:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_dict_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_dict_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_dict_getwe:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_dict_contains:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_seq_contains:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_err_clear:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #96]
    br    x9
_t_fromordinal:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #128]
    br    x9
_t_utf8str:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #136]
    br    x9
_t_bytes_fsas:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #144]
    br    x9
_t_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #160]
    br    x9
_t_uni_length:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #176]
    br    x9
_t_readchar:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #184]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #192]
    br    x9
_t_dict_setstr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #296]
    br    x9
_t_isinstance:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #304]
    br    x9

// _raise_value(x0 = C-string) — set ValueError, return
_raise_value:
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    mov   x1, x0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #112]                   // PyExc_ValueError
    ldr   x9, [x9, #104]                   // PyErr_SetString
    blr   x9
    ldp   x29, x30, [sp], #16
    ret

// helper: INCREF x0 (mortal)
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// void _dd_init(x0 = iface) — install table, build the alphabet:
//   visible = 33..126, 161..172, 174..255
//   pass 1: dict[chr(v)] = v for visible v (ascending)
//   pass 2: extra = 256; for v ascending, invisible:
//           dict[chr(extra)] = v; extra++
//   g_table[ord] = byte value in both passes
// Insertion order must match the Python original exactly (tests
// enumerate the dict).
// ============================================================
_dd_init:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 dict   x20 v   x21 extra   x22 key
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Li9
    mov   x19, x0

    // ---- pass 1: visible values ascending ----
    mov   x20, #0
Li1:
    cmp   x20, #256
    b.hs  Li2
    bl    _vis_check                       // w0 = visible(v)?
    cbz   w0, Li1_next
    // g_table[v] = v
    adrp  x9, g_table@PAGE
    add   x9, x9, g_table@PAGEOFF
    strb  w20, [x9, x20]
    mov   x0, x20                          // key ordinal = v
    bl    _alphabet_insert                 // dict[chr(v)] = v
    cbnz  w0, Li_err
Li1_next:
    add   x20, x20, #1
    b     Li1

    // ---- pass 2: invisible values, extra from 256 ----
Li2:
    mov   x20, #0
    mov   x21, #256
Li3:
    cmp   x20, #256
    b.hs  Li4
    bl    _vis_check
    cbnz  w0, Li3_next
    // g_table[extra] = v
    adrp  x9, g_table@PAGE
    add   x9, x9, g_table@PAGEOFF
    strb  w20, [x9, x21]
    mov   x0, x21                          // key ordinal = extra
    bl    _alphabet_insert                 // dict[chr(extra)] = v
    cbnz  w0, Li_err
    add   x21, x21, #1                     // extra++
Li3_next:
    add   x20, x20, #1
    b     Li3

Li4:
    adrp  x9, g_values@PAGE
    str   x19, [x9, g_values@PAGEOFF]
    b     Li9
Li_err:
    mov   x0, x19
    bl    _t_decref
Li9:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret

// _vis_check — w0 = is x20 visible (33..126, 161..172, >=174)?
// clobbers w9 only; x20 untouched.
_vis_check:
    mov   w0, #0
    cmp   x20, #33
    b.lo  9f
    cmp   x20, #126
    b.ls  8f
    cmp   x20, #161
    b.lo  9f
    cmp   x20, #172
    b.ls  8f
    cmp   x20, #174
    b.lo  9f
8:  mov   w0, #1
9:  ret

// _alphabet_insert — x0 = key ordinal, x20 = value; builds
// key = PyUnicode_FromOrdinal(x0), val = PyLong(v), inserts into
// x19 dict, drops key+val. Returns w0 = 0 ok / nonzero error.
// Frame: [16]=saved x21  [24]=val  [32]=rc (survives decref calls)
_alphabet_insert:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    str   x21, [sp, #16]                   // preserve caller's x21
    mov   x21, x0                          // key ordinal
    mov   x0, x20
    bl    _t_fromlong
    cbz   x0, Lai_err
    str   x0, [sp, #24]                     // val parked
    mov   x0, x21
    bl    _t_fromordinal
    cbz   x0, Lai_err_val
    mov   x21, x0                          // key
    mov   x0, x19
    mov   x1, x21
    ldr   x2, [sp, #24]
    bl    _t_dict_setitem
    str   x0, [sp, #32]                     // rc parked (decrefs clobber x9)
    mov   x0, x21
    bl    _t_decref                        // drop key
    ldr   x0, [sp, #24]
    bl    _t_decref                        // drop val
    ldr   x0, [sp, #32]
    b     Lai_out
Lai_err_val:
    ldr   x0, [sp, #24]
    bl    _t_decref
Lai_err:
    mov   x0, #-1
Lai_out:
    ldr   x21, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// helper _decode_call(x0 = utf8 obj BORROWED, x1 = raw bytes
//                     BORROWED or NULL, x2 = final 0/1) -> str
//   decode = utf8.decode
//   args = (raw,) or (b'',) ; kwargs = {'final': final}
// ============================================================
_decode_call:
    stp   x29, x30, [sp, #-80]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    // x19 utf8   x20 raw|NULL   x21 final   x22 method   x23 args
    // x24 kwargs
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x0, x19
    adrp  x1, a_decode@PAGE
    add   x1, x1, a_decode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldc_ret0
    mov   x22, x0
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ldc_drop_meth
    mov   x23, x0
    cbz   x20, 1f
    mov   x0, x20
    bl    _incref                          // tuple steals
    mov   x0, x23
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    b     2f
1:  // empty bytes
    adrp  x0, c_empty@PAGE
    add   x0, x0, c_empty@PAGEOFF
    mov   x1, #0
    bl    _t_bytes_fsas
    cbz   x0, Ldc_drop_args
    mov   x2, x0                          // item (stolen below)
    mov   x0, x23
    mov   x1, #0
    bl    _t_tuple_setitem
2:
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Ldc_drop_args
    mov   x24, x0
    // kwargs['final'] = True/False
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #256]                  // Py_True (VALUE — x2!)
    cmp   x21, #0
    b.ne  3f
    ldr   x2, [x9, #288]                   // Py_False
3:
    mov   x0, x24
    adrp  x1, k_final@PAGE
    add   x1, x1, k_final@PAGEOFF
    bl    _t_dict_setstr
    mov   x0, x22
    mov   x1, x23
    mov   x2, x24
    bl    _t_call
    str   x0, [x29, #64]                   // park result
    mov   x0, x23
    bl    _t_decref
    mov   x0, x24
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    ldr   x0, [x29, #64]
    b     Ldc_ret
Ldc_drop_args:
    mov   x0, x23
    bl    _t_decref
Ldc_drop_meth:
    mov   x0, x22
    bl    _t_decref
Ldc_ret0:
    mov   x0, #0
Ldc_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #80
    ret

// ============================================================
// PyObject* _dd_byte_values() — module-level BYTE_VALUES dict
// ============================================================
_dd_byte_values:
    adrp  x9, g_values@PAGE
    ldr   x9, [x9, g_values@PAGEOFF]
    ldr   x10, [x9]
    add   x10, x10, #1
    str   x10, [x9]
    mov   x0, x9
    ret

// ============================================================
// PyObject* _dd_stream_init(x0=self, x1=tokenizer, x2=allowed)
//   -> None or NULL (ValueError if not ByteLevel)
// ============================================================
_dd_stream_init:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    // x19 self  x20 tokenizer  x21 allowed (borrowed)
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    // ---- decoder = tokenizer.decoder ----
    mov   x0, x20
    adrp  x1, a_decoder@PAGE
    add   x1, x1, a_decoder@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsi_ret0
    mov   x22, x0
    // ---- ib = is_bytelevel(decoder) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsi_drop_dec
    mov   x23, x0
    mov   x0, x22
    bl    _incref
    mov   x0, x23
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #216]                   // is_bytelevel callable
    mov   x1, x23
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // ib
    mov   x0, x23
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    cbz   x24, Lsi_ret0
    mov   x0, x24
    bl    _t_istrue
    mov   x25, x0                          // rc
    mov   x0, x24
    bl    _t_decref
    cmp   x25, #1
    b.ne  Lsi_notbl
    // ---- self.tokenizer = tokenizer ----
    mov   x0, x19
    adrp  x1, a_tokenizer@PAGE
    add   x1, x1, a_tokenizer@PAGEOFF
    mov   x2, x20
    bl    _t_setattr_s
    cbnz  w0, Lsi_ret0
    // ---- self.added = tokenizer.get_added_tokens_decoder() ----
    mov   x0, x20
    adrp  x1, a_getadded@PAGE
    add   x1, x1, a_getadded@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsi_ret0
    mov   x22, x0
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lsi_drop_meth
    mov   x23, x0
    mov   x0, x22
    mov   x1, x23
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // added
    mov   x0, x23
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    cbz   x24, Lsi_ret0
    mov   x0, x19
    adrp  x1, a_added@PAGE
    add   x1, x1, a_added@PAGEOFF
    mov   x2, x24
    bl    _t_setattr_s
    mov   x25, x0                          // rc
    mov   x0, x24
    bl    _t_decref
    cbnz  x25, Lsi_ret0
    // ---- self.allowed = frozenset(allowed_added) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsi_ret0
    mov   x23, x0
    mov   x0, x21
    bl    _incref
    mov   x0, x23
    mov   x1, #0
    mov   x2, x21
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #208]                   // frozenset
    mov   x1, x23
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // frozenset
    mov   x0, x23
    bl    _t_decref
    cbz   x24, Lsi_ret0
    mov   x0, x19
    adrp  x1, a_allowed@PAGE
    add   x1, x1, a_allowed@PAGEOFF
    mov   x2, x24
    bl    _t_setattr_s
    mov   x25, x0
    mov   x0, x24
    bl    _t_decref
    cbnz  x25, Lsi_ret0
    // ---- cls = codecs.getincrementaldecoder('utf-8') ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsi_ret0
    mov   x23, x0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x22, [x9, #224]                  // 'utf-8'
    mov   x0, x22
    bl    _incref
    mov   x0, x23
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #200]                   // getincrementaldecoder
    mov   x1, x23
    mov   x2, #0
    bl    _t_call
    mov   x24, x0                          // cls
    mov   x0, x23
    bl    _t_decref
    cbz   x24, Lsi_ret0
    // ---- kwargs = {'errors': 'replace'} ----
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Lsi_drop_cls
    mov   x26, x0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #232]                   // 'replace'
    mov   x0, x26
    adrp  x1, k_errors@PAGE
    add   x1, x1, k_errors@PAGEOFF
    bl    _t_dict_setstr
    // ---- self.utf8 = cls(errors='replace') ----
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lsi_drop_kw
    mov   x23, x0
    mov   x0, x24
    mov   x1, x23
    mov   x2, x26
    bl    _t_call
    mov   x25, x0                          // utf8
    mov   x0, x23
    bl    _t_decref
    mov   x0, x26
    bl    _t_decref
    mov   x0, x24
    bl    _t_decref
    cbz   x25, Lsi_ret0
    mov   x0, x19
    adrp  x1, a_utf8@PAGE
    add   x1, x1, a_utf8@PAGEOFF
    mov   x2, x25
    bl    _t_setattr_s
    mov   x26, x0
    mov   x0, x25
    bl    _t_decref
    cbnz  x26, Lsi_ret0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #120]                   // None
    b     Lsi_ret
Lsi_notbl:
    adrp  x0, m_notbl@PAGE
    add   x0, x0, m_notbl@PAGEOFF
    bl    _raise_value
    b     Lsi_ret0
Lsi_drop_kw:
    mov   x0, x26
    bl    _t_decref
Lsi_drop_cls:
    mov   x0, x24
    bl    _t_decref
    b     Lsi_ret0
Lsi_drop_meth:
    mov   x0, x22
    bl    _t_decref
    b     Lsi_ret0
Lsi_drop_dec:
    mov   x0, x22
    bl    _t_decref
Lsi_ret0:
    mov   x0, #0
Lsi_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #112
    ret

// helper _ord_valid(x0 = ordinal) -> w0: alphabet char?
//   ord < 256: visible ranges ; ord in 256..323: extra keys
// PRESERVES x0 (the argument) — the result goes to w0 only at exits.
_ord_valid:
    cmp   x0, #256
    b.hs  1f
    cmp   x0, #33
    b.lo  9f
    cmp   x0, #126
    b.ls  8f
    cmp   x0, #161
    b.lo  9f
    cmp   x0, #172
    b.ls  8f
    cmp   x0, #174
    b.lo  9f
8:  mov   w0, #1
    ret
1:  cmp   x0, #323
    b.hi  9f
    mov   w0, #1
    ret
9:  mov   w0, #0
    ret

// ============================================================
// PyObject* _dd_feed(x0=self, x1=token) -> str or NULL
// ============================================================
_dd_feed:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    // x19 self  x20 token  x21 tok  x22-x28 temps
    // [x29,#104] i / parked  [x29,#112] parked  [x29,#120] parked
    mov   x19, x0
    mov   x20, x1
    // ---- tok = self.tokenizer ----
    mov   x0, x19
    adrp  x1, a_tokenizer@PAGE
    add   x1, x1, a_tokenizer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_ret0
    mov   x21, x0
    // ---- suppressed check ----
    mov   x0, x21
    adrp  x1, a_suppress@PAGE
    add   x1, x1, a_suppress@PAGEOFF
    bl    _t_getattr_s
    cbnz  x0, 1f
    bl    _t_err_clear
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x22, [x9, #272]                  // () default — BORROWED
    mov   x24, #1                          // borrowed marker
    b     2f
1:  mov   x22, x0
    mov   x24, #0                          // owned
2:
    mov   x0, x22
    mov   x1, x20
    bl    _t_seq_contains
    mov   x23, x0                          // rc
    cbnz  x24, 2f                          // borrowed: no decref
    mov   x0, x22
    bl    _t_decref
2:  cmp   x23, #0
    b.le  3f
    // suppressed: drop tok, return ''
    mov   x0, x21
    bl    _t_decref
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #264]                   // ''
    b     Lfd_ret
3:
    cbz   x23, 4f                          // rc==0 fine
    mov   x0, x21                          // rc<0: error
    bl    _t_decref
    b     Lfd_ret0
4:
    // ---- added = self.added ; token in added? ----
    mov   x0, x19
    adrp  x1, a_added@PAGE
    add   x1, x1, a_added@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_tok
    mov   x22, x0
    mov   x0, x22
    mov   x1, x20
    bl    _t_dict_contains
    mov   x23, x0
    cmp   x23, #0
    b.le  Lfd_added_done                    // drop added below
    // ===== added-token path =====
    mov   x0, x22
    mov   x1, x20
    bl    _t_dict_getwe
    cbz   x0, Lfd_drop_added_err
    mov   x24, x0                          // entry (borrowed -> owned)
    bl    _incref
    mov   x0, x22
    bl    _t_decref                        // drop added dict
    // spec = getattr(entry, 'special'); special = truthy(spec)
    mov   x0, x24
    adrp  x1, a_special@PAGE
    add   x1, x1, a_special@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_entry
    mov   x22, x0
    mov   x0, x22
    bl    _t_istrue
    mov   x25, x0                          // special (0/1/-1)
    mov   x0, x22
    bl    _t_decref
    cmp   x25, #0
    b.lt  Lfd_drop_entry
    // value = getattr(entry, 'content')
    mov   x0, x24
    adrp  x1, a_content@PAGE
    add   x1, x1, a_content@PAGEOFF
    bl    _t_getattr_s
    mov   x26, x0                          // value
    mov   x0, x24
    bl    _t_decref                        // drop entry
    cbz   x26, Lfd_drop_tok
    // if special: value must be in self.allowed
    cbz   x25, 5f
    mov   x0, x19
    adrp  x1, a_allowed@PAGE
    add   x1, x1, a_allowed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_value
    mov   x22, x0
    mov   x0, x22
    mov   x1, x26
    bl    _t_seq_contains
    mov   x23, x0
    mov   x0, x22
    bl    _t_decref
    cmp   x23, #1
    b.ne  Lfd_control_bad                  // 0 or error -> reject
5:
    // lit = getattr(tok, '_literal_tokens', {})  — default is BORROWED
    mov   x0, x21
    adrp  x1, a_literal@PAGE
    add   x1, x1, a_literal@PAGEOFF
    bl    _t_getattr_s
    cbnz  x0, 6f
    bl    _t_err_clear
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x22, [x9, #280]                  // {} default — BORROWED
    mov   x24, #1                          // borrowed marker
    b     7f
6:  mov   x22, x0
    mov   x24, #0                          // owned
7:
    mov   x0, x22
    mov   x1, x20
    bl    _t_dict_contains
    mov   x23, x0
    cbnz  x24, 7f                          // borrowed: no decref
    mov   x0, x22
    bl    _t_decref
7:  cmp   x23, #0
    b.lt  Lfd_drop_value
    // if special or rc3 > 0: flush + reset, return decoded + value
    cbnz  x25, Lfd_flush
    cmp   x23, #0
    b.le  Lfd_added_fallthrough           // ordinary added token: id2tok path
Lfd_flush:
    mov   x0, x19
    adrp  x1, a_utf8@PAGE
    add   x1, x1, a_utf8@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_value
    mov   x22, x0                          // utf8
    mov   x0, x22
    mov   x1, #0
    mov   x2, #1
    bl    _decode_call
    mov   x27, x0                          // decoded
    str   x26, [x29, #112]                 // park value (before any check)
    cbz   x27, Lfd_drop_utf8_value
    // utf8.reset()
    mov   x0, x22
    adrp  x1, a_reset@PAGE
    add   x1, x1, a_reset@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_dec_utf8_value
    mov   x24, x0
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfd_drop_rm
    mov   x25, x0
    mov   x0, x24
    mov   x1, x25
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                          // reset result (value parked)
    mov   x0, x25
    bl    _t_decref
    mov   x0, x24
    bl    _t_decref
    mov   x0, x26
    bl    _t_decref
    // result = decoded + value
    mov   x0, x27
    ldr   x1, [x29, #112]
    bl    _t_concat
    str   x0, [x29, #104]                   // park result
    mov   x0, x27
    bl    _t_decref                        // drop decoded
    ldr   x0, [x29, #112]
    bl    _t_decref                        // drop value
    mov   x0, x22
    bl    _t_decref                        // drop utf8
    mov   x0, x21
    bl    _t_decref                        // drop tok
    ldr   x0, [x29, #104]
    b     Lfd_ret
Lfd_drop_rm:
    mov   x0, x24
    bl    _t_decref
    b     Lfd_drop_dec_utf8_value
Lfd_drop_dec_utf8_value:
    mov   x0, x27
    bl    _t_decref
Lfd_drop_utf8_value:
    mov   x0, x22
    bl    _t_decref
    ldr   x0, [x29, #112]
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_added_done:
    mov   x0, x22
    bl    _t_decref
    b     Lfd_id2tok
Lfd_added_fallthrough:
    // non-special, non-literal added token: Python falls through to
    // id_to_token — drop the entry's content and continue
    mov   x0, x26
    bl    _t_decref
    b     Lfd_id2tok
Lfd_control_bad:
    adrp  x0, m_control@PAGE
    add   x0, x0, m_control@PAGEOFF
    bl    _raise_value
    mov   x0, x26
    bl    _t_decref
    b     Lfd_drop_tok0
    // ===== ordinary vocabulary path =====
Lfd_id2tok:
    // m = tok.id_to_token ; value = m(token)
    mov   x0, x21
    adrp  x1, a_id2tok@PAGE
    add   x1, x1, a_id2tok@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_tok
    mov   x24, x0
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfd_drop_idm
    mov   x25, x0
    mov   x0, x20
    bl    _incref
    mov   x0, x25
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    mov   x0, x24
    mov   x1, x25
    mov   x2, #0
    bl    _t_call
    mov   x26, x0                          // value
    mov   x0, x25
    bl    _t_decref
    mov   x0, x24
    bl    _t_decref
    cbz   x26, Lfd_drop_tok
    // isinstance(value, str)
    mov   x0, x26
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #168]                   // &PyUnicode_Type
    bl    _t_isinstance
    cmp   w0, #1
    b.ne  Lfd_vocab_bad
    // ---- map alphabet chars to bytes ----
    mov   x0, x26
    bl    _t_uni_length
    mov   x27, x0                          // n chars
    mov   x0, #0
    mov   x1, x27
    bl    _t_bytes_fsas                     // uninitialized, we own it
    cbz   x0, Lfd_drop_val2
    mov   x22, x0                          // raw
    add   x23, x22, #32                    // ob_sval
    str   xzr, [x29, #104]                 // i = 0
Lfd_map:
    ldr   x9, [x29, #104]
    cmp   x9, x27
    b.hs  Lfd_map_done
    mov   x0, x26
    mov   x1, x9
    bl    _t_readchar
    mov   x10, x0                          // ordinal
    mov   x0, x10
    bl    _ord_valid
    cbz   w0, Lfd_fallback
    adrp  x11, g_table@PAGE
    add   x11, x11, g_table@PAGEOFF
    ldrb  w12, [x11, x10]
    ldr   x9, [x29, #104]                   // reload i (calls clobbered it)
    strb  w12, [x23, x9]
    add   x9, x9, #1
    str   x9, [x29, #104]
    b     Lfd_map
Lfd_fallback:
    // unmapped char: whole piece falls back to UTF-8 bytes
    mov   x0, x22
    bl    _t_decref
    mov   x0, x26
    bl    _t_utf8str
    cbz   x0, Lfd_drop_val2
    mov   x22, x0
Lfd_map_done:
    mov   x0, x26
    bl    _t_decref                        // drop value
    mov   x0, x19
    adrp  x1, a_utf8@PAGE
    add   x1, x1, a_utf8@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd_drop_raw
    mov   x23, x0                          // utf8
    mov   x0, x23
    mov   x1, x22
    mov   x2, #0
    bl    _decode_call
    mov   x24, x0                          // result
    mov   x0, x22
    bl    _t_decref
    mov   x0, x23
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x0, x24
    b     Lfd_ret
Lfd_vocab_bad:
    adrp  x0, m_vocab@PAGE
    add   x0, x0, m_vocab@PAGEOFF
    bl    _raise_value
    mov   x0, x26
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_val2:
    mov   x0, x26
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_raw:
    mov   x0, x22
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_idm:
    mov   x0, x24
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_entry:
    mov   x0, x24
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_value:
    mov   x0, x26
    bl    _t_decref
    b     Lfd_drop_tok0
Lfd_drop_added_err:
    mov   x0, x22
    bl    _t_decref
Lfd_drop_tok:
    mov   x0, x21
    bl    _t_decref
Lfd_drop_tok0:
    mov   x0, #0
Lfd_ret0:
Lfd_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_finish(x0 = self) -> str
//   return self.utf8.decode(b'', final=True)
// ============================================================
_dd_finish:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    // x19 self   x20 utf8
    mov   x19, x0
    mov   x0, x19
    adrp  x1, a_utf8@PAGE
    add   x1, x1, a_utf8@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfin_ret0
    mov   x20, x0
    mov   x0, x20
    mov   x1, #0
    mov   x2, #1
    bl    _decode_call
    str   x0, [x29, #32]                   // park result
    mov   x0, x20
    bl    _t_decref
    ldr   x0, [x29, #32]
    b     Lfin_ret
Lfin_ret0:
    mov   x0, #0
Lfin_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret
