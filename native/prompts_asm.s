// damendaemon/prompts.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 3/370. 100% of validate_lengths,
// decoder_prefix, validate_raw, encode, decode_output.
//
// ABI table (offsets in bytes), installed via dd_init:
//     0 PyObject_GetAttrString    8 PyObject_Call
//    16 PyObject_IsTrue         24 PyLong_AsLongLong
//    32 PyDict_New              40 PyDict_SetItemString
//    48 PyObject_GetItem        56 PyErr_Clear
//    64 PyErr_SetString         72 PyExc_ValueError
//    80 None                    88 Py_True
//    96 &PyUnicode_Type        104 &PyLong_Type
//   112 &PyList_Type           120 Py_DecRef
//   128 PyTuple_New            136 PyTuple_SetItem
//   144 PyUnicode_AsUTF8String  152 PyList_GetSlice
//   160 PyUnicode_CompareWithASCIIString
//   168 key 'decoder_start_token_id'  176 PyList_New  184 PyList_SetItem
//
// isinstance(x, str/list) maps to exact ob_type comparisons; the
// original uses type(x) is int for token IDs (exact) — preserved.
// int tokens exceeding i64 range are accepted (documented delta).

.globl _dd_init
.globl _dd_validate_lengths
.globl _dd_decoder_prefix
.globl _dd_validate_raw
.globl _dd_encode
.globl _dd_decode_output

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_encdec:   .asciz "is_encoder_decoder"
a_context:  .asciz "context"
a_model:    .asciz "model"
a_config:   .asciz "config"
a_task:     .asciz "task"
a_rawtok:   .asciz "raw_tokens"
a_chattok:  .asciz "chat_tokens"
a_tokeniz:  .asciz "tokenizer"
a_decgen:   .asciz "decode_generated"
a_decode:   .asciz "decode"
k_enable:   .asciz "enable_thinking"
k_effort:   .asciz "reasoning_effort"
k_tools:    .asciz "tools"
k_tooljson: .asciz "tool_json"
k_skip:     .asciz "skip_special_tokens"
s_gen:      .asciz "generation"
m_enc:      .asciz "encoder prompt or decoder output exceeds configured context"
m_sum:      .asciz "prompt plus max_tokens exceeds configured context"
m_utf8:     .asciz "prompt must be valid UTF-8"
m_len:      .asciz "raw prompt exceeds configured context"
m_raw:      .asciz "prompt must be text or a nonempty list of token IDs"
m_task:     .asciz "this model does not support generation"
m_opt:      .asciz "raw prompts cannot include chat messages or template options"

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
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_aslong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_dict_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_dict_setstr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_obj_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_err_clear:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #120]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #128]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #136]
    br    x9
_t_utf8str:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #144]
    br    x9
_t_listslice:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #152]
    br    x9
_t_cmp_ascii:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #160]
    br    x9
_t_list_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #176]
    br    x9
_t_list_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #184]
    br    x9

// _raise_value(x0 = message C-string) -> ValueError set; returns
_raise_value:
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    mov   x1, x0                          // message
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #72]                   // PyExc_ValueError
    ldr   x9, [x9, #64]                   // PyErr_SetString
    blr   x9
    ldp   x29, x30, [sp], #16
    ret

// ============================================================
// PyObject* _dd_validate_lengths(x0 = executor, x1 = prompt_tokens,
//                                 x2 = max_tokens) -> None or NULL
// ============================================================
_dd_validate_lengths:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    str   x23, [sp, #48]
    str   x0, [x29, #56]                   // executor (parked; calls clobber x0)
    // x19 pt   x20 mt   x21 temp obj   x22 ctx   x23 enc flag
    mov   x0, x1
    bl    _t_aslong
    mov   x19, x0                          // pt (raw i64)
    mov   x0, x2
    bl    _t_aslong
    mov   x20, x0                          // mt (raw i64)
    mov   x23, #0                          // enc = 0
    // enc = truthy(getattr(executor, 'is_encoder_decoder', False))
    ldr   x0, [x29, #56]
    adrp  x1, a_encdec@PAGE
    add   x1, x1, a_encdec@PAGEOFF
    bl    _t_getattr_s
    cbnz  x0, 1f
    bl    _t_err_clear                     // missing -> default False
    b     2f
1:
    mov   x21, x0
    mov   x0, x21
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lvl_ret0                          // error in truthiness
    cset  x23, ne                           // enc = (w0 != 0)
    mov   x0, x21
    bl    _t_decref
2:
    // ctx = executor.context
    ldr   x0, [x29, #56]
    adrp  x1, a_context@PAGE
    add   x1, x1, a_context@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lvl_ret0                      // AttributeError propagates
    mov   x21, x0
    mov   x0, x21
    bl    _t_aslong
    mov   x22, x0                          // ctx (raw i64)
    mov   x0, x21
    bl    _t_decref
    cbz   x23, Lvl_sum
    cmp   x19, x22
    b.hi  Lvl_raise_enc
    cmp   x20, x22
    b.hi  Lvl_raise_enc
    b     Lvl_ok
Lvl_sum:
    add   x9, x19, x20
    cmp   x9, x22
    b.ls  Lvl_ok
Lvl_raise_enc:
    adrp  x0, m_enc@PAGE
    add   x0, x0, m_enc@PAGEOFF
    cmp   x23, #0
    b.ne  3f
    adrp  x0, m_sum@PAGE
    add   x0, x0, m_sum@PAGEOFF
3:
    bl    _raise_value
    b     Lvl_ret0
Lvl_ok:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #80]                    // None (immortal)
    b     Lvl_ret
Lvl_ret0:
    mov   x0, #0
Lvl_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldr   x23, [sp, #48]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_decoder_prefix(x0 = executor, x1 = prompt)
//   enc-decoder: [model.config['decoder_start_token_id']]
//   otherwise:   prompt (same object, INCREF'd)
// ============================================================
_dd_decoder_prefix:
    stp   x29, x30, [sp, #-80]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    // x19 enc flag   x20 model   x21 config   x22 start
    // x23 executor (kept)   x24 prompt (kept)   [x29,#64] result list
    mov   x24, x1
    mov   x23, x0
    mov   x19, #0
    adrp  x1, a_encdec@PAGE
    add   x1, x1, a_encdec@PAGEOFF
    bl    _t_getattr_s
    cbnz  x0, 1f
    bl    _t_err_clear
    b     2f
1:
    mov   x9, x0
    mov   x0, x9
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Ldp_ret0
    cset  x19, ne
    mov   x0, x9
    bl    _t_decref
2:
    cbz   x19, Ldp_plain
    // model = executor.model
    mov   x0, x23
    adrp  x1, a_model@PAGE
    add   x1, x1, a_model@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldp_ret0
    mov   x20, x0
    // config = model.config
    mov   x0, x20
    adrp  x1, a_config@PAGE
    add   x1, x1, a_config@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldp_drop_model
    mov   x21, x0
    // start = config['decoder_start_token_id']
    mov   x0, x21
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #168]
    bl    _t_obj_getitem
    cbz   x0, Ldp_drop_conf
    mov   x22, x0
    // result = [start]
    mov   x0, #1
    bl    _t_list_new
    cbz   x0, Ldp_drop_start
    str   x0, [x29, #64]                  // park the list across the call
    ldr   x0, [x29, #64]
    mov   x1, #0                          // pos
    mov   x2, x22                         // start
    bl    _t_list_setitem                  // steals start on success
    cbnz  w0, Ldp_drop_start
    mov   x0, x21
    bl    _t_decref                        // drop config
    mov   x0, x20
    bl    _t_decref                        // drop model
    ldr   x0, [x29, #64]                   // return the list
    b     Ldp_ret
Ldp_plain:
    mov   x0, x24                          // prompt (same object)
    ldr   x9, [x0]                         // INCREF (mortal)
    add   x9, x9, #1
    str   x9, [x0]
    b     Ldp_ret
Ldp_drop_start:
    mov   x0, x22
    bl    _t_decref
Ldp_drop_conf:
    mov   x0, x21
    bl    _t_decref
Ldp_drop_model:
    mov   x0, x20
    bl    _t_decref
Ldp_ret0:
    mov   x0, #0
Ldp_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #80
    ret

// ============================================================
// PyObject* _dd_validate_raw(x0 = prompt, x1 = max_length|None)
//   exact str: UTF-8 encodable -> same str (INCREF'd)
//   exact list of exact ints >= 0 -> fresh copy
//   otherwise ValueError
// ============================================================
_dd_validate_raw:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 prompt   x20 max_length   x21 n   [x29,#48] i
    mov   x19, x0
    mov   x20, x1
    ldr   x9, [x19, #8]                   // ob_type
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x11, [x10, #96]                 // &PyUnicode_Type
    cmp   x9, x11
    b.eq  Lvr_str
    ldr   x11, [x10, #112]                // &PyList_Type
    cmp   x9, x11
    b.ne  Lvr_raise_raw
    // -------- list path --------
    ldr   x21, [x19, #16]                 // n = len(prompt)
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x11, [x9, #80]                  // None
    cmp   x20, x11
    b.eq  Lvr_nonempty
    mov   x0, x20
    bl    _t_aslong
    cmp   x21, x0
    b.hi  Lvr_raise_len
Lvr_nonempty:
    cbz   x21, Lvr_raise_raw              // empty -> invalid
    str   xzr, [x29, #48]                 // i = 0
Lvr_loop:
    ldr   x12, [x29, #48]
    cmp   x12, x21
    b.hs  Lvr_copy
    ldr   x13, [x19, #24]                 // ob_item
    ldr   x14, [x13, x12, lsl #3]         // token
    ldr   x15, [x14, #8]                  // type(token)
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x16, [x9, #104]                 // &PyLong_Type
    cmp   x15, x16
    b.ne  Lvr_raise_raw
    mov   x0, x14
    bl    _t_aslong
    cmp   x0, #0
    b.lt  Lvr_raise_raw
    ldr   x12, [x29, #48]
    add   x12, x12, #1
    str   x12, [x29, #48]
    b     Lvr_loop
Lvr_copy:
    mov   x0, x19
    mov   x1, #0
    mov   x2, x21
    bl    _t_listslice                     // prompt[0:n] fresh copy
    b     Lvr_ret
Lvr_str:
    mov   x0, x19
    bl    _t_utf8str                       // raises UnicodeEncodeError
    cbz   x0, Lvr_utf8_bad
    bl    _t_decref                        // drop validation bytes
    mov   x0, x19
    ldr   x9, [x0]                        // INCREF prompt (mortal)
    add   x9, x9, #1
    str   x9, [x0]
    b     Lvr_ret
Lvr_utf8_bad:
    bl    _t_err_clear
    adrp  x0, m_utf8@PAGE
    add   x0, x0, m_utf8@PAGEOFF
    bl    _raise_value
    b     Lvr_ret0
Lvr_raise_len:
    adrp  x0, m_len@PAGE
    add   x0, x0, m_len@PAGEOFF
    bl    _raise_value
    b     Lvr_ret0
Lvr_raise_raw:
    adrp  x0, m_raw@PAGE
    add   x0, x0, m_raw@PAGEOFF
    bl    _raise_value
    b     Lvr_ret0
Lvr_ret0:
    mov   x0, #0
Lvr_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_encode(x0=model, x1=messages, x2=raw_prompt,
//   x3=enable_thinking, x4=tools, x5=tool_json, x6=reasoning_effort)
//   task != 'generation' -> ValueError
//   raw_prompt path -> model.raw_tokens(raw_prompt)
//   chat path -> model.chat_tokens(messages, **options)
// ============================================================
_dd_encode:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    // [x29,#96] = result parking slot (survives epilogue decrefs)
    // x19 model  x20 messages  x21 raw  x22 thinking  x23 tools
    // x24 tool_json  x25 effort  x26/x27 temps
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, x5
    mov   x25, x6
    // ---- task check ----
    adrp  x1, a_task@PAGE
    add   x1, x1, a_task@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, 1f
    mov   x26, x0
    mov   x0, x26
    adrp  x1, s_gen@PAGE
    add   x1, x1, s_gen@PAGEOFF
    bl    _t_cmp_ascii
    mov   x27, x0                          // save cmp result
    mov   x0, x26
    bl    _t_decref
    cbnz  x27, Lpe_raise_task             // != 'generation'
    b     2f
1:
    bl    _t_err_clear                     // default "generation"
2:
    // ---- raw branch ----
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x11, [x9, #80]                   // None
    cmp   x21, x11
    b.eq  Lpe_chat
    cmp   x20, x11
    b.ne  Lpe_raise_opt
    cmp   x22, x11
    b.ne  Lpe_raise_opt
    cmp   x23, x11
    b.ne  Lpe_raise_opt
    cmp   x25, x11
    b.ne  Lpe_raise_opt
    mov   x0, x24
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lpe_ret0                         // truthiness error
    cbnz  w0, Lpe_raise_opt
    // result = model.raw_tokens(raw_prompt)
    adrp  x1, a_rawtok@PAGE
    add   x1, x1, a_rawtok@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lpe_ret0
    mov   x26, x0
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lpe_drop_rt
    mov   x27, x0
    // raw_prompt is a borrowed ref; INCREF before the tuple steals it
    ldr   x9, [x21]
    add   x9, x9, #1
    str   x9, [x21]
    mov   x0, x27
    mov   x1, #0
    mov   x2, x21                          // raw
    bl    _t_tuple_setitem
    mov   x0, x26
    mov   x1, x27
    mov   x2, #0
    bl    _t_call
    str   x0, [x29, #96]                   // park result
    mov   x0, x27
    bl    _t_decref                        // drop args tuple
    mov   x0, x26
    bl    _t_decref                        // drop raw_tokens func
    ldr   x0, [x29, #96]
    b     Lpe_ret
Lpe_drop_rt:
    mov   x0, x26
    bl    _t_decref
    b     Lpe_ret0
Lpe_raise_opt:
    adrp  x0, m_opt@PAGE
    add   x0, x0, m_opt@PAGEOFF
    bl    _raise_value
    b     Lpe_ret0
Lpe_raise_task:
    adrp  x0, m_task@PAGE
    add   x0, x0, m_task@PAGEOFF
    bl    _raise_value
    b     Lpe_ret0
    // ---- chat branch ----
Lpe_chat:
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Lpe_ret0
    mov   x26, x0                          // x26 = options
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x28, [x9, #80]                   // x28 = None (callee-saved:
                                           // survives the setstr calls)
    cmp   x22, x28
    b.eq  3f
    mov   x0, x26
    adrp  x1, k_enable@PAGE
    add   x1, x1, k_enable@PAGEOFF
    mov   x2, x22
    bl    _t_dict_setstr
3:
    cmp   x25, x28
    b.eq  4f
    mov   x0, x26
    adrp  x1, k_effort@PAGE
    add   x1, x1, k_effort@PAGEOFF
    mov   x2, x25
    bl    _t_dict_setstr
4:
    cmp   x23, x28
    b.eq  5f
    mov   x0, x26
    adrp  x1, k_tools@PAGE
    add   x1, x1, k_tools@PAGEOFF
    mov   x2, x23
    bl    _t_dict_setstr
5:
    mov   x0, x24
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lpe_drop_opts
    b.eq  6f
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #88]                    // Py_True
    mov   x0, x26
    adrp  x1, k_tooljson@PAGE
    add   x1, x1, k_tooljson@PAGEOFF
    bl    _t_dict_setstr
6:
    // chat = model.chat_tokens
    adrp  x1, a_chattok@PAGE
    add   x1, x1, a_chattok@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lpe_drop_opts
    mov   x27, x0                          // x27 = chat
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lpe_drop_chat
    mov   x28, x0                          // x28 = args
    // messages is a borrowed ref; INCREF before the tuple steals it
    ldr   x9, [x20]
    add   x9, x9, #1
    str   x9, [x20]
    mov   x0, x28
    mov   x1, #0
    mov   x2, x20                          // messages
    bl    _t_tuple_setitem
    // result = chat(messages, **options)
    mov   x0, x27
    mov   x1, x28
    mov   x2, x26
    bl    _t_call
    str   x0, [x29, #96]                   // park result
    mov   x0, x28
    bl    _t_decref                        // drop args tuple
    mov   x0, x26
    bl    _t_decref                        // drop options
    mov   x0, x27
    bl    _t_decref                        // drop chat func
    ldr   x0, [x29, #96]
    b     Lpe_ret
Lpe_drop_chat:
    mov   x0, x27
    bl    _t_decref
Lpe_drop_opts:
    mov   x0, x26
    bl    _t_decref
    b     Lpe_ret0
Lpe_ret0:
    mov   x0, #0
Lpe_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #112
    ret

// ============================================================
// PyObject* _dd_decode_output(x0 = model, x1 = tokens)
//   decode = tokenizer.decode_generated | tokenizer.decode
//   return decode(tokens, skip_special_tokens=True)
// ============================================================
_dd_decode_output:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    str   x1, [x29, #48]                   // tokens (parked)
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 tokenizer   x20 decode func   x21 kwargs   x22 args
    // [x29,#56] result parking
    adrp  x1, a_tokeniz@PAGE
    add   x1, x1, a_tokeniz@PAGEOFF
    bl    _t_getattr_s                     // x0 = model
    cbz   x0, Ldo_ret0
    mov   x19, x0
    // decode = tokenizer.decode_generated
    mov   x0, x19
    adrp  x1, a_decgen@PAGE
    add   x1, x1, a_decgen@PAGEOFF
    bl    _t_getattr_s
    cbnz  x0, 1f
    bl    _t_err_clear                     // fall back to tokenizer.decode
    mov   x0, x19
    adrp  x1, a_decode@PAGE
    add   x1, x1, a_decode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldo_drop_tok
1:
    mov   x20, x0
    // kwargs = {'skip_special_tokens': True}
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Ldo_drop_dec
    mov   x21, x0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #88]                    // Py_True
    mov   x0, x21
    adrp  x1, k_skip@PAGE
    add   x1, x1, k_skip@PAGEOFF
    bl    _t_dict_setstr
    // args = (tokens,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ldo_drop_kwargs
    mov   x22, x0
    // tokens is a borrowed ref; INCREF before the tuple steals it
    ldr   x10, [x29, #48]                  // x10 = the tokens object
    ldr   x9, [x10]                        // its refcount
    add   x9, x9, #1
    str   x9, [x10]
    mov   x0, x22
    mov   x1, #0
    ldr   x2, [x29, #48]                   // tokens
    bl    _t_tuple_setitem
    // result = decode(tokens, skip_special_tokens=True)
    mov   x0, x20
    mov   x1, x22
    mov   x2, x21
    bl    _t_call
    str   x0, [x29, #56]
    mov   x0, x22
    bl    _t_decref                        // drop args
    mov   x0, x21
    bl    _t_decref                        // drop kwargs
    mov   x0, x20
    bl    _t_decref                        // drop decode func
    mov   x0, x19
    bl    _t_decref                        // drop tokenizer
    ldr   x0, [x29, #56]
    b     Ldo_ret
Ldo_drop_kwargs:
    mov   x0, x21
    bl    _t_decref
Ldo_drop_dec:
    mov   x0, x20
    bl    _t_decref
Ldo_drop_tok:
    mov   x0, x19
    bl    _t_decref
Ldo_ret0:
    mov   x0, #0
Ldo_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret
