// damendaemon/harmony.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 8/370. 100% of the module: the TextStream
// constructor (marker-table validation, all 17 state attributes, tool
// name set), _decode, _begin_body (header role/channel/format
// validation, recipients, function handoff names), _commit_body, the
// full feed state machine (start/message/end/return/eot/call markers,
// header/body token accumulation, header bound, control-token
// rejection, completed/emitted bookkeeping through stopping.text_delta,
// closed latch, 3-tuple return), and output_stream (model gate,
// unsupported-feature rejection, TextStream construction).
//
// ABI table (byte offsets), installed via dd_init:
//    0 GetAttrString    8 SetAttrString   16 PyObject_Call
//   24 PyTuple_New     32 PyTuple_SetItem 40 Py_DecRef
//   48 PyDict_New      56 PyDict_SetItem  64 PyDict_GetItemWithError
//   72 PyDict_Contains 88 PyObject_IsTrue 96 PyErr_Clear
//  104 PyErr_SetString 112 PyExc_ValueError 120 None
//  128 PyLong_AsLongLong 136 PyLong_FromLongLong
// 144 PyUnicode_FromString 152 PyUnicode_Concat
// 160 PyObject_GetIter 168 PyObject_Size  176 PyIter_Next
//  216 PySequence_GetSlice 224 PySequence_Concat
//  232 PyList_New      240 PyList_Append
//  248 PyUnicode_CompareWithASCIIString  256 PyDict_Copy
//  264 PySet_New       272 PySet_Add     280 True    288 False
//  296 text_delta (stopping bridge)  304 _json_object
//  312 validate_tools  320 uuid.uuid4  328 json.dumps
//  336 TextStream class  344 PyUnicode_AsUTF8AndSize
// Python singletons held by the bridge (borrowed): 352+ name slots.
// g_name_py[] — PyUnicode objects for the 12 marker names (built at
// dd_init, owned by this lib). g_kind_py[] — 8 kind strings.
// g_markers — {id: kind} dict. g_kw_ss — decode kwargs. g_kw_dumps —
// json.dumps kwargs. g_empty_list — shared empty list (template only).

.globl _dd_init
.globl _dd_text_init
.globl _dd_feed
.globl _dd_output_stream
.globl _dd_markers
.globl _dd_decode

.section __DATA,__data
.p2align 3
g_iface:      .quad 0
g_markers:    .quad 0
g_kw_ss:      .quad 0
g_kw_dumps:   .quad 0
g_empty_list: .quad 0
g_name_py:    .space 96          // 12 name objects
g_kind_py:    .space 64          // 8 kind objects
// marker table: kind C-string, expected id, name0 slot idx, name1 slot
// idx (name1 = 0xFF sentinel when single-name). Order MUST match
// the Python dict iteration below; semantic only, tests use the dict.
mk_table:
    .quad a_start,   200006, 0,  0xFF
    .quad a_message, 200008, 1,  0xFF
    .quad a_end,     200007, 2,  0xFF
    .quad a_return,  200002, 3,  4
    .quad a_call,    200012, 5,  6
    .quad a_channel, 200005, 7,  8
    .quad a_format,  200003, 9,  10
    .quad a_eot,     199999, 11, 0xFF

.section __TEXT,__cstring
a_start:   .asciz "start"
a_message: .asciz "message"
a_end:     .asciz "end"
a_return:  .asciz "return"
a_call:    .asciz "call"
a_channel: .asciz "channel"
a_format:  .asciz "format"
a_eot:     .asciz "eot"
n_start:   .asciz "<|start|>"
n_message: .asciz "<|message|>"
n_end:     .asciz "<|end|>"
n_return:  .asciz "<|return|>"
n_fimsuf:  .asciz "<|fim_suffix|>"
n_call:    .asciz "<|call|>"
n_ghissue: .asciz "<|ghissue|>"
n_channel: .asciz "<|channel|>"
n_metasep: .asciz "<|meta_sep|>"
n_constr:  .asciz "<|constrain|>"
n_metastart: .asciz "<|meta_start|>"
n_eot:     .asciz "<|endoftext|>"
// attribute names
at_tk:     .asciz "tokenizer"
at_stops:  .asciz "stops"
at_markers: .asciz "markers"
at_phase:  .asciz "phase"
at_prefix: .asciz "prefix"
at_header: .asciz "header"
at_field:  .asciz "field"
at_hcount: .asciz "header_count"
at_channel: .asciz "channel"
at_format: .asciz "format"
at_body:   .asciz "body"
at_completed: .asciz "completed"
at_emitted: .asciz "emitted"
at_closed: .asciz "closed"
at_names:  .asciz "names"
at_recipient: .asciz "recipient"
at_calls:  .asciz "calls"
// method names
m_token_to_id: .asciz "token_to_id"
m_decode: .asciz "decode"
m_split:  .asciz "split"
m_strip:  .asciz "strip"
m_startswith: .asciz "startswith"
m_append: .asciz "append"
m_get:    .asciz "get"
m_hex:    .asciz "hex"
// dict keys
k_role:   .asciz "role"
k_content: .asciz "content"
k_analysis: .asciz "analysis"
k_function: .asciz "function"
k_name:   .asciz "name"
k_id:     .asciz "id"
k_type:   .asciz "type"
k_arguments: .asciz "arguments"
k_ss:     .asciz "skip_special_tokens"
k_ensure_ascii: .asciz "ensure_ascii"
k_allow_nan: .asciz "allow_nan"
k_model_type: .asciz "model_type"
// values
v_header: .asciz "header"
v_after:  .asciz "after"
v_body:   .asciz "body"
v_assistant: .asciz "assistant"
v_final:  .asciz "final"
v_commentary: .asciz "commentary"
v_all:    .asciz "all"
v_json:   .asciz "json"
v_to:     .asciz "to="
v_funcs:  .asciz "functions."
v_call_p: .asciz "call_"
v_gpt_oss: .asciz "gpt_oss"
at_config: .asciz "config"
// messages
m_bad_marker: .asciz "unsupported Harmony tokenizer marker: "
m_bad_header: .asciz "invalid Harmony assistant header"
m_not_declared: .asciz "Harmony function recipient is not declared"
m_bad_chan: .asciz "unsupported Harmony output channel or format"
m_chan_comment: .asciz "Harmony function calls require the commentary channel"
m_need_handoff: .asciz "Harmony function call requires a handoff marker"
m_bad_start: .asciz "unexpected Harmony message start"
m_bad_meta: .asciz "unexpected Harmony header metadata"
m_bad_sep: .asciz "unexpected Harmony body separator"
m_bad_end: .asciz "unexpected Harmony message end"
m_ret_nobody: .asciz "Harmony return without a message body"
m_bad_handoff: .asciz "Harmony handoff requires a declared function body"
m_bad_control: .asciz "unsupported non-text Harmony control token"
m_h_limit: .asciz "Harmony header exceeds token limit"
m_outside: .asciz "Harmony content outside a message"
m_continued: .asciz "Harmony output continued after its terminal marker"
m_unsupported: .asciz "Harmony text chat does not yet support structured output, logprobs or ignoring EOS"
v_analysis_cs: .asciz "analysis"
c_empty:  .asciz ""

.text

// ---- tail-call trampolines ----
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
_t_aslong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #128]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #136]
    br    x9
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #144]
    br    x9
_t_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #152]
    br    x9
_t_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #168]
    br    x9
_t_seq_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #208]
    br    x9
_t_getslice:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #216]
    br    x9
_t_seq_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #224]
    br    x9
_t_list_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #232]
    br    x9
_t_list_append:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #240]
    br    x9
_t_strcmp:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #248]
    br    x9
_t_dict_copy:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #256]
    br    x9
_t_set_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #264]
    br    x9
_t_set_add:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #272]
    br    x9
_t_contains:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_getiter:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #160]
    br    x9
_t_iter_next:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #176]
    br    x9

// _raise_value(x0 = message C-string)
_raise_value:
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    mov   x1, x0
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #112]
    ldr   x9, [x9, #104]
    blr   x9
    ldp   x29, x30, [sp], #16
    ret

// _raise_value_cat(x0 = prefix C-string, x1 = suffix str obj BORROWED)
_raise_value_cat:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    mov   x19, x1
    bl    _t_fromstring                 // prefix str NEW
    mov   x20, x0
    cbz   x20, Lrvc_out
    mov   x0, x20
    mov   x1, x19
    bl    _t_concat                     // msg NEW
    mov   x20, x0
    cbz   x20, Lrvc_out
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #344]
    mov   x0, x20
    mov   x1, #0
    blr   x9                            // cstr
    mov   x1, x0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]
    ldr   x9, [x10, #104]
    blr   x9                            // PyErr_SetString
    mov   x0, x20
    bl    _t_decref
Lrvc_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// _incref(x0) — immortal-aware
_incref:
    ldr   x9, [x0]
    mov   w10, #0xffff
    movk  w10, #0xffff, lsl #16
    cmp   x9, x10
    b.hs  9f
    add   x9, x9, #1
    str   x9, [x0]
9:  ret

// ============================================================
// void _dd_init(x0 = iface) — install table; build the 12 marker
// name strings, 8 kind strings, g_markers {id:kind}, g_kw_ss
// {'skip_special_tokens': False}, g_kw_dumps {'ensure_ascii': False,
// 'allow_nan': False}, g_empty_list.
// ============================================================
_dd_init:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    // ---- build the 12 name strings ----
    // name C-string table (indexes match mk_table slot idx)
    adrp  x21, nm_table@PAGE
    add   x21, x21, nm_table@PAGEOFF
    mov   x20, #0
Lni1:
    cmp   x20, #12
    b.hs  Lni2
    ldr   x0, [x21, x20, lsl #3]
    bl    _t_fromstring
    cbz   x0, Lni_out
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    str   x0, [x9, x20, lsl #3]
    add   x20, x20, #1
    b     Lni1
Lni2:
    // ---- build the 8 kind strings + g_markers ----
    bl    _t_dict_new
    cbz   x0, Lni_out
    str   x0, [sp, #72]                    // markers dict
    adrp  x19, mk_table@PAGE
    add   x19, x19, mk_table@PAGEOFF
    mov   x20, #0
Lni3:
    cmp   x20, #8
    b.hs  Lni4
    ldr   x0, [x19]                        // kind C-string
    bl    _t_fromstring
    cbz   x0, Lni_out
    adrp  x9, g_kind_py@PAGE
    add   x9, x9, g_kind_py@PAGEOFF
    str   x0, [x9, x20, lsl #3]            // kind str (owned)
    ldr   x0, [x19, #8]                    // expected id
    bl    _t_fromlong
    cbz   x0, Lni_out
    mov   x22, x0                          // id key
    adrp  x9, g_kind_py@PAGE
    add   x9, x9, g_kind_py@PAGEOFF
    ldr   x2, [x9, x20, lsl #3]
    mov   x1, x22                           // key = id
    ldr   x0, [sp, #72]
    bl    _t_dict_setitem                   // markers[id] = kind
    add   x19, x19, #32
    add   x20, x20, #1
    b     Lni3
Lni4:
    adrp  x9, g_markers@PAGE
    ldr   x10, [sp, #72]
    str   x10, [x9, g_markers@PAGEOFF]
    // ---- g_kw_ss = {'skip_special_tokens': False} ----
    bl    _t_dict_new
    cbz   x0, Lni_out
    adrp  x9, g_kw_ss@PAGE
    str   x0, [x9, g_kw_ss@PAGEOFF]
    adrp  x0, k_ss@PAGE
    add   x0, x0, k_ss@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lni_out
    str   x0, [sp, #80]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #288]                  // False (borrowed)
    adrp  x10, g_kw_ss@PAGE
    ldr   x0, [x10, g_kw_ss@PAGEOFF]
    ldr   x1, [sp, #80]
    bl    _t_dict_setitem
    ldr   x0, [sp, #80]
    bl    _t_decref
    // ---- g_kw_dumps = {'ensure_ascii': False, 'allow_nan': False} ----
    bl    _t_dict_new
    cbz   x0, Lni_out
    adrp  x9, g_kw_dumps@PAGE
    str   x0, [x9, g_kw_dumps@PAGEOFF]
    adrp  x0, k_ensure_ascii@PAGE
    add   x0, x0, k_ensure_ascii@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lni_out
    str   x0, [sp, #80]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #288]
    adrp  x10, g_kw_dumps@PAGE
    ldr   x0, [x10, g_kw_dumps@PAGEOFF]
    ldr   x1, [sp, #80]
    bl    _t_dict_setitem
    ldr   x0, [sp, #80]
    bl    _t_decref
    adrp  x0, k_allow_nan@PAGE
    add   x0, x0, k_allow_nan@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lni_out
    str   x0, [sp, #80]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #288]
    adrp  x10, g_kw_dumps@PAGE
    ldr   x0, [x10, g_kw_dumps@PAGEOFF]
    ldr   x1, [sp, #80]
    bl    _t_dict_setitem
    ldr   x0, [sp, #80]
    bl    _t_decref
    // ---- g_empty_list ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lni_out
    adrp  x9, g_empty_list@PAGE
    str   x0, [x9, g_empty_list@PAGEOFF]
    // ---- build the 16 value strings (g_val_py) ----
    adrp  x21, vt_table@PAGE
    add   x21, x21, vt_table@PAGEOFF
    mov   x20, #0
Lni5:
    cmp   x20, #21
    b.hs  Lni_out
    ldr   x0, [x21, x20, lsl #3]
    bl    _t_fromstring
    cbz   x0, Lni_out
    adrp  x9, g_val_py@PAGE
    add   x9, x9, g_val_py@PAGEOFF
    str   x0, [x9, x20, lsl #3]
    add   x20, x20, #1
    b     Lni5
Lni_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #96
    ret

.section __DATA,__data
.p2align 3
nm_table:
    .quad n_start, n_message, n_end, n_return, n_fimsuf
    .quad n_call, n_ghissue, n_channel, n_metasep, n_constr
    .quad n_metastart, n_eot
vt_table:
    .quad v_header, v_after, v_body, v_assistant, v_final, v_all
    .quad k_content, k_analysis, k_role, v_call_p, k_id, k_type
    .quad k_function, k_name, k_arguments, v_to
    .quad v_funcs, v_json, v_commentary, at_channel, at_format
g_val_py:  .space 168
.text

// ============================================================
// helper _meth_call(x0 = obj BORROWED, x1 = method C-string,
//                   x2 = args tuple BORROWED or NULL,
//                   x3 = kwargs dict BORROWED or NULL) -> NEW
// ============================================================
_meth_call:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 obj  x20 args  x21 kwargs  x22 method
    mov   x19, x0
    mov   x20, x2
    mov   x21, x3
    bl    _t_getattr_s
    cbz   x0, Lmc_out
    mov   x22, x0                          // method NEW
    mov   x0, x22
    mov   x1, x20
    mov   x2, x21
    bl    _t_call
    str   x0, [sp, #48]                     // park result
    mov   x0, x22
    bl    _t_decref                         // drop method
    ldr   x0, [sp, #48]
Lmc_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// helper _decode(x0 = self, x1 = tokens BORROWED) -> str NEW
//   self.tokenizer.decode(tokens, skip_special_tokens=False)
// ============================================================
_decode:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    // x19 self  x20 tokens
    mov   x19, x0
    mov   x20, x1
    // args = (tokens,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ldc_out
    str   x0, [sp, #56]                     // args tuple (park; x21 is caller's)
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #56]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    // tokenizer = self.tokenizer
    mov   x0, x19
    adrp  x1, at_tk@PAGE
    add   x1, x1, at_tk@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldc_drop_args
    str   x0, [sp, #32]                     // tokenizer
    // decm = tokenizer.decode
    ldr   x0, [sp, #32]
    adrp  x1, m_decode@PAGE
    add   x1, x1, m_decode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ldc_drop_tk
    str   x0, [sp, #40]                     // park decm
    ldr   x0, [sp, #40]
    ldr   x1, [sp, #56]
    adrp  x9, g_kw_ss@PAGE
    ldr   x2, [x9, g_kw_ss@PAGEOFF]
    bl    _t_call
    str   x0, [sp, #48]                     // park result
    ldr   x0, [sp, #40]
    bl    _t_decref                         // drop decm
    ldr   x0, [sp, #32]
    bl    _t_decref                         // drop tokenizer
    ldr   x0, [sp, #56]
    bl    _t_decref                         // drop args
    str   xzr, [sp, #56]
    ldr   x0, [sp, #48]
    b     Ldc_out
Ldc_drop_tk:
    ldr   x0, [sp, #32]
    bl    _t_decref
Ldc_drop_args:
    ldr   x0, [sp, #56]
    bl    _t_decref
    str   xzr, [sp, #56]
    mov   x0, #0
Ldc_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// helper _val(x0 = index) -> borrowed value str from g_val_py
_val:
    adrp  x9, g_val_py@PAGE
    add   x9, x9, g_val_py@PAGEOFF
    ldr   x0, [x9, x0, lsl #3]
    ret

// helper _setattr_const(x0 = self, x1 = name C-string, x2 = value
//   BORROWED) -> w0 rc (0 ok / -1 error)
_setattr_const:
    mov   x2, x2
    b     _t_setattr_s

// ============================================================
// void _dd_text_init(x0 = self, x1 = tokenizer, x2 = stops,
//                    x3 = tools)  — returns None or NULL (error)
// x19 self  x20 tokenizer  x21 stops  x22 tools  x23 mk table
// x24 marker idx  x25 expected/found
// Parks: [88] header [96] completed [104] emitted [112] nset
//        [120] tools_list [128] iter [136] args [144] scratch
// ============================================================
_dd_text_init:
    stp   x29, x30, [sp, #-160]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #152]
    stp   xzr, xzr, [sp, #104]
    stp   xzr, xzr, [sp, #120]
    stp   xzr, xzr, [sp, #136]
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    // ================= marker validation =================
    adrp  x23, mk_table@PAGE
    add   x23, x23, mk_table@PAGEOFF
    mov   x24, #0
Lmv1:
    cmp   x24, #8
    b.hs  Lmv_done
    ldr   x25, [x23, #8]                    // expected
    // ---- name slot 0 ----
    ldr   x26, [x23, #16]                   // slot idx
    bl    _marker_check                     // x0 = tokenizer(keep x20), clobbers? uses x0-x3
    cbnz  w0, Lmv_next
    // ---- name slot 1 (if not 0xFF) ----
    ldr   x26, [x23, #24]
    cmp   x26, #0xFF
    b.eq  Lmv_fail
    bl    _marker_check
    cbnz  w0, Lmv_next
Lmv_fail:
    // raise 'unsupported Harmony tokenizer marker: ' + kind
    adrp  x0, m_bad_marker@PAGE
    add   x0, x0, m_bad_marker@PAGEOFF
    adrp  x9, g_kind_py@PAGE
    add   x9, x9, g_kind_py@PAGEOFF
    ldr   x1, [x9, x24, lsl #3]
    bl    _raise_value_cat
    b     Lti_err
Lmv_next:
    add   x23, x23, #32
    add   x24, x24, #1
    b     Lmv1
Lmv_done:
    // ================= attribute setup =================
    // self.tokenizer = tokenizer; self.stops = stops
    mov   x0, x19
    adrp  x1, at_tk@PAGE
    add   x1, x1, at_tk@PAGEOFF
    mov   x2, x20
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    mov   x0, x19
    adrp  x1, at_stops@PAGE
    add   x1, x1, at_stops@PAGEOFF
    mov   x2, x21
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    // self.markers = g_markers (global dict, borrowed — SetAttr increfs)
    adrp  x9, g_markers@PAGE
    ldr   x2, [x9, g_markers@PAGEOFF]
    mov   x0, x19
    adrp  x1, at_markers@PAGE
    add   x1, x1, at_markers@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    // self.phase = 'header'; self.prefix = 'assistant'
    mov   x0, #0
    bl    _val                              // 'header'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    mov   x0, #3
    bl    _val                              // 'assistant'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_prefix@PAGE
    add   x1, x1, at_prefix@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    // header = {'role': []}
    bl    _t_dict_new
    cbz   x0, Lti_err
    str   x0, [sp, #152]
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lti_drop_header
    str   x0, [sp, #144]
    mov   x0, #8
    bl    _val                              // 'role'
    mov   x1, x0
    ldr   x0, [sp, #152]
    ldr   x2, [sp, #144]
    bl    _t_dict_setitem
    ldr   x0, [sp, #144]
    bl    _t_decref
    mov   x0, x19
    adrp  x1, at_header@PAGE
    add   x1, x1, at_header@PAGEOFF
    ldr   x2, [sp, #152]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_header
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    // field = 'role'; header_count = 0
    mov   x0, #8
    bl    _val
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_field@PAGE
    add   x1, x1, at_field@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lti_err
    str   x0, [sp, #144]
    mov   x0, x19
    adrp  x1, at_hcount@PAGE
    add   x1, x1, at_hcount@PAGEOFF
    ldr   x2, [sp, #144]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_144
    ldr   x0, [sp, #144]
    bl    _t_decref
    // channel = None; body = []
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    mov   x0, x19
    adrp  x1, at_channel@PAGE
    add   x1, x1, at_channel@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lti_err
    str   x0, [sp, #144]
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    ldr   x2, [sp, #144]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_144
    ldr   x0, [sp, #144]
    bl    _t_decref
    // completed = emitted = {'content':'','analysis':''}
    bl    _t_dict_new
    cbz   x0, Lti_err
    str   x0, [sp, #96]
    bl    _t_dict_new
    cbz   x0, Lti_drop_completed
    str   x0, [sp, #104]
    // fill both dicts: keys 6 'content', 7 'analysis' -> ''
    adrp  x27, c_empty@PAGE
    add   x27, x27, c_empty@PAGEOFF         // "" C-string
    mov   x28, #6
Lti_dicts:
    mov   x0, x27
    bl    _t_fromstring                     // '' NEW
    cbz   x0, Lti_drop_dicts
    str   x0, [sp, #144]
    mov   x0, x28
    bl    _val                              // key str borrowed
    mov   x1, x0
    ldr   x0, [sp, #96]
    ldr   x2, [sp, #144]
    bl    _t_dict_setitem
    mov   x0, x28
    bl    _val
    mov   x1, x0
    ldr   x0, [sp, #104]
    ldr   x2, [sp, #144]
    bl    _t_dict_setitem
    ldr   x0, [sp, #144]
    bl    _t_decref
    add   x28, x28, #1
    cmp   x28, #8
    b.lo  Lti_dicts
    mov   x0, x19
    adrp  x1, at_completed@PAGE
    add   x1, x1, at_completed@PAGEOFF
    ldr   x2, [sp, #96]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_dicts
    mov   x0, x19
    adrp  x1, at_emitted@PAGE
    add   x1, x1, at_emitted@PAGEOFF
    ldr   x2, [sp, #104]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_dicts
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    // closed = False
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #288]
    mov   x0, x19
    adrp  x1, at_closed@PAGE
    add   x1, x1, at_closed@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    // ================= names set =================
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #264]                   // PySet_New
    mov   x0, #0
    blr   x9
    cbz   x0, Lti_err
    str   x0, [sp, #112]
    // tools_list = validate_tools(tools) or []
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lti_drop_nset
    str   x0, [sp, #136]
    mov   x0, x22
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #312]                   // validate_tools
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #120]
    cbz   x0, Lti_drop_nset
    bl    _t_istrue
    cmp   w0, #1
    b.eq  Lti_tools_ok
    // falsy: replace with the shared empty list
    ldr   x0, [sp, #120]
    bl    _t_decref
    adrp  x9, g_empty_list@PAGE
    ldr   x9, [x9, g_empty_list@PAGEOFF]
    str   x9, [sp, #120]
    // borrowed now — no decref at cleanup: mark by storing 0? keep
    // borrowed flag in [152]
    mov   w9, #1
    str   w9, [sp, #152]
    b     Lti_names_loop
Lti_tools_ok:
    str   wzr, [sp, #152]
Lti_names_loop:
    ldr   x0, [sp, #120]
    bl    _t_getiter
    cbz   x0, Lti_drop_toolslist
    str   x0, [sp, #128]
Lti_names_next:
    ldr   x0, [sp, #128]
    bl    _t_iter_next                      // t NEW
    cbz   x0, Lti_names_done
    str   x0, [sp, #136]
    // f = t['function']; name = f['name']
    mov   x0, #12
    bl    _val                              // 'function'
    mov   x1, x0
    ldr   x0, [sp, #136]
    bl    _t_dict_getwe
    cbz   x0, Lti_drop_t
    str   x0, [sp, #144]
    mov   x0, #13
    bl    _val                              // 'name'
    mov   x1, x0
    ldr   x0, [sp, #144]
    bl    _t_dict_getwe
    cbz   x0, Lti_drop_f
    mov   x1, x0                           // name borrowed
    ldr   x0, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #272]                   // PySet_Add
    blr   x9
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    b     Lti_names_next
Lti_names_done:
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    // self.names = nset
    mov   x0, x19
    adrp  x1, at_names@PAGE
    add   x1, x1, at_names@PAGEOFF
    ldr   x2, [sp, #112]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_toolslist
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   w9, [sp, #152]
    cbnz  w9, Lti_skip_tl
    ldr   x0, [sp, #120]
    bl    _t_decref
Lti_skip_tl:
    str   xzr, [sp, #120]
    // recipient = None; calls = []
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lti_err
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lti_err
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, at_calls@PAGE
    add   x1, x1, at_calls@PAGEOFF
    ldr   x2, [sp, #136]
    bl    _t_setattr_s
    cbnz  w0, Lti_drop_136
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // return None
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]
    b     Lti_out
// ---- cleanup / error ----
Lti_drop_f:
    ldr   x0, [sp, #144]
    bl    _t_decref
Lti_drop_t:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lti_drop_iter2
Lti_drop_iter2:
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    b     Lti_drop_toolslist
Lti_drop_toolslist:
    ldr   w9, [sp, #152]
    cbnz  w9, 1f
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
1:  str   xzr, [sp, #120]
Lti_drop_nset:
    ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  ldr   x0, [sp, #136]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #136]
3:  ldr   x0, [sp, #144]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #144]
4:  b     Lti_err
Lti_drop_136:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lti_err
Lti_drop_dicts:
    ldr   x0, [sp, #104]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #104]
5:  ldr   x0, [sp, #96]
    cbz   x0, 6f
    bl    _t_decref
    str   xzr, [sp, #96]
6:  b     Lti_err
Lti_drop_completed:
    ldr   x0, [sp, #96]
    bl    _t_decref
    b     Lti_err
Lti_drop_header:
    ldr   x0, [sp, #152]
    bl    _t_decref
Lti_drop_144:
    ldr   x0, [sp, #144]
    cbz   x0, 7f
    bl    _t_decref
    str   xzr, [sp, #144]
7:  b     Lti_err
Lti_err:
    mov   x0, #0
Lti_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #160
    ret

// _marker_check — x20 tokenizer, x25 expected, x26 name slot idx
//   -> w0 = 1 if token_to_id(name) == expected
// Saves x19/x20; clobbers x0-x3 only.
_marker_check:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    mov   x19, x26                         // slot idx
    // t2i = tokenizer.token_to_id
    mov   x0, x20
    adrp  x1, m_token_to_id@PAGE
    add   x1, x1, m_token_to_id@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lmkc_out0
    str   x0, [sp, #48]                     // t2i
    // nameobj = g_name_py[idx] (borrowed)
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x20, [x9, x19, lsl #3]
    // args = (nameobj,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lmkc_drop_t2i
    str   x0, [sp, #32]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #32]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    // result = t2i(nameobj)
    ldr   x0, [sp, #48]
    ldr   x1, [sp, #32]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #40]                     // result
    ldr   x0, [sp, #32]
    bl    _t_decref
    ldr   x0, [sp, #48]
    bl    _t_decref
    ldr   x0, [sp, #40]
    cbz   x0, Lmkc_out0                    // token_to_id may return None
    bl    _t_aslong
    mov   x20, x0                          // v
    ldr   x0, [sp, #40]
    bl    _t_decref
    cmp   x20, #-1
    b.eq  Lmkc_out0
    cmp   x20, x25
    b.ne  Lmkc_out0
    mov   w0, #1
    b     Lmkc_out
Lmkc_drop_t2i:
    ldr   x0, [sp, #48]
    bl    _t_decref
Lmkc_out0:
    mov   w0, #0
Lmkc_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// helper _startswith(x0 = s BORROWED, x1 = prefix str BORROWED) -> w0 1/0
_startswith:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    mov   x19, x1                          // prefix
    // m = s.startswith
    adrp  x1, m_startswith@PAGE
    add   x1, x1, m_startswith@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsw_out0
    mov   x20, x0
    // args = (prefix,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lsw_drop_m
    str   x0, [sp, #32]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #32]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    mov   x0, x20
    ldr   x1, [sp, #32]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #40]
    ldr   x0, [sp, #32]
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref
    ldr   x0, [sp, #40]
    cbz   x0, Lsw_out0
    bl    _t_istrue
    cmp   w0, #1
    b.ne  Lsw_out0
    mov   w0, #1
    b     Lsw_out
Lsw_drop_m:
    mov   x0, x20
    bl    _t_decref
Lsw_out0:
    mov   w0, #0
Lsw_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// _meth0(x0 = obj BORROWED, x1 = method C-string) -> NEW
_meth0:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    str   x19, [sp, #16]
    mov   x19, x0
    bl    _t_getattr_s
    cbz   x0, Lm0_out
    str   x0, [sp, #24]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lm0_drop_m
    str   x0, [sp, #32]
    ldr   x0, [sp, #24]
    ldr   x1, [sp, #32]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #40]
    ldr   x0, [sp, #32]
    bl    _t_decref
    ldr   x0, [sp, #24]
    bl    _t_decref
    ldr   x0, [sp, #40]
    b     Lm0_out
Lm0_drop_m:
    ldr   x0, [sp, #24]
    bl    _t_decref
    mov   x0, #0
Lm0_out:
    ldr   x19, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// _getitem_str(x0 = obj BORROWED, x1 = key str BORROWED) -> NEW
//   obj[key]
_getitem_str:
    stp   x29, x30, [sp, #-32]!
    mov   x29, sp
    str   x19, [sp, #16]
    mov   x19, x1
    bl    _t_seq_getitem
    ldp   x29, x30, [sp], #32
    ret

// _getdefault(x0 = dict BORROWED, x1 = key str BORROWED,
//             x2 = default BORROWED) -> NEW (dict.get(key, default))
_getdefault:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    // x19 dict  x20 key  x21 default  x22 getm
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    adrp  x1, m_get@PAGE
    add   x1, x1, m_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lgd_out
    mov   x22, x0                          // getm
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lgd_drop_m
    str   x0, [sp, #48]                     // args
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #48]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #48]
    mov   x1, #1
    mov   x2, x21
    bl    _t_tuple_setitem
    mov   x0, x22
    ldr   x1, [sp, #48]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #56]                     // park result
    ldr   x0, [sp, #48]
    bl    _t_decref                        // drop args
    mov   x0, x22
    bl    _t_decref                        // drop getm
    ldr   x0, [sp, #56]
    b     Lgd_drop_keep
Lgd_drop_m:
    mov   x0, x22
    bl    _t_decref
    mov   x0, #0
Lgd_drop_keep:
Lgd_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #64
    ret

// _slice_from(x0 = seq BORROWED, x1 = start int64) -> NEW (s[start:])
_slice_from:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    // x19 seq  x20 start
    mov   x19, x0
    mov   x20, x1
    mov   x0, x19
    bl    _t_size
    cmp   x0, #0
    b.ge  1f
    // length error: use PY_SSIZE_T_MAX = -1 >>> 1 (logical)
    mov   x2, #-1
    lsr   x2, x2, #1
    b     2f
1:  mov   x2, x0                          // stop = len
2:  mov   x0, x19
    mov   x1, x20
    bl    _t_getslice
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// _dd_commit_body(x0 = self) -> 0 ok / -1 error (ValueError set)
//   if self.recipient: raise
//   key = 'analysis' if self.channel=='analysis' else 'content'
//   self.completed[key] += self._decode(self.body)
//   self.body = []
// x19 self  x20 key  x21 cur  x22 dec  x23 new
// ============================================================
_dd_commit_body:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    mov   x19, x0
    // ---- recipient truthy? ----
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_err
    str   x0, [sp, #64]
    bl    _t_istrue
    mov   x24, x0
    ldr   x0, [sp, #64]
    bl    _t_decref
    cmp   x24, #1
    b.eq  Lcb_handoff_bad
    // ---- key by channel ----
    mov   x0, x19
    adrp  x1, at_channel@PAGE
    add   x1, x1, at_channel@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_err
    str   x0, [sp, #64]
    adrp  x1, v_analysis_cs@PAGE
    add   x1, x1, v_analysis_cs@PAGEOFF
    bl    _t_strcmp                         // channel == 'analysis'?
    mov   x24, x0
    ldr   x0, [sp, #64]
    bl    _t_decref
    mov   x0, #7                            // 'analysis'
    cmp   x24, #0
    b.eq  1f
    mov   x0, #6                            // 'content'
1:  bl    _val
    mov   x20, x0                           // key (borrowed)
    // ---- cur = completed[key] ----
    mov   x0, x19
    adrp  x1, at_completed@PAGE
    add   x1, x1, at_completed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_err
    str   x0, [sp, #72]                     // completed dict
    ldr   x0, [sp, #72]
    mov   x1, x20
    bl    _t_dict_getwe
    cbz   x0, Lcb_drop_comp
    mov   x21, x0                          // cur NEW
    // ---- dec = _decode(self, self.body) ----
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcb_drop_cur
    str   x0, [sp, #80]
    mov   x0, x19
    ldr   x1, [sp, #80]
    bl    _decode
    cbz   x0, Lcb_drop_bd
    mov   x22, x0                          // dec NEW
    // ---- new = cur + dec ----
    mov   x0, x21
    mov   x1, x22
    bl    _t_concat
    cbz   x0, Lcb_drop_dec
    mov   x23, x0                          // new NEW
    // ---- completed[key] = new ----
    ldr   x0, [sp, #72]
    mov   x1, x20
    mov   x2, x23
    bl    _t_dict_setitem
    // ---- self.body = [] ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lcb_drop_new
    str   x0, [sp, #88]
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    ldr   x2, [sp, #88]
    bl    _t_setattr_s
    ldr   x0, [sp, #88]
    bl    _t_decref
    // ---- drop temps; success ----
    mov   x0, x23
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    ldr   x0, [sp, #80]
    bl    _t_decref
    ldr   x0, [sp, #72]
    bl    _t_decref
    mov   w0, #0
    b     Lcb_out
Lcb_handoff_bad:
    adrp  x0, m_need_handoff@PAGE
    add   x0, x0, m_need_handoff@PAGEOFF
    bl    _raise_value
    mov   w0, #-1
    b     Lcb_out
Lcb_drop_new:
    mov   x0, x23
    bl    _t_decref
Lcb_drop_dec:
    mov   x0, x22
    bl    _t_decref
Lcb_drop_cur:
    mov   x0, x21
    bl    _t_decref
Lcb_drop_bd:
    ldr   x0, [sp, #80]
    bl    _t_decref
    str   xzr, [sp, #80]
    ldr   x0, [sp, #72]
    bl    _t_decref
    str   xzr, [sp, #72]
Lcb_drop_comp:
    ldr   x0, [sp, #72]
    bl    _t_decref
Lcb_err:
    mov   w0, #-1
Lcb_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// _dd_begin_body(x0 = self) -> 0 ok / -1 error
// x19 self  x20 role  x21 channel  x22 formatting  x23 fields
// x24 recipients  x25 recipient  x26 recipient_name
// Parks: [96]header [104]dec [112]prefix [120]a [128]b [136]iter
//        [144]item [152]c [160]d
// ============================================================
_dd_begin_body:
    stp   x29, x30, [sp, #-192]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    stp   xzr, xzr, [sp, #128]
    stp   xzr, xzr, [sp, #144]
    stp   xzr, xzr, [sp, #160]
    mov   x20, #0
    mov   x21, #0
    mov   x22, #0
    mov   x23, #0
    mov   x24, #0
    mov   x25, #0
    mov   x26, #0
    mov   x19, x0
    // ---- header / prefix ----
    mov   x0, x19
    adrp  x1, at_header@PAGE
    add   x1, x1, at_header@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lbb_err
    str   x0, [sp, #96]
    mov   x0, x19
    adrp  x1, at_prefix@PAGE
    add   x1, x1, at_prefix@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lbb_err
    str   x0, [sp, #112]
    // ---- role = (prefix + decode(header['role'])).split() ----
    mov   x0, #8
    bl    _val                              // 'role'
    mov   x1, x0
    ldr   x0, [sp, #96]
    bl    _t_dict_getwe
    cbz   x0, Lbb_err
    str   x0, [sp, #120]
    mov   x0, x19
    ldr   x1, [sp, #120]
    bl    _decode
    cbz   x0, Lbb_drop120
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    bl    _t_concat
    cbz   x0, Lbb_drop104
    str   x0, [sp, #128]
    ldr   x0, [sp, #128]
    adrp  x1, m_split@PAGE
    add   x1, x1, m_split@PAGEOFF
    bl    _meth0
    cbz   x0, Lbb_drop128
    mov   x20, x0                          // role
    // ---- channel = decode(header.get('channel', [])).split() ----
    mov   x0, #19
    bl    _val                              // 'channel' key
    mov   x1, x0
    adrp  x9, g_iface@PAGE
    ldr   x2, [x9, g_empty_list@PAGEOFF]
    ldr   x0, [sp, #96]
    bl    _getdefault
    cbz   x0, Lbb_drop_role
    str   x0, [sp, #120]
    mov   x0, x19
    ldr   x1, [sp, #120]
    bl    _decode
    cbz   x0, Lbb_drop120
    str   x0, [sp, #104]
    ldr   x0, [sp, #104]
    adrp  x1, m_split@PAGE
    add   x1, x1, m_split@PAGEOFF
    bl    _meth0
    cbz   x0, Lbb_drop104
    mov   x21, x0                          // channel
    // ---- formatting = decode(header.get('format', [])).strip() ----
    mov   x0, #20
    bl    _val                              // 'format' key
    mov   x1, x0
    adrp  x9, g_iface@PAGE
    ldr   x2, [x9, g_empty_list@PAGEOFF]
    ldr   x0, [sp, #96]
    bl    _getdefault
    cbz   x0, Lbb_drop_ch
    str   x0, [sp, #120]
    mov   x0, x19
    ldr   x1, [sp, #120]
    bl    _decode
    cbz   x0, Lbb_drop120
    str   x0, [sp, #104]
    ldr   x0, [sp, #104]
    adrp  x1, m_strip@PAGE
    add   x1, x1, m_strip@PAGEOFF
    bl    _meth0
    cbz   x0, Lbb_drop104
    mov   x22, x0                          // formatting
    // ---- fields = role[1:] + channel[1:] ----
    mov   x0, x20
    mov   x1, #1
    bl    _slice_from
    cbz   x0, Lbb_drop_fmt
    str   x0, [sp, #120]                     // role slice OWNED
    mov   x0, x21
    mov   x1, #1
    bl    _slice_from
    cbz   x0, Lbb_drop120
    str   x0, [sp, #176]                     // channel slice OWNED
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #176]
    bl    _t_seq_concat
    str   x0, [sp, #128]                     // fields NEW
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    ldr   x0, [sp, #128]
    cbz   x0, Lbb_drop_fmt
    mov   x23, x0                          // fields
    str   xzr, [sp, #128]
    // ---- recipients = [x[3:] for x in fields if x.startswith('to=')] ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lbb_drop_fmt
    mov   x24, x0                          // recipients
    mov   x0, x23
    bl    _t_getiter
    cbz   x0, Lbb_drop_fmt
    str   x0, [sp, #136]
Lbb_r_loop:
    ldr   x0, [sp, #136]
    bl    _t_iter_next                      // x NEW
    cbz   x0, Lbb_r_done
    str   x0, [sp, #144]
    mov   x0, #15
    bl    _val                              // 'to='
    mov   x1, x0
    ldr   x0, [sp, #144]
    bl    _startswith
    cmp   w0, #1
    b.ne  Lbb_r_next
    ldr   x0, [sp, #144]
    mov   x1, #3
    bl    _slice_from
    cbz   x0, Lbb_drop_item
    str   x0, [sp, #152]
    mov   x0, x24
    ldr   x1, [sp, #152]
    bl    _t_list_append
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
Lbb_r_next:
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    b     Lbb_r_loop
Lbb_r_done:
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // continue in part B
    b     Lbb_partB
Lbb_partB:
    // ---- validation: not role / role[0]!='assistant' ----
    mov   x0, x20
    bl    _t_size
    cmp   x0, #0
    b.le  Lbb_bad_header
    mov   x0, x20
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lbb_bad_header
    str   x0, [sp, #128]
    adrp  x1, v_assistant@PAGE
    add   x1, x1, v_assistant@PAGEOFF
    ldr   x0, [sp, #128]
    bl    _t_strcmp
    mov   x27, x0
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    cmp   x27, #0
    b.ne  Lbb_bad_header
    // ---- len(recipients) > 1 ----
    mov   x0, x24
    bl    _t_size
    cmp   x0, #1
    b.gt  Lbb_bad_header
    // ---- any(x != 'json' and not x.startswith('to=')) ----
    mov   x0, x23
    bl    _t_getiter
    cbz   x0, Lbb_err
    str   x0, [sp, #136]
Lbb_v_loop:
    ldr   x0, [sp, #136]
    bl    _t_iter_next
    cbz   x0, Lbb_v_done
    str   x0, [sp, #144]
    mov   x0, #15
    bl    _val                              // 'to='
    mov   x1, x0
    ldr   x0, [sp, #144]
    bl    _startswith
    cmp   w0, #1
    b.eq  Lbb_v_next
    // x != 'json'?
    adrp  x1, v_json@PAGE
    add   x1, x1, v_json@PAGEOFF
    ldr   x0, [sp, #144]
    bl    _t_strcmp
    cbz   x0, Lbb_v_next                    // == 'json' -> ok
    // bad field
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    b     Lbb_bad_header
Lbb_v_next:
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    b     Lbb_v_loop
Lbb_v_done:
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // ---- recipient = recipients[0] if recipients else 'all' ----
    mov   x0, x24
    bl    _t_size
    cmp   x0, #0
    b.le  1f
    mov   x0, x24
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lbb_err
    mov   x25, x0
    b     2f
1:  mov   x0, #5
    bl    _val                              // 'all' borrowed
    mov   x0, x0
    bl    _incref
    mov   x25, x0                          // owned ref to 'all'
2:
    // ---- self.recipient = None ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lbb_err
    // ---- if recipient != 'all' ----
    adrp  x1, v_all@PAGE
    add   x1, x1, v_all@PAGEOFF
    mov   x0, x25
    bl    _t_strcmp
    cbz   x0, Lbb_recv_all
    // startswith 'functions.'?
    mov   x0, #16
    bl    _val
    mov   x1, x0
    mov   x0, x25
    bl    _startswith
    cmp   w0, #1
    b.ne  Lbb_not_declared
    // rname = recipient[10:]
    mov   x0, x25
    mov   x1, #10
    bl    _slice_from
    cbz   x0, Lbb_err
    mov   x26, x0
    // rname in self.names?
    mov   x0, x19
    adrp  x1, at_names@PAGE
    add   x1, x1, at_names@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lbb_drop_rname
    str   x0, [sp, #128]
    ldr   x0, [sp, #128]
    mov   x1, x26
    bl    _t_contains
    mov   x27, x0
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    cmp   x27, #1
    b.ne  Lbb_not_declared2
    // self.recipient = rname
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    mov   x2, x26
    bl    _t_setattr_s
    cbnz  w0, Lbb_err
Lbb_recv_all:
    // ---- chan = channel[0] if channel else 'final' ----
    mov   x0, x21
    bl    _t_size
    cmp   x0, #0
    b.le  1f
    mov   x0, x21
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lbb_err
    str   x0, [sp, #128]                     // chan NEW
    b     2f
1:  mov   x0, #4
    bl    _val                              // 'final' borrowed
    bl    _incref
    str   x0, [sp, #128]                     // chan owned
2:  // ---- chan in analysis/final/commentary? ----
    adrp  x1, v_analysis_cs@PAGE
    add   x1, x1, v_analysis_cs@PAGEOFF
    ldr   x0, [sp, #128]
    bl    _t_strcmp
    cbz   x0, Lbb_chan_ok
    adrp  x1, v_final@PAGE
    add   x1, x1, v_final@PAGEOFF
    ldr   x0, [sp, #128]
    bl    _t_strcmp
    cbz   x0, Lbb_chan_ok
    adrp  x1, v_commentary@PAGE
    add   x1, x1, v_commentary@PAGEOFF
    ldr   x0, [sp, #128]
    bl    _t_strcmp
    cbnz  x0, Lbb_bad_chan
Lbb_chan_ok:
    // ---- formatting in ('', 'json')? ----
    adrp  x1, c_empty@PAGE
    add   x1, x1, c_empty@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbz   x0, Lbb_fmt_ok
    adrp  x1, v_json@PAGE
    add   x1, x1, v_json@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, Lbb_bad_chan
Lbb_fmt_ok:
    // ---- self.channel = chan ----
    mov   x0, x19
    adrp  x1, at_channel@PAGE
    add   x1, x1, at_channel@PAGEOFF
    ldr   x2, [sp, #128]
    bl    _t_setattr_s
    cbnz  w0, Lbb_err
    // ---- recipient and chan != 'commentary' ----
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lbb_err
    str   x0, [sp, #136]
    bl    _t_istrue
    mov   x27, x0
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    cmp   x27, #1
    b.ne  Lbb_phase_body
    adrp  x1, v_commentary@PAGE
    add   x1, x1, v_commentary@PAGEOFF
    ldr   x0, [sp, #128]
    bl    _t_strcmp
    cbnz  x0, Lbb_need_commentary
Lbb_phase_body:
    // ---- self.phase = 'body'; self.body = [] ----
    mov   x0, #2
    bl    _val
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lbb_err
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lbb_err
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    ldr   x2, [sp, #136]
    bl    _t_setattr_s
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x28, #0                          // result parked across cleanup
    b     Lbb_cleanup_ok
// ---- error exits ----
Lbb_bad_header:
    adrp  x0, m_bad_header@PAGE
    add   x0, x0, m_bad_header@PAGEOFF
    bl    _raise_value
    b     Lbb_err
Lbb_not_declared:
Lbb_not_declared2:
    adrp  x0, m_not_declared@PAGE
    add   x0, x0, m_not_declared@PAGEOFF
    bl    _raise_value
    b     Lbb_err
Lbb_bad_chan:
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    adrp  x0, m_bad_chan@PAGE
    add   x0, x0, m_bad_chan@PAGEOFF
    bl    _raise_value
    b     Lbb_err
Lbb_need_commentary:
    adrp  x0, m_chan_comment@PAGE
    add   x0, x0, m_chan_comment@PAGEOFF
    bl    _raise_value
    b     Lbb_err
Lbb_drop_rname:
    mov   x0, x26
    bl    _t_decref
    mov   x26, #0
    b     Lbb_err
// part-A intermediate drops — the Lbb_err cleanup drops every park,
// so these are pure aliases
Lbb_drop_item:
Lbb_drop120:
Lbb_drop104:
Lbb_drop128:
Lbb_drop_role:
Lbb_drop_ch:
Lbb_drop_fmt:
Lbb_err:
    mov   x28, #-1                         // result parked across cleanup
Lbb_cleanup_ok:
    cmp   x26, #0
    b.eq  1f
    mov   x0, x26
    bl    _t_decref
    mov   x26, #0
1:  cmp   x25, #0
    b.eq  2f
    mov   x0, x25
    bl    _t_decref
    mov   x25, #0
2:  cmp   x24, #0
    b.eq  3f
    mov   x0, x24
    bl    _t_decref
    mov   x24, #0
3:  cmp   x23, #0
    b.eq  4f
    mov   x0, x23
    bl    _t_decref
    mov   x23, #0
4:  cmp   x22, #0
    b.eq  5f
    mov   x0, x22
    bl    _t_decref
    mov   x22, #0
5:  cmp   x21, #0
    b.eq  6f
    mov   x0, x21
    bl    _t_decref
    mov   x21, #0
6:  cmp   x20, #0
    b.eq  7f
    mov   x0, x20
    bl    _t_decref
    mov   x20, #0
    // ---- drop parks ----
7:  ldr   x0, [sp, #176]
    cbz   x0, 70f
    bl    _t_decref
    str   xzr, [sp, #176]
70: ldr   x0, [sp, #160]
    cbz   x0, 8f
    bl    _t_decref
    str   xzr, [sp, #160]
8:  ldr   x0, [sp, #152]
    cbz   x0, 9f
    bl    _t_decref
    str   xzr, [sp, #152]
9:  ldr   x0, [sp, #144]
    cbz   x0, 10f
    bl    _t_decref
    str   xzr, [sp, #144]
10: ldr   x0, [sp, #136]
    cbz   x0, 11f
    bl    _t_decref
    str   xzr, [sp, #136]
11: ldr   x0, [sp, #128]
    cbz   x0, 12f
    bl    _t_decref
    str   xzr, [sp, #128]
12: ldr   x0, [sp, #120]
    cbz   x0, 13f
    bl    _t_decref
    str   xzr, [sp, #120]
13: ldr   x0, [sp, #112]
    cbz   x0, 14f
    bl    _t_decref
    str   xzr, [sp, #112]
14: ldr   x0, [sp, #104]
    cbz   x0, 15f
    bl    _t_decref
    str   xzr, [sp, #104]
15: ldr   x0, [sp, #96]
    cbz   x0, 16f
    bl    _t_decref
    str   xzr, [sp, #96]
16: mov   w0, w28                          // result survives cleanup
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #192
    ret

// ============================================================
// PyObject* _dd_feed(x0 = self, x1 = token obj BORROWED,
//                    x2 = final bool BORROWED) -> 3-tuple NEW / NULL
// x19 self  x20 t64  x21 final flag  x22 marker  x23 terminal
// x24 phase(borrowed) x25 misc
// Parks: [96]kobj [104]content [112]reasoning [120]tuple/scratch
//        [128]a [136]b [144]c [152]d [160]e [168]f [176]result
// ============================================================
_dd_feed:
    stp   x29, x30, [sp, #-208]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    stp   xzr, xzr, [sp, #128]
    stp   xzr, xzr, [sp, #144]
    stp   xzr, xzr, [sp, #160]
    stp   xzr, xzr, [sp, #176]
    mov   x19, x0
    str   x1, [sp, #184]                    // token obj (borrowed, for appends)
    mov   x23, #0                           // terminal = 0
    // ---- final flag (convert NOW: x2 dies at the first C call) ----
    mov   x0, x2
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lfd2_err
    mov   w21, w0                           // final
    // ---- closed latch ----
    mov   x0, x19
    adrp  x1, at_closed@PAGE
    add   x1, x1, at_closed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    bl    _t_istrue
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cmp   x25, #1
    b.eq  Lfd2_continued
    // ---- t64 / kobj ----
    ldr   x0, [sp, #184]
    bl    _t_aslong
    mov   x20, x0
    bl    _t_fromlong
    cbz   x0, Lfd2_err
    str   x0, [sp, #96]
    // ---- marker = markers.get(t64) ----
    adrp  x9, g_markers@PAGE
    ldr   x0, [x9, g_markers@PAGEOFF]
    ldr   x1, [sp, #96]
    bl    _t_dict_getwe
    mov   x22, x0                          // marker borrowed (0 = None)
    // ================= dispatch =================
    cbz   x22, Lfd2_not_marker
    // ---- 'start'? ----
    adrp  x1, a_start@PAGE
    add   x1, x1, a_start@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 1f
    b     Lfd2_m_start
1:  // ---- 'message'? ----
    adrp  x1, a_message@PAGE
    add   x1, x1, a_message@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 2f
    b     Lfd2_m_message
2:  // ---- 'end'? ----
    adrp  x1, a_end@PAGE
    add   x1, x1, a_end@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 3f
    b     Lfd2_m_end
3:  // ---- 'return'? ----
    adrp  x1, a_return@PAGE
    add   x1, x1, a_return@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 4f
    b     Lfd2_m_return
4:  // ---- 'eot'? ----
    adrp  x1, a_eot@PAGE
    add   x1, x1, a_eot@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 5f
    b     Lfd2_m_return                     // same handler as return
5:  // ---- 'call'? ----
    adrp  x1, a_call@PAGE
    add   x1, x1, a_call@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbnz  x0, 6f
    b     Lfd2_m_call
6:  // ---- 'channel' or 'format'? ----
    adrp  x1, a_channel@PAGE
    add   x1, x1, a_channel@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbz   x0, Lfd2_m_meta
    adrp  x1, a_format@PAGE
    add   x1, x1, a_format@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbz   x0, Lfd2_m_meta
    // unknown marker kind: treat as... (cannot happen; markers built
    // from the fixed table) — fall through to not_marker
    b     Lfd2_not_marker

// ============ 'start' handler ============
Lfd2_m_start:
    // phase must be 'after'
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_after@PAGE
    add   x1, x1, v_after@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_bad_start
    // phase='header'; prefix=''; header={'role':[]}; field='role';
    // header_count=0
    mov   x0, #0
    bl    _val                              // 'header'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_err
    adrp  x0, c_empty@PAGE
    add   x0, x0, c_empty@PAGEOFF
    bl    _t_fromstring                     // '' NEW
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    mov   x0, x19
    adrp  x1, at_prefix@PAGE
    add   x1, x1, at_prefix@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    bl    _t_dict_new
    cbz   x0, Lfd2_err
    str   x0, [sp, #128]                     // new header
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lfd2_drop_hdrA
    str   x0, [sp, #136]
    mov   x0, #8
    bl    _val                              // 'role'
    mov   x1, x0
    ldr   x0, [sp, #128]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    mov   x0, x19
    adrp  x1, at_header@PAGE
    add   x1, x1, at_header@PAGEOFF
    ldr   x2, [sp, #128]
    bl    _t_setattr_s
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    mov   x0, #8
    bl    _val                              // 'role'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_field@PAGE
    add   x1, x1, at_field@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_err
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    mov   x0, x19
    adrp  x1, at_hcount@PAGE
    add   x1, x1, at_hcount@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    b     Lfd2_tail

// ============ 'channel'/'format' handler ============
Lfd2_m_meta:
    // phase must be 'header' and marker not in header
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_header@PAGE
    add   x1, x1, v_header@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_bad_meta
    mov   x0, x19
    adrp  x1, at_header@PAGE
    add   x1, x1, at_header@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]                     // header
    ldr   x0, [sp, #120]
    mov   x1, x22                           // marker kind str borrowed
    bl    _t_dict_contains
    cmp   w0, #1
    b.eq  Lfd2_bad_meta
    // header[marker] = []; field = marker
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lfd2_drop_metaA
    str   x0, [sp, #136]
    ldr   x0, [sp, #120]
    mov   x1, x22
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    mov   x0, x19
    adrp  x1, at_field@PAGE
    add   x1, x1, at_field@PAGEOFF
    mov   x2, x22
    bl    _t_setattr_s
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    b     Lfd2_tail

// ============ 'message' handler ============
Lfd2_m_message:
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_header@PAGE
    add   x1, x1, v_header@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_bad_sep
    mov   x0, x19
    bl    _dd_begin_body
    cmp   w0, #0
    b.lt  Lfd2_err
    b     Lfd2_tail

// ============ 'end' handler ============
Lfd2_m_end:
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_body@PAGE
    add   x1, x1, v_body@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_bad_end
    mov   x0, x19
    bl    _dd_commit_body
    cmp   w0, #0
    b.lt  Lfd2_err
    mov   x0, #1
    bl    _val                              // 'after'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_err
    b     Lfd2_tail

// ============ 'return'/'eot' handler ============
Lfd2_m_return:
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_body@PAGE
    add   x1, x1, v_body@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    cbnz  x25, 1f
    // phase == body -> commit
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    mov   x0, x19
    bl    _dd_commit_body
    cmp   w0, #0
    b.lt  Lfd2_err
    b     2f
1:  ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    // not body: if marker == 'return' -> raise
    adrp  x1, a_return@PAGE
    add   x1, x1, a_return@PAGEOFF
    mov   x0, x22
    bl    _t_strcmp
    cbz   x0, Lfd2_ret_nobody
2:  mov   x0, #1
    bl    _val                              // 'after'
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_err
    mov   x23, #1                           // terminal = True
    b     Lfd2_tail

// ============ 'call' handler ============
Lfd2_m_call:
    // phase must be 'body'
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_body@PAGE
    add   x1, x1, v_body@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_bad_handoff
    // recipient truthy?
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]                     // recipient owned
    bl    _t_istrue
    cmp   w0, #1
    b.ne  Lfd2_drop_recv_h
    // args_str = _decode(self, self.body)
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop_recv
    str   x0, [sp, #104]                     // body OWNED
    mov   x0, x19
    ldr   x1, [sp, #104]
    bl    _decode
    str   x0, [sp, #128]                     // args_str NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #128]
    cbz   x0, Lfd2_drop_recv
    // arguments = _json_object(args_str)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop128
    str   x0, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #304]                   // _json_object
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #160]                     // park arguments (reuses slot)
    ldr   x0, [sp, #136]
    bl    _t_decref                          // drop args tuple
    str   xzr, [sp, #136]
    ldr   x0, [sp, #160]
    cbz   x0, Lfd2_drop128
    str   x0, [sp, #136]                     // arguments NEW at [136]
    // uuid4 = uuid.uuid4(); hexm = uuid4.hex
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #320]                   // uuid.uuid4
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop136
    str   x0, [sp, #144]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #320]                   // uuid.uuid4
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    mov   x21, x0                           // uuid4 NEW (park in x21; free here)
    ldr   x0, [sp, #144]
    bl    _t_decref                         // drop empty args tuple
    cbnz  x21, Lfd2_uuid_ok
    b     Lfd2_drop136
Lfd2_uuid_ok:
    mov   x0, x21
    str   x0, [sp, #144]                     // uuid4
    ldr   x0, [sp, #144]
    adrp  x1, m_hex@PAGE
    add   x1, x1, m_hex@PAGEOFF
    bl    _t_getattr_s                       // uuid4.hex is an ATTRIBUTE (str), not a method
    str   x0, [sp, #152]                     // hexm NEW
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    ldr   x0, [sp, #152]
    cbz   x0, Lfd2_drop136
    // callid = 'call_' + hex
    mov   x0, #9
    bl    _val                              // 'call_'
    ldr   x1, [sp, #152]
    bl    _t_concat
    cbz   x0, Lfd2_drop152
    str   x0, [sp, #160]                     // callid
    // argjson = json.dumps(arguments, **g_kw_dumps)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop160
    str   x0, [sp, #104]                     // dumps args tuple ([104] free here)
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #328]                   // json.dumps
    ldr   x1, [sp, #104]                     // (arguments,)
    adrp  x9, g_kw_dumps@PAGE
    ldr   x2, [x9, g_kw_dumps@PAGEOFF]
    bl    _t_call
    str   x0, [sp, #168]                     // argjson
    ldr   x0, [sp, #104]
    bl    _t_decref                          // drop dumps args tuple
    str   xzr, [sp, #104]
    ldr   x0, [sp, #168]
    cbz   x0, Lfd2_drop160
    // fdict = {'name': recipient, 'arguments': argjson}
    bl    _t_dict_new
    cbz   x0, Lfd2_drop168
    str   x0, [sp, #176]                     // fdict
    mov   x0, #13
    bl    _val                              // 'name'
    mov   x1, x0
    ldr   x0, [sp, #176]
    ldr   x2, [sp, #120]
    bl    _t_dict_setitem
    mov   x0, #14
    bl    _val                              // 'arguments'
    mov   x1, x0
    ldr   x0, [sp, #176]
    ldr   x2, [sp, #168]
    bl    _t_dict_setitem
    // cdict = {'id': callid, 'type': 'function', 'function': fdict}
    bl    _t_dict_new
    cbz   x0, Lfd2_drop176
    str   x0, [sp, #192]                     // cdict (park [192])
    mov   x0, #10
    bl    _val                              // 'id'
    mov   x1, x0
    ldr   x0, [sp, #192]
    ldr   x2, [sp, #160]
    bl    _t_dict_setitem
    // 'function': fdict
    mov   x0, #12
    bl    _val                              // 'function' key
    mov   x1, x0
    ldr   x0, [sp, #192]
    ldr   x2, [sp, #176]
    bl    _t_dict_setitem
    // 'type': 'function'
    mov   x0, #11
    bl    _val                              // 'type' key
    mov   x1, x0
    adrp  x9, g_val_py@PAGE
    add   x9, x9, g_val_py@PAGEOFF
    ldr   x2, [x9, #96]                     // val(12) = 'function' value
    ldr   x0, [sp, #192]
    bl    _t_dict_setitem
    // calls.append(cdict)
    mov   x0, x19
    adrp  x1, at_calls@PAGE
    add   x1, x1, at_calls@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop192
    str   x0, [sp, #144]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #192]
    bl    _t_list_append
    ldr   x0, [sp, #144]
    bl    _t_decref
    str   xzr, [sp, #144]
    // drop call-struct temps
    ldr   x0, [sp, #192]
    bl    _t_decref
    str   xzr, [sp, #192]
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    ldr   x0, [sp, #168]
    bl    _t_decref
    str   xzr, [sp, #168]
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    // body = []; phase = 'after'; terminal = 1
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    mov   x0, #1
    bl    _val
    mov   x2, x0
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_err
    mov   x23, #1
    b     Lfd2_tail

// ============ non-marker token ============
Lfd2_not_marker:
    // token >= 199998 -> raise
    movz  x9, #0x0d3e
    movk  x9, #0x3, lsl #16
    cmp   x20, x9
    b.ge  Lfd2_bad_control
    // phase dispatch
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    adrp  x1, v_header@PAGE
    add   x1, x1, v_header@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    cbz   x0, Lfd2_hdr_token
    adrp  x1, v_body@PAGE
    add   x1, x1, v_body@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    cbz   x0, Lfd2_body_token
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    b     Lfd2_outside
Lfd2_hdr_token:
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    // header_count += 1
    mov   x0, x19
    adrp  x1, at_hcount@PAGE
    add   x1, x1, at_hcount@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    bl    _t_aslong
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    add   x25, x25, #1
    cmp   x25, #256
    b.gt  Lfd2_h_limit
    mov   x0, x25                          // count (x0 was clobbered by decref)
    bl    _t_fromlong
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    mov   x0, x19
    adrp  x1, at_hcount@PAGE
    add   x1, x1, at_hcount@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    // header[field].append(token)
    mov   x0, x19
    adrp  x1, at_header@PAGE
    add   x1, x1, at_header@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #128]
    mov   x0, x19
    adrp  x1, at_field@PAGE
    add   x1, x1, at_field@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop128
    str   x0, [sp, #136]                     // field key
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #136]
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop136
    bl    _incref
    str   x0, [sp, #144]                     // hlist
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #184]
    bl    _t_list_append
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    b     Lfd2_tail
Lfd2_body_token:
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #184]
    bl    _t_list_append
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    b     Lfd2_tail

// helper _tuple_item(x0 = tuple, x1 = index) -> borrowed item
_tuple_item:
    add   x0, x0, #24
    ldr   x0, [x0, x1, lsl #3]
    ret

// ================= feed tail =================
Lfd2_tail:
    // decoded = dict(self.completed)   (drop kobj parked at [96] first)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x19
    adrp  x1, at_completed@PAGE
    add   x1, x1, at_completed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_err
    str   x0, [sp, #120]
    ldr   x0, [sp, #120]
    bl    _t_dict_copy
    str   x0, [sp, #96]                      // decoded
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #96]
    cbz   x0, Lfd2_err
    // if phase == 'body' and not self.recipient
    mov   x0, x19
    adrp  x1, at_phase@PAGE
    add   x1, x1, at_phase@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop_dec
    str   x0, [sp, #120]
    adrp  x1, v_body@PAGE
    add   x1, x1, v_body@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cbnz  x25, Lfd2_after_bodykey
    mov   x0, x19
    adrp  x1, at_recipient@PAGE
    add   x1, x1, at_recipient@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop_dec
    str   x0, [sp, #120]
    bl    _t_istrue
    mov   w26, w0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    cmp   x26, #1
    b.eq  Lfd2_after_bodykey
    // key = 'analysis' if channel=='analysis' else 'content'
    mov   x0, x19
    adrp  x1, at_channel@PAGE
    add   x1, x1, at_channel@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop_dec
    str   x0, [sp, #120]
    adrp  x1, v_analysis_cs@PAGE
    add   x1, x1, v_analysis_cs@PAGEOFF
    ldr   x0, [sp, #120]
    bl    _t_strcmp
    mov   x25, x0
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    mov   x0, #7
    cbz   x25, 1f
    mov   x0, #6
1:  bl    _val
    mov   x24, x0                           // key borrowed
    // decoded[key] += _decode(self.body)
    ldr   x0, [sp, #96]
    mov   x1, x24
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop_dec
    bl    _incref
    str   x0, [sp, #128]                     // cur
    mov   x0, x19
    adrp  x1, at_body@PAGE
    add   x1, x1, at_body@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop128t
    str   x0, [sp, #176]                     // body OWNED
    mov   x0, x19
    ldr   x1, [sp, #176]
    bl    _decode
    str   x0, [sp, #120]                     // dec NEW
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    ldr   x0, [sp, #120]
    cbz   x0, Lfd2_drop128t
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #120]
    bl    _t_concat
    cbz   x0, Lfd2_drop120t
    str   x0, [sp, #136]                     // new
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #96]
    mov   x1, x24
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
Lfd2_after_bodykey:
    // fflag = final or terminal
    orr   w26, w21, w23
    // ---- content text_delta ----
    mov   x0, #6
    bl    _val
    mov   x1, x0
    ldr   x0, [sp, #96]
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop_dec
    bl    _incref
    str   x0, [sp, #128]                     // dcontent
    mov   x0, x19
    adrp  x1, at_emitted@PAGE
    add   x1, x1, at_emitted@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop128t
    str   x0, [sp, #136]                     // emitted
    mov   x0, #6
    bl    _val
    mov   x1, x0
    ldr   x0, [sp, #136]
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop136t
    str   x0, [sp, #144]                     // econtent borrowed
    mov   x0, x19
    adrp  x1, at_stops@PAGE
    add   x1, x1, at_stops@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop136t
    str   x0, [sp, #152]                     // stops borrowed
    // args = (dcontent, econtent, stops, fbool)
    mov   x0, #4
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop152t
    str   x0, [sp, #160]
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #0
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #1
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #152]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #2
    ldr   x2, [sp, #152]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #280]                   // True
    cmp   w26, #0
    b.ne  1f
    ldr   x2, [x10, #288]                   // False
1:  ldr   x0, [sp, #160]
    mov   x1, #3
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #296]                   // text_delta
    ldr   x1, [sp, #160]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                     // tdresult
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    ldr   x0, [sp, #104]
    cbz   x0, Lfd2_drop152t
    // content = tdresult[0]; stopped = tdresult[2]
    ldr   x0, [sp, #104]
    mov   x1, #0
    bl    _tuple_item
    bl    _incref
    str   x0, [sp, #112]                     // content owned
    ldr   x0, [sp, #104]
    mov   x1, #2
    bl    _tuple_item
    bl    _incref
    str   x0, [sp, #160]
    ldr   x0, [sp, #160]
    bl    _t_istrue
    mov   w27, w0                           // stopped_c
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    // emitted['content'] = tdresult[1]
    ldr   x0, [sp, #104]
    mov   x1, #1
    bl    _tuple_item
    bl    _incref
    mov   x2, x0
    mov   x0, #6
    bl    _val                              // 'content'
    mov   x1, x0
    ldr   x0, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    // ---- reasoning text_delta ----
    mov   x0, #7
    bl    _val
    mov   x1, x0
    ldr   x0, [sp, #96]
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop_dec
    bl    _incref
    str   x0, [sp, #128]                     // danalysis
    mov   x0, x19
    adrp  x1, at_emitted@PAGE
    add   x1, x1, at_emitted@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfd2_drop128t
    str   x0, [sp, #136]                     // emitted2
    mov   x0, #7
    bl    _val
    mov   x1, x0
    ldr   x0, [sp, #136]
    bl    _t_dict_getwe
    cbz   x0, Lfd2_drop136t
    str   x0, [sp, #144]                     // eanalysis borrowed
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop136t
    str   x0, [sp, #152]                     // empty tuple
    mov   x0, #4
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop152t
    str   x0, [sp, #160]
    ldr   x0, [sp, #128]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #0
    ldr   x2, [sp, #128]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #1
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #152]
    bl    _incref
    ldr   x0, [sp, #160]
    mov   x1, #2
    ldr   x2, [sp, #152]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #280]                   // True
    cmp   w26, #0
    b.ne  1f
    ldr   x2, [x10, #288]
1:  ldr   x0, [sp, #160]
    mov   x1, #3
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #296]                   // text_delta
    ldr   x1, [sp, #160]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                     // tdresult2
    ldr   x0, [sp, #160]
    bl    _t_decref
    str   xzr, [sp, #160]
    ldr   x0, [sp, #104]
    cbz   x0, Lfd2_drop152t
    // reasoning = tdresult2[0]; emitted['analysis'] = tdresult2[1]
    ldr   x0, [sp, #104]
    mov   x1, #0
    bl    _tuple_item
    bl    _incref
    str   x0, [sp, #168]                     // reasoning owned
    ldr   x0, [sp, #104]
    mov   x1, #1
    bl    _tuple_item
    bl    _incref
    mov   x2, x0
    mov   x0, #7
    bl    _val                              // 'analysis'
    mov   x1, x0
    ldr   x0, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    ldr   x0, [sp, #144]
    str   xzr, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // ---- closed / return ----
    orr   w26, w23, w27                      // stopflag
    orr   w26, w21, w26                      // closed
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #280]                   // True
    cmp   w26, #0
    b.ne  1f
    ldr   x2, [x10, #288]
1:  mov   x0, x19
    adrp  x1, at_closed@PAGE
    add   x1, x1, at_closed@PAGEOFF
    bl    _t_setattr_s
    cbnz  w0, Lfd2_drop_dec
    // ret tuple = (content, reasoning, terminal or stopped)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lfd2_drop_dec
    str   x0, [sp, #200]
    ldr   x0, [sp, #112]
    bl    _incref
    ldr   x0, [sp, #200]
    mov   x1, #0
    ldr   x2, [sp, #112]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #168]
    bl    _incref
    ldr   x0, [sp, #200]
    mov   x1, #1
    ldr   x2, [sp, #168]
    bl    _t_tuple_setitem
    // stopflag = terminal or stopped
    orr   w26, w23, w27
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #280]                   // True
    cmp   w26, #0
    b.ne  1f
    ldr   x2, [x10, #288]                   // False
1:  ldr   x0, [sp, #200]
    mov   x1, #2
    bl    _t_tuple_setitem
    ldr   x0, [sp, #200]
    b     Lfd2_ret

// ================= feed epilogue =================
// ---- section-A drop: every park owned-or-zero; [184] borrowed (skip) ----
Lfd2_err:
Lfd2_drop_hdrA:
Lfd2_drop_metaA:
Lfd2_drop_kobj:
Lfd2_drop_recv:
Lfd2_drop128:
Lfd2_drop136:
Lfd2_drop152:
Lfd2_drop160:
Lfd2_drop168:
Lfd2_drop176:
Lfd2_drop192:
_fd_dropA:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  ldr   x0, [sp, #136]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #136]
1:  ldr   x0, [sp, #144]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #144]
1:  ldr   x0, [sp, #152]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #152]
1:  ldr   x0, [sp, #160]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #160]
1:  ldr   x0, [sp, #168]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #168]
1:  ldr   x0, [sp, #176]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #176]
1:  ldr   x0, [sp, #192]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #192]
1:  mov   x0, #0
    b     Lfd2_ret0

// ---- raise labels (message C-string in x0 → shared raise+drop) ----
Lfd2_continued:
    adrp  x0, m_continued@PAGE
    add   x0, x0, m_continued@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_start:
    adrp  x0, m_bad_start@PAGE
    add   x0, x0, m_bad_start@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_meta:
    adrp  x0, m_bad_meta@PAGE
    add   x0, x0, m_bad_meta@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_sep:
    adrp  x0, m_bad_sep@PAGE
    add   x0, x0, m_bad_sep@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_end:
    adrp  x0, m_bad_end@PAGE
    add   x0, x0, m_bad_end@PAGEOFF
    b     _fd_raiseA
Lfd2_ret_nobody:
    adrp  x0, m_ret_nobody@PAGE
    add   x0, x0, m_ret_nobody@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_handoff:
Lfd2_drop_recv_h:
    adrp  x0, m_bad_handoff@PAGE
    add   x0, x0, m_bad_handoff@PAGEOFF
    b     _fd_raiseA
Lfd2_bad_control:
    adrp  x0, m_bad_control@PAGE
    add   x0, x0, m_bad_control@PAGEOFF
    b     _fd_raiseA
Lfd2_h_limit:
    adrp  x0, m_h_limit@PAGE
    add   x0, x0, m_h_limit@PAGEOFF
    b     _fd_raiseA
Lfd2_outside:
    adrp  x0, m_outside@PAGE
    add   x0, x0, m_outside@PAGEOFF
    b     _fd_raiseA
_fd_raiseA:
    bl    _raise_value
    b     _fd_dropA

// ---- tail drop: [144]/[184] borrowed, [200] result — skip ----
Lfd2_drop_dec:
Lfd2_drop120t:
Lfd2_drop128t:
Lfd2_drop136t:
Lfd2_drop152t:
_fd_dropT:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  ldr   x0, [sp, #136]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #136]
1:  ldr   x0, [sp, #152]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #152]
1:  ldr   x0, [sp, #160]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #160]
1:  ldr   x0, [sp, #168]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #168]
1:  ldr   x0, [sp, #176]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #176]
1:  ldr   x0, [sp, #192]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #192]
1:  mov   x0, #0
    b     Lfd2_ret0

// ---- exits ----
Lfd2_ret:
    ldr   x0, [sp, #200]
Lfd2_ret0:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #208
    ret

// ============================================================
// _dd_output_stream(x0 = model, x1 = raw_prompt, x2 = stops,
//                   x3 = schema, x4 = logprobs, x5 = ignore_eos,
//                   x6 = tools) -> TextStream | None (py_object)
//   if raw_prompt is not None: return None
//   config = getattr(model,'config',{})
//   if config.get('model_type') != 'gpt_oss': return None
//   if schema is not None or logprobs is not None or ignore_eos:
//       raise ValueError(m_unsupported)
//   return TextStream(model.tokenizer, stops, tools)
// x19 model  x20 raw_prompt  x21 stops  x22 schema  x23 logprobs
// x24 ignore_eos  x25 tools  x26 scratch
// parks: [96] config/tok  [104] key/args  [112] mt/instance
// ============================================================
_dd_output_stream:
    stp   x29, x30, [sp, #-160]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    stp   xzr, xzr, [sp, #128]
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, x3
    mov   x23, x4
    mov   x24, x5
    mov   x25, x6
    // ---- raw_prompt is not None -> None ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x26, [x10, #120]                   // None singleton
    cmp   x20, x26
    b.ne  Los_none
    // ---- config = getattr(model, 'config', {}) ----
    mov   x0, x19
    adrp  x1, at_config@PAGE
    add   x1, x1, at_config@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Los_noconf
    str   x0, [sp, #96]
    b     1f
Los_noconf:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Los_err
    str   x0, [sp, #96]
1:  // ---- mt = config.get('model_type') ----
    adrp  x0, k_model_type@PAGE
    add   x0, x0, k_model_type@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Los_drop_cfg
    str   x0, [sp, #104]
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #104]
    mov   x2, x26                           // None default (borrowed)
    bl    _getdefault
    str   x0, [sp, #112]                     // mt NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Los_drop_cfg
    cmp   x0, x26                           // mt is None -> not gpt_oss -> None
    b.eq  Los_drop_cfg
    adrp  x1, v_gpt_oss@PAGE
    add   x1, x1, v_gpt_oss@PAGEOFF
    bl    _t_strcmp
    mov   x26, x0
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    cbnz  x26, Los_not_oss
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    // ---- schema / logprobs / ignore_eos ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x26, [x10, #120]                   // None
    cmp   x22, x26
    b.ne  Los_unsupported
    cmp   x23, x26
    b.ne  Los_unsupported
    mov   x0, x24
    bl    _t_istrue
    cmp   w0, #1
    b.eq  Los_unsupported
    // ---- tok = model.tokenizer ----
    mov   x0, x19
    adrp  x1, at_tk@PAGE
    add   x1, x1, at_tk@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Los_err
    str   x0, [sp, #96]
    // ---- args = (tok, stops, tools) ----
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Los_drop96
    str   x0, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #1
    mov   x2, x21
    bl    _t_tuple_setitem
    mov   x0, x25
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #2
    mov   x2, x25
    bl    _t_tuple_setitem
    // ---- TextStream(tok, stops, tools) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #336]                   // TextStream class
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                     // instance NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #112]
    cbz   x0, Los_err
    b     Los_ret
// ---- returns ----
Los_none:
Los_not_oss:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]                   // None (borrowed)
    b     Los_ret
Los_unsupported:
    adrp  x0, m_unsupported@PAGE
    add   x0, x0, m_unsupported@PAGEOFF
    bl    _raise_value
    mov   x0, #0
    b     Los_ret
// ---- error drops (exception already set) ----
Los_drop96:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, #0
    b     Los_ret
Los_drop_cfg:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #120]                   // None (borrowed)
    b     Los_ret
Los_err:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  mov   x0, #0
    b     Los_ret
// ---- exit ----
Los_ret:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #160
    ret

// exported _decode trampoline for the bridge's TextStream._decode
_dd_decode:
    b     _decode

// ============================================================
// _dd_markers() -> NEW _MARKERS dict {kind: (id, (name, ...))}
//   built from mk_table / g_kind_py / g_name_py (dd_init order).
// x19 row  x20 i  x21 dict  x22 val  parks [0] idobj [8] names
// ============================================================
_dd_markers:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    bl    _t_dict_new
    cbz   x0, Ldm_out
    mov   x21, x0
    adrp  x19, mk_table@PAGE
    add   x19, x19, mk_table@PAGEOFF
    mov   x20, #0
Ldm1:
    cmp   x20, #8
    b.hs  Ldm_done
    // idobj = PyLong(expected)
    ldr   x0, [x19, #8]
    bl    _t_fromlong
    cbz   x0, Ldm_drop_dict
    str   x0, [sp, #48]
    // names tuple
    ldr   x9, [x19, #24]                    // slot1
    cmp   x9, #0xFF
    b.eq  Ldm_one
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Ldm_drop_id
    str   x0, [sp, #56]
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x0, [x19, #16]
    ldr   x0, [x9, x0, lsl #3]
    bl    _incref
    ldr   x0, [sp, #56]
    mov   x1, #0
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x2, [x19, #16]
    ldr   x2, [x9, x2, lsl #3]
    bl    _t_tuple_setitem
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x0, [x19, #24]
    ldr   x0, [x9, x0, lsl #3]
    bl    _incref
    ldr   x0, [sp, #56]
    mov   x1, #1
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x2, [x19, #24]
    ldr   x2, [x9, x2, lsl #3]
    bl    _t_tuple_setitem
    b     Ldm_val
Ldm_one:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ldm_drop_id
    str   x0, [sp, #56]
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x0, [x19, #16]
    ldr   x0, [x9, x0, lsl #3]
    bl    _incref
    ldr   x0, [sp, #56]
    mov   x1, #0
    adrp  x9, g_name_py@PAGE
    add   x9, x9, g_name_py@PAGEOFF
    ldr   x2, [x19, #16]
    ldr   x2, [x9, x2, lsl #3]
    bl    _t_tuple_setitem
Ldm_val:
    // val = (idobj, names)
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Ldm_drop_names
    mov   x22, x0
    ldr   x0, [sp, #48]
    bl    _incref
    mov   x0, x22
    mov   x1, #0
    ldr   x2, [sp, #48]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #56]
    bl    _incref
    mov   x0, x22
    mov   x1, #1
    ldr   x2, [sp, #56]
    bl    _t_tuple_setitem
    // dict[kind] = val; drop temps
    adrp  x9, g_kind_py@PAGE
    add   x9, x9, g_kind_py@PAGEOFF
    ldr   x1, [x9, x20, lsl #3]
    mov   x0, x21
    mov   x2, x22
    bl    _t_dict_setitem
    ldr   x0, [sp, #48]
    bl    _t_decref
    str   xzr, [sp, #48]
    ldr   x0, [sp, #56]
    bl    _t_decref
    str   xzr, [sp, #56]
    mov   x0, x22
    bl    _t_decref
    add   x19, x19, #32
    add   x20, x20, #1
    b     Ldm1
Ldm_done:
    mov   x0, x21
    b     Ldm_ret
// ---- errors ----
Ldm_drop_names:
    ldr   x0, [sp, #56]
    bl    _t_decref
    str   xzr, [sp, #56]
Ldm_drop_id:
    ldr   x0, [sp, #48]
    bl    _t_decref
    str   xzr, [sp, #48]
Ldm_drop_dict:
    mov   x0, x21
    bl    _t_decref
    mov   x0, #0
    b     Ldm_ret
Ldm_out:
    mov   x0, #0
Ldm_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x29, x30, [sp], #48
    ret
