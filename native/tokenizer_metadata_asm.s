// damendaemon/tokenizer_metadata.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 7/370. 100% of the module: bounded metadata
// readers (Path.open via the with-protocol, read(limit+1), byte-limit
// enforcement, parse_metadata_json dispatch, object validation),
// special_token_text (bos/eos extraction, structured AddedToken content,
// UTF-8 encodability), _select_named_template (1..64 entries, exact
// name/template entry shape, nonempty names, duplicate detection, shared
// UTF-8 byte budget, explicit default, tool_use selection), and
// checkpoint_template (chat_template precedence, chat_template.jinja
// fallback file, byte budget, UTF-8 validation).
//
// Module-global limits (MAX_TOKENIZER_CONFIG_BYTES, MAX_GENERATION_CONFIG_
// BYTES, MAX_CHECKPOINT_TEMPLATE_BYTES) are resolved at CALL TIME via
// GetAttrString on the shim module — the test suite patches them there.
//
// ABI table (offsets in bytes), installed via dd_init:
//    0 PyObject_GetAttrString     8 PyObject_Call
//   16 PyTuple_New               32 PyTuple_SetItem
//   40 Py_DecRef                  48 PyDict_New
//   56 PyDict_SetItem             64 PyDict_GetItemWithError
//   72 PyDict_Contains            88 PyObject_IsTrue
//   96 PyErr_Clear               104 PyErr_SetString
//  112 PyExc_ValueError          120 None
//  144 PyUnicode_FromString     152 PyUnicode_AsUTF8String? (unused; placeholder=PyUnicode_AsEncodedString)
//  136 PyUnicode_AsEncodedString (str,'utf-8',NULL)
//  160 PyUnicode_Concat          168 PyUnicode_Type? no -> 168 = dict type
//  176 list type                 184 str type
//  192 PyObject_Size             200 PyIter_Next
//  208 PyObject_GetIter          216 PySequence_GetItem
//  224 PyLong_AsLongLong         232 PyLong_FromLongLong
//  240 PyBytes_Size              248 parse_metadata_json (py callable)
//  256 Path (py callable)        264 'get' key C-str is separate; slot = shim module obj
//  272 True                      280 PyObject_IsInstance
//  288 'name' key str            296 'template' key str
//  304 'default' key str        312 'tool_use' key str
//  320 'content' key str        328 'chat_template' key str
//  336 'utf-8' str (encode arg)
// Singletons ('name','template',...) are held by the bridge; the asm
// treats them as borrowed (dict lookups only).

.globl _dd_init
.globl _dd_read_object
.globl _dd_read_tokenizer_config
.globl _dd_read_generation_config
.globl _dd_special_token_text
.globl _dd_checkpoint_template

.section __DATA,__data
g_iface:  .quad 0
g_names2: .quad 0          // ('bos_token','eos_token')

.section __TEXT,__cstring
a_open:     .asciz "open"
a_read:     .asciz "read"
a_name:     .asciz "name"
a_enter:    .asciz "__enter__"
a_exit:     .asciz "__exit__"
a_get:      .asciz "get"
a_items:    .asciz "items"
a_decode:   .asciz "decode"
a_exists:   .asciz "exists"
a_truediv:  .asciz "__truediv__"
k_rb:       .asciz "rb"
k_utf8:     .asciz "utf-8"
s_jinja:    .asciz "chat_template.jinja"
n_maxtok:   .asciz "MAX_TOKENIZER_CONFIG_BYTES"
n_maxgen:   .asciz "MAX_GENERATION_CONFIG_BYTES"
n_maxtpl:   .asciz "MAX_CHECKPOINT_TEMPLATE_BYTES"
m_limit1:   .asciz " exceeds metadata byte limit"
m_object:   .asciz " must be an object"
m_content:  .asciz " metadata requires text content"
m_bos_or:   .asciz " must be text or structured token metadata"
m_utf8a:    .asciz " must be valid UTF-8"
m_named_len:.asciz "named chat templates require 1..64 entries"
m_named_ent:.asciz "named chat templates require name/template entries"
m_named_ne: .asciz "named chat templates require nonempty names and text templates"
m_named_dup:.asciz "duplicate named chat template"
m_named_u8: .asciz "named chat templates require valid UTF-8"
m_named_lim:.asciz "named chat templates exceed metadata byte limit"
m_named_def:.asciz "named chat templates require an explicit default"
m_tpl_lim:  .asciz "checkpoint chat template exceeds metadata byte limit"
m_tpl_text: .asciz "model needs one text chat template"
m_tpl_u8:   .asciz "checkpoint chat template must be valid UTF-8"
c_empty:    .asciz ""

.text

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
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
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
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #144]
    br    x9
_t_asencoded:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #136]
    br    x9
_t_concat:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #160]
    br    x9
_t_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #192]
    br    x9
_t_iter_next:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #200]
    br    x9
_t_getiter:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #208]
    br    x9
_t_seq_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #216]
    br    x9
_t_aslong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #224]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #232]
    br    x9
_t_bytes_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #240]
    br    x9
_t_isinstance:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #280]
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

// _raise_value_cat(x0 = prefix str obj BORROWED, x1 = suffix C-string)
//   raises ValueError(f'{prefix}{suffix}')
_raise_value_cat:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    mov   x19, x0                          // prefix (borrowed)
    mov   x0, x1                           // suffix C-string
    bl    _t_fromstring                    // suffix str NEW
    cbz   x0, Lrvc_out
    mov   x20, x0
    mov   x0, x19                          // prefix borrowed
    mov   x1, x20                          // suffix
    bl    _t_concat                        // msg str NEW
    mov   x20, x0                          // msg
    cbz   x20, Lrvc_out
    // cstr = PyUnicode_AsUTF8AndSize(msg, NULL) — borrowed from msg
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #368]
    mov   x0, x20
    mov   x1, #0
    blr   x9
    str   x0, [sp, #32]                    // park cstr
    ldr   x1, [sp, #32]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]                   // PyExc_ValueError
    ldr   x9, [x10, #104]                   // PyErr_SetString (copies)
    blr   x9
    mov   x0, x20
    bl    _t_decref                        // drop msg (cstr was borrowed)
Lrvc_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// helper _incref(x0) — immortal-aware (None/True/False refcount is
// 0xFFFFFFFF in 3.12; a raw +1 would wrap it to 0 and break the world)
_incref:
    ldr   x9, [x0]
    mov   w10, #0xffff
    movk  w10, #0xffff, lsl #16            // w10 = 0xFFFFFFFF
    cmp   x9, x10
    b.hs  9f                               // 0 (bug) or immortal: skip
    add   x9, x9, #1
    str   x9, [x0]
9:  ret

// ============================================================
// void _dd_init(x0 = iface)
//   install table; build g_names2 = ('bos_token','eos_token')
// ============================================================
_dd_init:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lini_out
    mov   x19, x0                          // tuple
    adrp  x9, g_iface@PAGE
    ldr   x20, [x9, g_iface@PAGEOFF]
    ldr   x0, [x20, #352]                  // 'bos_token' str (borrowed)
    bl    _incref
    mov   x0, x19
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #352]
    bl    _t_tuple_setitem                 // steals increfed ref
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #360]                  // 'eos_token'
    bl    _incref
    mov   x0, x19
    mov   x1, #1
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #360]
    bl    _t_tuple_setitem
    adrp  x9, g_names2@PAGE
    str   x19, [x9, g_names2@PAGEOFF]
Lini_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// helper _call0(x0 = callable BORROWED) -> NEW result or NULL
//   PyObject_Call(fn, (), NULL)
_call0:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    str   x19, [sp, #16]
    mov   x19, x0                          // callable
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lc0_out
    str   x0, [sp, #24]                     // park tuple
    mov   x0, x19
    ldr   x1, [sp, #24]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #32]                     // park result
    ldr   x0, [sp, #24]
    bl    _t_decref                         // drop tuple
    ldr   x0, [sp, #32]
Lc0_out:
    ldr   x19, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// PyObject* _dd_read_object(x0 = path BORROWED, x1 = limit BORROWED)
//   path = Path(path); with path.open('rb') as reader:
//       raw = reader.read(limit+1)
//   len(raw) > limit -> ValueError
//   value = raw.decode('utf-8'); parsed = parse_metadata_json(value)
//   not isinstance(parsed, dict) -> ValueError
// Parks: [88]path2 [96]file [104]reader [112]raw [120]strval
//        [128]parsed [136]scratch-a [144]scratch-b [152]scratch-c
//        [160]limit64
// ============================================================
_dd_read_object:
    stp   x29, x30, [sp, #-176]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #88]                // path2, file
    stp   xzr, xzr, [sp, #104]               // reader, raw
    stp   xzr, xzr, [sp, #120]               // strval, parsed
    stp   xzr, xzr, [sp, #136]               // scratch a, b
    stp   xzr, xzr, [sp, #152]               // scratch c, d
    mov   x19, x0                            // path (borrowed)
    // ---- limit64 = PyLong_AsLongLong(limit) ----
    mov   x0, x1                             // limit object
    bl    _t_aslong
    str   x0, [sp, #160]
    // ---- path2 = Path(path) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // args = (path,)
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem                   // fill the slot (steals increfed ref)
    ldr   x1, [sp, #136]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #256]                   // Path
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #88]                      // path2
    ldr   x0, [sp, #136]
    bl    _t_decref                          // drop args
    ldr   x0, [sp, #88]
    cbz   x0, Lro_out
    // ---- openm = path2.open ----
    ldr   x0, [sp, #88]
    adrp  x1, a_open@PAGE
    add   x1, x1, a_open@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // openm
    // ---- args = ('rb',) ----
    adrp  x0, k_rb@PAGE
    add   x0, x0, k_rb@PAGEOFF
    bl    _t_fromstring                      // 'rb' NEW
    cbz   x0, Lro_drop_openm
    str   x0, [sp, #144]                     // rbstr
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lro_drop_rb
    str   x0, [sp, #152]                     // args
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    // ---- file = openm('rb') ----
    ldr   x0, [sp, #136]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #96]                      // file
    ldr   x0, [sp, #152]
    bl    _t_decref
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref                          // openm gone
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    ldr   x0, [sp, #96]
    cbz   x0, Lro_out
    // ---- enterm = file.__enter__ ; reader = enterm() ----
    ldr   x0, [sp, #96]
    adrp  x1, a_enter@PAGE
    add   x1, x1, a_enter@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // enterm
    ldr   x0, [sp, #136]
    bl    _call0
    str   x0, [sp, #104]                     // reader
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #104]
    cbz   x0, Lro_out
    // ---- rdm = reader.read ; raw = rdm(limit+1) ----
    ldr   x0, [sp, #104]
    adrp  x1, a_read@PAGE
    add   x1, x1, a_read@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // rdm
    ldr   x0, [sp, #160]
    add   x0, x0, #1                         // limit64+1
    bl    _t_fromlong
    cbz   x0, Lro_drop_rdm
    str   x0, [sp, #144]                     // n1
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lro_drop_n1
    str   x0, [sp, #152]                     // args
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #136]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                     // raw (may be NULL)
    ldr   x0, [sp, #152]
    bl    _t_decref
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #136]
    // ---- exitm = file.__exit__ ; exitm(None, None, None) ----
    ldr   x0, [sp, #96]
    adrp  x1, a_exit@PAGE
    add   x1, x1, a_exit@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out                        // raw may dangle; acceptable
    str   x0, [sp, #136]                     // exitm
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lro_drop_exitm
    str   x0, [sp, #152]                     // args3
    // None is immortal: pass the borrowed pointer to the stealing
    // SetItem directly — the tuple's eventual decref is a no-op.
    ldr   x0, [sp, #152]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #152]
    mov   x1, #1
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #152]
    mov   x1, #2
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #136]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #144]                     // exitrc
    ldr   x0, [sp, #152]
    bl    _t_decref
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    // ---- if raw is NULL: propagate ----
    ldr   x0, [sp, #112]
    cbz   x0, Lro_out
    // ---- len(raw) > limit? ----
    ldr   x0, [sp, #112]
    bl    _t_size
    cmp   x0, #0
    b.lt  Lro_out                            // size error propagates
    ldr   x9, [sp, #160]
    cmp   x0, x9
    b.le   Lro_len_ok
    // ValueError(f'{path2.name} exceeds metadata byte limit')
    ldr   x0, [sp, #88]
    adrp  x1, a_name@PAGE
    add   x1, x1, a_name@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // nameobj
    adrp  x1, m_limit1@PAGE
    add   x1, x1, m_limit1@PAGEOFF
    bl    _raise_value_cat
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    b     Lro_out
Lro_len_ok:
    // ---- value = raw.decode('utf-8') ----
    ldr   x0, [sp, #112]
    adrp  x1, a_decode@PAGE
    add   x1, x1, a_decode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]                     // decm
    adrp  x0, k_utf8@PAGE
    add   x0, x0, k_utf8@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lro_drop_decm
    str   x0, [sp, #144]                     // u8str
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lro_drop_u8
    str   x0, [sp, #152]
    ldr   x0, [sp, #144]
    bl    _incref
    ldr   x0, [sp, #152]
    mov   x1, #0
    ldr   x2, [sp, #144]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #136]
    ldr   x1, [sp, #152]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]                     // strval
    ldr   x0, [sp, #152]
    bl    _t_decref
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #136]
    ldr   x0, [sp, #120]
    cbz   x0, Lro_out                        // UnicodeDecodeError propagates
    // ---- parsed = parse_metadata_json(strval) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lro_out
    str   x0, [sp, #136]
    ldr   x0, [sp, #120]
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    ldr   x2, [sp, #120]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #248]                   // parse_metadata_json
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // parsed
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    ldr   x0, [sp, #128]
    cbz   x0, Lro_out                        // ValueError propagates
    // ---- isinstance(parsed, dict)? ----
    ldr   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #168]                   // dict type
    bl    _t_isinstance
    cmp   w0, #1
    b.eq  Lro_return
    // ValueError(f'{path2.name} must be an object')
    ldr   x0, [sp, #88]
    adrp  x1, a_name@PAGE
    add   x1, x1, a_name@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lro_out
    str   x0, [sp, #136]
    adrp  x1, m_object@PAGE
    add   x1, x1, m_object@PAGEOFF
    bl    _raise_value_cat
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lro_out
Lro_return:
    // transfer parsed to the result park; [128] cleared so cleanup
    // does not drop the object we are returning
    ldr   x0, [sp, #128]
    str   xzr, [sp, #128]
    b     Lro_out
Lro_drop_decm:
Lro_drop_u8:
Lro_drop_n1:
Lro_drop_rdm:
Lro_drop_exitm:
Lro_drop_rb:
Lro_drop_openm:
    // all parks are dropped by Lro_out in order; nothing extra here
    b     Lro_out
Lro_out:
    // park the result first: the decrefs below clobber x0
    str   x0, [sp, #168]
    // cleanup: drop every nonzero park
    ldr   x0, [sp, #136]
    cbz   x0, 1f
    bl    _t_decref
1:  ldr   x0, [sp, #144]
    cbz   x0, 2f
    bl    _t_decref
2:  ldr   x0, [sp, #152]
    cbz   x0, 3f
    bl    _t_decref
3:  ldr   x0, [sp, #128]
    cbz   x0, 4f
    bl    _t_decref
4:  ldr   x0, [sp, #120]
    cbz   x0, 5f
    bl    _t_decref
5:  ldr   x0, [sp, #112]
    cbz   x0, 6f
    bl    _t_decref
6:  ldr   x0, [sp, #104]
    cbz   x0, 7f
    bl    _t_decref
7:  ldr   x0, [sp, #96]
    cbz   x0, 8f
    bl    _t_decref
8:  ldr   x0, [sp, #88]
    cbz   x0, 9f
    bl    _t_decref
9:  ldr   x0, [sp, #168]                     // result survives cleanup
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #176
    ret

// ============================================================
// PyObject* _dd_read_tokenizer_config(x0 = path BORROWED)
//   limit resolved at call time from the shim module global.
// ============================================================
_dd_read_tokenizer_config:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    str   x19, [sp, #16]
    mov   x19, x0                            // path (borrowed)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #264]                   // shim module (borrowed)
    adrp  x1, n_maxtok@PAGE
    add   x1, x1, n_maxtok@PAGEOFF
    bl    _t_getattr_s                       // limit NEW
    cbz   x0, Ltk_out
    str   x0, [sp, #24]
    mov   x0, x19
    ldr   x1, [sp, #24]
    bl    _dd_read_object
    str   x0, [sp, #32]
    ldr   x0, [sp, #24]
    bl    _t_decref
    ldr   x0, [sp, #32]
Ltk_out:
    ldr   x19, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// PyObject* _dd_read_generation_config(x0 = path BORROWED)
// ============================================================
_dd_read_generation_config:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    str   x19, [sp, #16]
    mov   x19, x0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #264]
    adrp  x1, n_maxgen@PAGE
    add   x1, x1, n_maxgen@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Ltk_out
    str   x0, [sp, #24]
    mov   x0, x19
    ldr   x1, [sp, #24]
    bl    _dd_read_object
    str   x0, [sp, #32]
    ldr   x0, [sp, #24]
    bl    _t_decref
    ldr   x0, [sp, #32]
    ldr   x19, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret

// ============================================================
// PyObject* _dd_special_token_text(x0 = metadata BORROWED) -> dict
// Parks: [96]name(borrowed) [104]value [112]scratch [120]scratch
//        [128]value2
// ============================================================
_dd_special_token_text:
    stp   x29, x30, [sp, #-144]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]                // name, value
    stp   xzr, xzr, [sp, #112]               // scratch, scratch
    str   xzr, [sp, #128]                    // value2
    mov   x19, x0                            // metadata
    bl    _t_dict_new
    cbz   x0, Lst_ret0
    mov   x20, x0                            // result
    adrp  x9, g_names2@PAGE
    ldr   x0, [x9, g_names2@PAGEOFF]
    bl    _t_getiter
    cbz   x0, Lst_drop_result
    mov   x21, x0                            // iter
Lst_loop:
    mov   x0, x21
    bl    _t_iter_next                        // name BORROWED
    cbz   x0, Lst_done
    str   x0, [sp, #96]
    // ---- value = metadata.get(name) ----
    mov   x0, x19
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lst_err
    str   x0, [sp, #112]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lst_drop_getm
    str   x0, [sp, #120]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #120]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #120]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                     // value NEW
    ldr   x0, [sp, #120]
    bl    _t_decref
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #120]
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    cbz   x0, Lst_err
    // ---- isinstance(value, dict)? ----
    ldr   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #168]
    bl    _t_isinstance
    cmp   w0, #0
    b.lt  Lst_err
    cmp   w0, #1
    b.ne  Lst_not_dict
    // ---- value = value.get('content') ----
    ldr   x0, [sp, #104]
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lst_err
    str   x0, [sp, #112]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lst_drop_getm
    str   x0, [sp, #120]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #320]                   // 'content' str borrowed
    bl    _incref
    ldr   x0, [sp, #120]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #320]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #120]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // value2
    ldr   x0, [sp, #120]
    bl    _t_decref
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #120]
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref                           // drop old value
    ldr   x0, [sp, #128]
    str   x0, [sp, #104]
    str   xzr, [sp, #128]
    cbz   x0, Lst_err
    // if not isinstance(value2, str): raise
    ldr   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #184]                   // str type
    bl    _t_isinstance
    cmp   w0, #1
    b.eq  Lst_enc_check
    ldr   x0, [sp, #96]
    adrp  x1, m_content@PAGE
    add   x1, x1, m_content@PAGEOFF
    bl    _raise_value_cat
    b     Lst_err
Lst_not_dict:
    // ---- elif value is not None and not isinstance(value, str) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #120]                   // None (fresh load)
    ldr   x22, [sp, #104]
    cmp   x22, x9
    b.eq  Lst_enc_check
    ldr   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #184]
    bl    _t_isinstance
    cmp   w0, #1
    b.eq  Lst_enc_check
    ldr   x0, [sp, #96]
    adrp  x1, m_bos_or@PAGE
    add   x1, x1, m_bos_or@PAGEOFF
    bl    _raise_value_cat
    b     Lst_err
Lst_enc_check:
    // ---- if value is not None: encodable? ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #120]                   // None
    ldr   x22, [sp, #104]
    cmp   x22, x9
    b.eq  Lst_setitem
    ldr   x0, [sp, #104]
    adrp  x1, k_utf8@PAGE
    add   x1, x1, k_utf8@PAGEOFF
    mov   x2, #0
    bl    _t_asencoded
    cbz   x0, Lst_enc_bad
    bl    _t_decref
    b     Lst_setitem
Lst_enc_bad:
    bl    _t_err_clear
    ldr   x0, [sp, #96]
    adrp  x1, m_utf8a@PAGE
    add   x1, x1, m_utf8a@PAGEOFF
    bl    _raise_value_cat
    b     Lst_err
Lst_setitem:
    mov   x0, x20
    ldr   x1, [sp, #96]
    ldr   x2, [sp, #104]
    bl    _t_dict_setitem
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    b     Lst_loop
Lst_done:
    mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    b     Lst_out
Lst_drop_getm:
    ldr   x0, [sp, #112]
    bl    _t_decref
    b     Lst_err
Lst_drop_result:
    mov   x0, x20
    bl    _t_decref
    b     Lst_ret0
Lst_err:
    // drop parks that may be live, then iter + result
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
1:  ldr   x0, [sp, #120]
    cbz   x0, 2f
    bl    _t_decref
2:  ldr   x0, [sp, #112]
    cbz   x0, 3f
    bl    _t_decref
3:  ldr   x0, [sp, #104]
    cbz   x0, 4f
    bl    _t_decref
4:  mov   x0, x21
    bl    _t_decref
    mov   x0, x20
    bl    _t_decref
Lst_ret0:
    mov   x0, #0
Lst_out:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #144
    ret

// ============================================================
// _select_named(x0 = collection BORROWED, x1 = tools BORROWED,
//               x2 = limit64) -> str NEW or NULL
// Parks: [96]it0 [104]view [112]enc/scratch [120]scratch [128]scratch
// Regs: x19 collection x20 templates x21 size x22 iter x25 limit64
//       x26 name x27 template x28 tools
// ============================================================
_select_named:
    stp   x29, x30, [sp, #-160]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    str   xzr, [sp, #128]
    mov   x19, x0
    mov   x28, x1                            // tools (borrowed)
    mov   x25, x2                            // limit64
    mov   x20, #0                            // templates: not yet created
    mov   x22, #0
    mov   x24, #0
    mov   x26, #0
    mov   x27, #0
    // ---- 1 <= len(collection) <= 64 ----
    mov   x0, x19
    bl    _t_size
    cmp   x0, #1
    b.lt  Lsn_len_bad
    cmp   x0, #64
    b.gt  Lsn_len_bad
    // ---- is_dict = isinstance(collection, dict) ----
    mov   x0, x19
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #168]
    bl    _t_isinstance
    cmp   w0, #0
    b.lt  Lsn_ret0
    str   w0, [sp, #128]                     // is_dict flag (byte range)
    cbnz  w0, Lsn_dict_path
    // ---- list path: validate every entry first ----
    mov   x0, x19
    bl    _t_getiter
    cbz   x0, Lsn_ret0
    str   x0, [sp, #96]                      // it0
Lsn_v_loop:
    ldr   x0, [sp, #96]
    bl    _t_iter_next                       // e NEW (PyIter_Next owns)
    cbz   x0, Lsn_v_done
    mov   x24, x0
    mov   x0, x24
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #168]
    bl    _t_isinstance
    cmp   w0, #1
    b.ne  Lsn_v_bad
    mov   x0, x24
    bl    _t_size
    cmp   x0, #2
    b.ne  Lsn_v_bad
    mov   x0, x24
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #288]                   // 'name'
    bl    _t_dict_contains
    cmp   w0, #1
    b.ne  Lsn_v_bad
    mov   x0, x24
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #296]                   // 'template'
    bl    _t_dict_contains
    cmp   w0, #1
    b.ne  Lsn_v_bad
    mov   x0, x24
    bl    _t_decref                          // drop validation item
    mov   x24, #0
    b     Lsn_v_loop
Lsn_v_bad:
    adrp  x0, m_named_ent@PAGE
    add   x0, x0, m_named_ent@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_v_done:
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x19
    bl    _t_getiter
    cbz   x0, Lsn_ret0
    mov   x22, x0                            // iter
    b     Lsn_have_iter
Lsn_dict_path:
    mov   x0, x19
    adrp  x1, a_items@PAGE
    add   x1, x1, a_items@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lsn_ret0
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    bl    _call0                             // view
    ldr   x9, [sp, #112]
    mov   x1, x0
    mov   x0, x9
    str   x1, [sp, #104]                     // park view
    bl    _t_decref                          // drop items method
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    cbz   x0, Lsn_ret0
    bl    _t_getiter
    str   x0, [sp, #112]                     // park iter
    ldr   x0, [sp, #104]
    bl    _t_decref                          // drop view
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Lsn_ret0
    mov   x22, x0
    str   xzr, [sp, #112]
Lsn_have_iter:
    // ---- templates = {} ; size = 0 ----
    bl    _t_dict_new
    cbz   x0, Lsn_drop_iter
    mov   x20, x0
    mov   x21, #0
Lsn_loop:
    mov   x0, x22
    bl    _t_iter_next                       // item NEW (PyIter_Next owns)
    cbz   x0, Lsn_end_loop
    mov   x24, x0
    mov   x26, #0
    mov   x27, #0
    ldr   w9, [sp, #128]                     // is_dict flag
    cbnz  w9, Lsn_unpack_dict
    // list: name/template from entry dict (borrowed -> incref)
    mov   x0, x24
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #288]
    bl    _t_dict_getwe
    cbz   x0, Lsn_named_ne
    mov   x26, x0
    bl    _incref
    mov   x0, x24
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #296]
    bl    _t_dict_getwe
    cbz   x0, Lsn_named_ne
    mov   x27, x0
    bl    _incref
    b     Lsn_have_nt
Lsn_unpack_dict:
    mov   x0, x24
    mov   x1, #0
    bl    _t_seq_getitem
    cbz   x0, Lsn_ret0
    mov   x26, x0
    mov   x0, x24
    mov   x1, #1
    bl    _t_seq_getitem
    cbz   x0, Lsn_drop_name
    mov   x27, x0
Lsn_have_nt:
    // ---- checks: name str, truthy; template str ----
    mov   x0, x26
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #184]
    bl    _t_isinstance
    cmp   w0, #1
    b.ne  Lsn_named_ne
    mov   x0, x26
    bl    _t_istrue
    cmp   w0, #1
    b.ne  Lsn_named_ne
    mov   x0, x27
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #184]
    bl    _t_isinstance
    cmp   w0, #1
    b.ne  Lsn_named_ne
    // ---- duplicate? ----
    mov   x0, x20
    mov   x1, x26
    bl    _t_dict_contains
    cmp   w0, #1
    b.eq  Lsn_named_dup
    // ---- size += len(template.encode()); name.encode() ----
    mov   x0, x27
    adrp  x1, k_utf8@PAGE
    add   x1, x1, k_utf8@PAGEOFF
    mov   x2, #0
    bl    _t_asencoded
    cbz   x0, Lsn_named_u8
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    bl    _t_bytes_size
    add   x21, x21, x0
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    mov   x0, x26
    adrp  x1, k_utf8@PAGE
    add   x1, x1, k_utf8@PAGEOFF
    mov   x2, #0
    bl    _t_asencoded
    cbz   x0, Lsn_named_u8
    bl    _t_decref
    // ---- if size > limit: raise ----
    cmp   x21, x25
    b.gt  Lsn_named_lim
    // ---- templates[name] = template ----
    mov   x0, x20
    mov   x1, x26
    mov   x2, x27
    bl    _t_dict_setitem
    mov   x0, x26
    bl    _t_decref
    mov   x0, x27
    bl    _t_decref
    mov   x26, #0
    mov   x27, #0
    mov   x0, x24
    bl    _t_decref                          // PyIter_Next gave us the item
    mov   x24, #0
    b     Lsn_loop
Lsn_end_loop:
    // ---- 'default' present? ----
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #304]
    bl    _t_dict_contains
    cmp   w0, #1
    b.ne  Lsn_named_def
    // ---- select default / tool_use ----
    mov   x0, x28
    bl    _t_istrue
    cmp   w0, #1
    b.ne  Lsn_pick_default
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #312]                   // 'tool_use'
    bl    _t_dict_getwe
    cbnz  x0, Lsn_pick_done
    bl    _t_err_clear
Lsn_pick_default:
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #304]                   // 'default'
    bl    _t_dict_getwe
    cbz   x0, Lsn_named_def
Lsn_pick_done:
    mov   x23, x0                            // borrowed selection
    mov   x0, x23
    bl    _incref                            // now owned by us
    mov   x0, x22
    bl    _t_decref                          // drop iter
    mov   x0, x20
    bl    _t_decref                          // drop templates
    mov   x0, x23                            // return the owned ref
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #160
    ret
Lsn_drop_name:
    mov   x0, x26
    bl    _t_decref
    b     Lsn_ret0
Lsn_len_bad:
    adrp  x0, m_named_len@PAGE
    add   x0, x0, m_named_len@PAGEOFF
    bl    _raise_value
    b     Lsn_ret0
Lsn_named_ne:
    adrp  x0, m_named_ne@PAGE
    add   x0, x0, m_named_ne@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_named_dup:
    adrp  x0, m_named_dup@PAGE
    add   x0, x0, m_named_dup@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_named_u8:
    bl    _t_err_clear
    adrp  x0, m_named_u8@PAGE
    add   x0, x0, m_named_u8@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_named_lim:
    adrp  x0, m_named_lim@PAGE
    add   x0, x0, m_named_lim@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_named_def:
    adrp  x0, m_named_def@PAGE
    add   x0, x0, m_named_def@PAGEOFF
    bl    _raise_value
    b     Lsn_err
Lsn_drop_iter:
    mov   x0, x22
    cbz   x0, Lsn_ret0
    bl    _t_decref
Lsn_ret0:
    mov   x0, #0
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #160
    ret
Lsn_err:
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
3:  cmp   x26, #0
    b.eq  4f
    mov   x0, x26
    bl    _t_decref
    mov   x26, #0
4:  cmp   x27, #0
    b.eq  5f
    mov   x0, x27
    bl    _t_decref
    mov   x27, #0
5:  cmp   x24, #0
    b.eq  6f
    mov   x0, x24
    bl    _t_decref
    mov   x24, #0
6:  mov   x0, x22
    cbz   x0, 7f
    bl    _t_decref
7:  mov   x0, x20
    cbz   x0, 8f
    bl    _t_decref
8:  b     Lsn_ret0

// ============================================================
// PyObject* _dd_checkpoint_template(x0 = path BORROWED,
//         x1 = metadata BORROWED, x2 = tools BORROWED bool)
//   -> str / None NEW or NULL
// Regs: x19 path x20 limit64 x21 metadata x22 template x28 tools
// Parks: [96]path2 [104]fallback [112]file/raw [120]reader
//        [128]scratch [136]scratch [144]scratch [152]scratch
// ============================================================
_dd_checkpoint_template:
    stp   x29, x30, [sp, #-176]!
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
    mov   x19, x0                            // path
    mov   x21, x1                            // metadata
    mov   x28, x2                            // tools
    mov   x22, #0                            // template: none yet
    // ---- limit64 = MAX_CHECKPOINT_TEMPLATE_BYTES (module attr) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #264]                   // shim module
    adrp  x1, n_maxtpl@PAGE
    add   x1, x1, n_maxtpl@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    ldr   x0, [sp, #128]
    bl    _t_aslong
    mov   x20, x0                            // limit64
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    // ---- template = metadata.get('chat_template') ----
    mov   x0, x21
    adrp  x1, a_get@PAGE
    add   x1, x1, a_get@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_getm
    str   x0, [sp, #136]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #328]                   // 'chat_template' borrowed
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #328]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #136]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #144]                     // template NEW
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    ldr   x0, [sp, #144]
    cbz   x0, Lcp_ret0
    // ---- isinstance(template, (dict, list))? -> _select_named ----
    ldr   x22, [sp, #144]
    str   xzr, [sp, #144]                     // x22 now owns it; clear the park
    mov   x0, x22
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #168]                   // dict
    bl    _t_isinstance
    cmp   w0, #1
    b.eq  Lcp_named
    mov   x0, x22
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #176]                   // list
    bl    _t_isinstance
    cmp   w0, #1
    b.ne  Lcp_after_named
Lcp_named:
    mov   x0, x22
    mov   x1, x28
    mov   x2, x20
    bl    _select_named
    str   x22, [sp, #128]                    // park old (still owned)
    mov   x22, x0
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    cbz   x22, Lcp_ret0
Lcp_after_named:
    // ---- if template is None: chat_template.jinja fallback ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #120]                   // None
    cmp   x22, x9
    b.ne  Lcp_final_checks
    // path2 = Path(path)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #128]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #256]                   // Path
    ldr   x1, [sp, #128]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #96]                      // path2
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #96]
    cbz   x0, Lcp_ret0
    // fbm = path2.__truediv__ ; fallback = fbm('chat_template.jinja')
    ldr   x0, [sp, #96]
    adrp  x1, a_truediv@PAGE
    add   x1, x1, a_truediv@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    adrp  x0, s_jinja@PAGE
    add   x0, x0, s_jinja@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcp_drop_fbm
    str   x0, [sp, #136]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_jstr
    str   x0, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #144]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                     // fallback
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    ldr   x0, [sp, #104]
    cbz   x0, Lcp_ret0
    // exm = fallback.exists ; ex = exm()
    ldr   x0, [sp, #104]
    adrp  x1, a_exists@PAGE
    add   x1, x1, a_exists@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    ldr   x0, [sp, #128]
    bl    _call0
    str   x0, [sp, #136]                     // ex
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #136]
    bl    _t_istrue
    ldr   x9, [sp, #136]
    mov   x23, x0                            // truth rc
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    cmp   x23, #1
    b.ne  Lcp_drop_fallback                  // not exists -> template stays None
    // ---- with fallback.open('rb') as reader: raw = reader.read(M+1) ----
    ldr   x0, [sp, #104]
    adrp  x1, a_open@PAGE
    add   x1, x1, a_open@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]                     // openm
    adrp  x0, k_rb@PAGE
    add   x0, x0, k_rb@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcp_drop_openm
    str   x0, [sp, #136]                     // rbstr
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_rbstr
    str   x0, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #144]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #152]                     // file
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    ldr   x0, [sp, #152]
    cbz   x0, Lcp_ret0
    // reader = file.__enter__()
    ldr   x0, [sp, #152]
    adrp  x1, a_enter@PAGE
    add   x1, x1, a_enter@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    ldr   x0, [sp, #128]
    bl    _call0
    str   x0, [sp, #120]                     // reader
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #120]
    cbz   x0, Lcp_ret0
    // raw = reader.read(limit+1)
    ldr   x0, [sp, #120]
    adrp  x1, a_read@PAGE
    add   x1, x1, a_read@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]                     // rdm
    add   x0, x20, #1
    bl    _t_fromlong
    cbz   x0, Lcp_drop_rdm
    str   x0, [sp, #136]                     // n1
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_n1
    str   x0, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #144]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                     // raw (replaces file park)
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    ldr   x0, [sp, #112]
    cbz   x0, Lcp_ret0
    // ---- exit = file.__exit__ ; exit(None,None,None) ----
    ldr   x0, [sp, #152]
    adrp  x1, a_exit@PAGE
    add   x1, x1, a_exit@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_drop_file
    str   x0, [sp, #128]
    // build (None, None, None) and call
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_exitm
    str   x0, [sp, #144]
    ldr   x0, [sp, #144]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    mov   x1, #1
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #144]
    mov   x1, #2
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #120]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #136]
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    // drop the file (closed by __exit__ above)
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
Lcp_decode:
    // ---- len(raw) > limit? ----
    ldr   x0, [sp, #112]
    bl    _t_size
    cmp   x0, #0
    b.lt  Lcp_ret0
    cmp   x0, x20
    b.le  1f
    adrp  x0, m_tpl_lim@PAGE
    add   x0, x0, m_tpl_lim@PAGEOFF
    bl    _raise_value
    b     Lcp_ret0
1:  // ---- template = raw.decode('utf-8') ----
    ldr   x0, [sp, #112]
    adrp  x1, a_decode@PAGE
    add   x1, x1, a_decode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lcp_ret0
    str   x0, [sp, #128]
    adrp  x0, k_utf8@PAGE
    add   x0, x0, k_utf8@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcp_drop_decm
    str   x0, [sp, #136]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcp_drop_u8s
    str   x0, [sp, #144]
    ldr   x0, [sp, #136]
    bl    _incref
    ldr   x0, [sp, #144]
    mov   x1, #0
    ldr   x2, [sp, #136]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #144]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #152]                     // decoded template
    ldr   x0, [sp, #144]
    bl    _t_decref
    ldr   x0, [sp, #136]
    bl    _t_decref
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #144]
    str   xzr, [sp, #136]
    str   xzr, [sp, #128]
    // replace template: drop old None (immortal-safe via Py_DecRef)
    ldr   x0, [sp, #152]
    cbz   x0, Lcp_ret0
    // x22 currently holds the None singleton (borrowed, not owned) —
    // the NEW ref from .get was a new ref to None; dropping it is a
    // no-op. Transfer decoded into x22; drop raw parks.
    ldr   x22, [sp, #152]
    str   xzr, [sp, #152]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lcp_final_checks
Lcp_drop_fallback:
    // exists() returned false: drop fallback, keep template=None
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    b     Lcp_final_checks
Lcp_final_checks:
    // ---- if template is not None ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #120]                   // None
    cmp   x22, x9
    b.eq  Lcp_return
    // isinstance(template, str)?
    mov   x0, x22
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #184]
    bl    _t_isinstance
    cmp   w0, #1
    b.eq  1f
    adrp  x0, m_tpl_text@PAGE
    add   x0, x0, m_tpl_text@PAGEOFF
    bl    _raise_value
    b     Lcp_ret0
1:  // encodable + size budget
    mov   x0, x22
    adrp  x1, k_utf8@PAGE
    add   x1, x1, k_utf8@PAGEOFF
    mov   x2, #0
    bl    _t_asencoded
    cbz   x0, Lcp_tpl_u8
    mov   x27, x0                           // enc (callee-saved park)
    mov   x0, x27
    bl    _t_bytes_size
    mov   x26, x0                           // size
    mov   x0, x27
    bl    _t_decref
    cmp   x26, x20
    b.le  2f
    adrp  x0, m_tpl_lim@PAGE
    add   x0, x0, m_tpl_lim@PAGEOFF
    bl    _raise_value
    b     Lcp_ret0
2:  b     Lcp_return
Lcp_tpl_u8:
    bl    _t_err_clear
    adrp  x0, m_tpl_u8@PAGE
    add   x0, x0, m_tpl_u8@PAGEOFF
    bl    _raise_value
    b     Lcp_ret0
Lcp_return:
    mov   x23, x22                           // result (owned)
    b     Lcp_cleanup
Lcp_drop_file:
    ldr   x0, [sp, #152]
    bl    _t_decref
    str   xzr, [sp, #152]
    b     Lcp_ret0
Lcp_drop_exitm:
    ldr   x0, [sp, #128]
    bl    _t_decref
    b     Lcp_drop_file
Lcp_drop_decm:
    ldr   x0, [sp, #128]
    bl    _t_decref
    b     Lcp_ret0
Lcp_drop_u8s:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lcp_drop_decm
Lcp_drop_n1:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lcp_drop_rdm
Lcp_drop_rdm:
    ldr   x0, [sp, #128]
    bl    _t_decref
    b     Lcp_ret0
Lcp_drop_rbstr:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lcp_drop_openm
Lcp_drop_openm:
    ldr   x0, [sp, #128]
    bl    _t_decref
    b     Lcp_ret0
Lcp_drop_jstr:
    ldr   x0, [sp, #136]
    bl    _t_decref
    b     Lcp_drop_fbm
Lcp_drop_fbm:
    ldr   x0, [sp, #128]
    bl    _t_decref
    b     Lcp_ret0
Lcp_drop_getm:
    ldr   x0, [sp, #128]
    bl    _t_decref
Lcp_ret0:
    // drop the template ref (0, immortal None, or a real str)
    cmp   x22, #0
    b.eq  11f
    mov   x0, x22
    bl    _t_decref
11: mov   x23, #0                            // result = NULL
Lcp_cleanup:
    // drop parks; x23 (callee-saved) carries the result out
    ldr   x0, [sp, #152]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #152]
1:  ldr   x0, [sp, #144]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #144]
2:  ldr   x0, [sp, #136]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #136]
3:  ldr   x0, [sp, #128]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #128]
4:  ldr   x0, [sp, #120]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #120]
5:  ldr   x0, [sp, #112]
    cbz   x0, 6f
    bl    _t_decref
    str   xzr, [sp, #112]
6:  ldr   x0, [sp, #104]
    cbz   x0, 7f
    bl    _t_decref
    str   xzr, [sp, #104]
7:  ldr   x0, [sp, #96]
    cbz   x0, 8f
    bl    _t_decref
    str   xzr, [sp, #96]
8:  mov   x0, x23                            // result
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x27, x28, [sp, #80]
    ldp   x29, x30, [sp], #176
    ret
