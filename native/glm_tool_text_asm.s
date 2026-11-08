// damendaemon/glm_tool_text.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 39/370. 100% of the module:
//     class GlmToolText(GlmText):
//         def tool_chat_tokens(self,messages,tools,**options):
//             prompt=render_tool_chat(messages,tools,**options)
//             tokens=self.tokenizer.encode(prompt,add_special_tokens=False).ids
//             if not tokens or len(tokens)>MAX_TOKENS:
//                 raise ValueError('GLM tool prompt token count exceeds bounds')
//             return tokens
// Fidelity notes:
//   - class is docless; ns gets __module__/__qualname__/tool_chat_tokens;
//     __init__ inherited from GlmText (real class machinery).
//   - render_tool_chat / MAX_TOKENS are bridge-held snapshots — identical
//     to the original module's own `from .glm_text import ...` binding.
//   - encode called with kwargs {'add_special_tokens': False}.
//   - gate order: `not tokens` (IsTrue) first, then len()>MAX_TOKENS
//     (PyObject_Size -> FromSsize_t -> RichCompareBool GT) — CPython order.
//   - raise path drops locals BEFORE PyErr_SetString (war story 41/46).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear       104 type            112 GlmText
//   120 MAX_TOKENS        128 render_tool_chat  136 ValueError
//   144 False             152 _tool_chat_tokens fn  160 PyObject_IsTrue
//   168 RichCompareBool   176 PyLong_FromSsize_t  184 TypeError

.globl _dd_init
.globl _dd_classes
.globl _dd_tool_chat_tokens

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
s_prepare:  .asciz "__prepare__"
s_conflict: .asciz "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases"
k_module:   .asciz "__module__"
k_qualname: .asciz "__qualname__"
k_method:   .asciz "tool_chat_tokens"
k_tokenizer: .asciz "tokenizer"
k_encode:   .asciz "encode"
k_ids:      .asciz "ids"
k_addspec:  .asciz "add_special_tokens"
m_bounds:   .asciz "GLM tool prompt token count exceeds bounds"
cn_0:       .asciz "GlmToolText"

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
_t_dict_new:        IFACE #40
_t_dict_setitem:   IFACE #48
_t_fromstring:     IFACE #56
_t_issub:          IFACE #64
_t_size:           IFACE #72
_t_seq_getitem:    IFACE #80
_t_seterr:         IFACE #88
_t_err_clear:      IFACE #96
_t_istrue:         IFACE #160
_t_rcb:            IFACE #168
_t_fromssize:      IFACE #176

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
    ldr   x0, [x10, #184]                   // TypeError
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
// PyObject* _dd_tool_chat_tokens(x0=self, x1=messages, x2=tools,
//                                 x3=options dict)  — all BORROWED
// x25 self  x26 options  x27 messages  x28 tools     (borrowed)
// x19 prompt NEW   x20 tokenizer NEW  x21 encode_m NEW
// x22 encode-result NEW (freed after .ids)  x23 ids NEW
// x24 size-long NEW (transient)
// parks: [96] render args  [104] encode kwargs  [112] encode args
//       [120] result park
// ============================================================
_dd_tool_chat_tokens:
    stp   x29, x30, [sp, #-160]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    mov   x25, x0
    mov   x26, x3
    mov   x27, x1
    mov   x28, x2
    mov   x19, xzr
    mov   x20, xzr
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr

    // ---- args = (messages, tools) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #96]
    mov   x0, x27
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x27
    bl    _t_tuple_setitem
    mov   x0, x28
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #1
    mov   x2, x28
    bl    _t_tuple_setitem

    // ---- prompt = render_tool_chat(messages, tools, **options) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // render_tool_chat
    ldr   x1, [sp, #96]
    mov   x2, x26                           // options borrowed
    bl    _t_call
    str   x0, [sp, #120]                    // park result across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x19, [sp, #120]                   // prompt NEW
    str   xzr, [sp, #120]
    cbz   x19, Lerr

    // ---- tok = self.tokenizer; enc = tok.encode ----
    mov   x0, x25
    adrp  x1, k_tokenizer@PAGE
    add   x1, x1, k_tokenizer@PAGEOFF
    bl    _t_getattr_s
    mov   x20, x0                           // tokenizer NEW
    cbz   x20, Lerr
    mov   x0, x20
    adrp  x1, k_encode@PAGE
    add   x1, x1, k_encode@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                           // encode method NEW
    cbz   x21, Lerr

    // ---- kwargs = {'add_special_tokens': False} ----
    bl    _t_dict_new
    cbz   x0, Lerr
    str   x0, [sp, #104]
    adrp  x0, k_addspec@PAGE
    add   x0, x0, k_addspec@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lerr
    mov   x24, x0                           // key NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #104]
    mov   x1, x24
    ldr   x2, [x10, #144]                   // False (immortal)
    bl    _t_dict_setitem
    cmn   w0, #1                            // -1 = error (int return!)
    b.eq  Lerr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr

    // ---- args = (prompt,) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lerr
    str   x0, [sp, #112]
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x19
    bl    _t_tuple_setitem

    // ---- enc_result = enc(prompt, add_special_tokens=False) ----
    mov   x0, x21
    ldr   x1, [sp, #112]
    ldr   x2, [sp, #104]
    bl    _t_call
    str   x0, [sp, #120]                    // park result across cleanup
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x22, [sp, #120]                   // encode result NEW
    str   xzr, [sp, #120]
    cbz   x22, Lerr

    // ---- tokens = enc_result.ids; drop the unnamed temp ----
    mov   x0, x22
    adrp  x1, k_ids@PAGE
    add   x1, x1, k_ids@PAGEOFF
    bl    _t_getattr_s
    str   x0, [sp, #120]                    // park ids across decref
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    ldr   x23, [sp, #120]                   // ids NEW
    str   xzr, [sp, #120]
    cbz   x23, Lerr

    // ---- gate 1: not tokens ----
    mov   x0, x23
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lerr
    cbz   w0, Lraise

    // ---- gate 2: len(tokens) > MAX_TOKENS ----
    mov   x0, x23
    bl    _t_size
    cmn   x0, #1                            // -1 = size error
    b.eq  Lerr
    bl    _t_fromssize
    cbz   x0, Lerr
    mov   x24, x0                           // size long NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    mov   x0, x24
    ldr   x1, [x10, #120]                   // MAX_TOKENS
    mov   x2, #4                            // Py_GT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lerr_drop24
    cmp   w0, #1
    b.eq  Lraise
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr

    // ---- success: return tokens (NEW ref) ----
    mov   x0, x23
    bl    _incref
    str   x0, [sp, #120]
    mov   x23, xzr                          // ownership -> park
    b     Lsucc

Lraise:
    cbz   x24, Lra1
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
Lra1:
    cbz   x23, Lra2
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Lra2:
    cbz   x21, Lra3
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lra3:
    cbz   x20, Lra4
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Lra4:
    cbz   x19, Lra5
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
Lra5:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // ValueError
    adrp  x1, m_bounds@PAGE
    add   x1, x1, m_bounds@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lout

Lerr_drop24:
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    b     Lerr
Lerr:
    ldr   x0, [sp, #96]
    cbz   x0, Le1
    bl    _t_decref
    str   xzr, [sp, #96]
Le1:
    ldr   x0, [sp, #104]
    cbz   x0, Le2
    bl    _t_decref
    str   xzr, [sp, #104]
Le2:
    ldr   x0, [sp, #112]
    cbz   x0, Le3
    bl    _t_decref
    str   xzr, [sp, #112]
Le3:
    cbz   x24, Le4
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
Le4:
    cbz   x23, Le5
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Le5:
    cbz   x22, Le6
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Le6:
    cbz   x21, Le7
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Le7:
    cbz   x20, Le8
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Le8:
    cbz   x19, Le9
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
Le9:
    mov   x0, xzr                           // NULL + pending exception
    b     Lout

Lsucc:                                      // locals still owed; result parked
    cbz   x21, Ls1
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Ls1:
    cbz   x20, Ls2
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Ls2:
    cbz   x19, Ls3
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
Ls3:
    ldr   x0, [sp, #120]
    str   xzr, [sp, #120]

Lout:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #160
    ret

// ============================================================
// PyObject* _dd_classes(x0 = modname BORROWED) -> tuple of 1 class
// Per class (docless): bases -> _calc_meta -> __prepare__ -> ns
// {__module__, __qualname__, tool_chat_tokens} -> meta(name, bases, ns).
// x19 meta(owned)  x28 result park across cleanup
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

// ===== class 0: GlmToolText(GlmText) =====
Lcls_0:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lc_err
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]                   // GlmText
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
    // tool_chat_tokens (wrapper fn; docless — no __doc__)
    adrp  x0, k_method@PAGE
    add   x0, x0, k_method@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]                    // ns
    ldr   x1, [sp, #176]                    // 'tool_chat_tokens' key NEW
    ldr   x2, [x10, #152]                   // _tool_chat_tokens (borrowed)
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
    mov   x28, xzr
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
