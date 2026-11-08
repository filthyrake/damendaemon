// damendaemon/glm_tool_output.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 40/370. 100% of the module:
//     class GlmToolTokenStream(TokenStream):
//         def __init__(self,text,codec):
//             if not isinstance(text,GlmToolText):
//                 raise ValueError('GLM tools require explicit tool tokenizer')
//             super().__init__(text)
//             self.decoder=ByteLevelStream(text.tokenizer,
//                 CONTROLS+(OPEN,CLOSE,KEY,END_KEY,VALUE,END_VALUE))
//             self.parser=FlashToolOutput(codec)
//         def close(self):self.parser.close()
// Fidelity notes:
//   - isinstance (subtype-true) via PyObject_IsInstance, GlmToolText from
//     the module-39 asm shim (same import-time binding as the original).
//   - the wire tuple literal (OPEN,...,END_VALUE) is built FRESH each call
//     (CPython builds the tuple per evaluation), then CONTROLS+tuple via
//     PyNumber_Add, then decref of the literal.
//   - super().__init__(text) = real super builtin with (defining_class,
//     self), defining class resolved per call from the bridge _CLSREG.
//   - setattr via PyObject_SetAttrString (no steal); NEW refs decref'd
//     after the store. SetAttr errors (-1) checked.
//   - close() returns None regardless of parser.close()'s value.
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear       104 type            112 TokenStream
//   120 GlmToolText        128 ByteLevelStream  136 FlashToolOutput
//   144 CONTROLS           152 OPEN            160 CLOSE
//   168 KEY                176 END_KEY         184 VALUE
//   192 END_VALUE          200 ValueError      208 TypeError
//   216 IsInstance         224 PyNumber_Add    232 SetAttrString
//   240 super              248 _CLSREG         256 key 'GlmToolTokenStream'
//   264 DictGetItemWithError  272 PyErr_Occurred  280 _init shell
//   288 _close shell       296 None

.globl _dd_init
.globl _dd_classes
.globl _dd_tool_stream_init
.globl _dd_tool_stream_close

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
s_prepare:  .asciz "__prepare__"
s_conflict: .asciz "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases"
k_module:   .asciz "__module__"
k_qualname: .asciz "__qualname__"
k_init:     .asciz "__init__"
k_close:    .asciz "close"
a_init:     .asciz "__init__"
a_tokenizer: .asciz "tokenizer"
a_decoder:  .asciz "decoder"
a_parser:   .asciz "parser"
a_close:    .asciz "close"
m_textmsg:  .asciz "GLM tools require explicit tool tokenizer"
cn_0:       .asciz "GlmToolTokenStream"

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
_t_isinstance:     IFACE #216
_t_num_add:        IFACE #224
_t_setattr_s:      IFACE #232
_t_dict_getwe:      IFACE #264
_t_err_occurred:   IFACE #272

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
// PyObject* _dd_tool_stream_init(x0=self, x1=text, x2=codec)  — BORROWED
// x19 self b  x20 text b  x21 codec b
// x22 tok NEW  x23 parser NEW  x24 super_obj NEW  x25 init_m NEW
// x26 s (wire concat) NEW  x27 decoder NEW
// parks: [96] args  [104] wire tuple  [112] result park
// ============================================================
_dd_tool_stream_init:
    stp   x29, x30, [sp, #-160]!
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

    // ---- isinstance(text, GlmToolText) ----
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #120]                   // GlmToolText
    bl    _t_isinstance
    cmn   w0, #1
    b.eq  Lti_err
    cbz   w0, Lti_raise

    // ---- cls = _CLSREG[key] (borrowed) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #248]                   // _CLSREG
    ldr   x1, [x10, #256]                   // key
    bl    _t_dict_getwe
    cbnz  x0, 1f
    bl    _t_err_occurred
    cbnz  x0, Lti_err                       // lookup error -> propagate
    b     Lti_err                           // missing key -> error
1:  mov   x28, x0                           // cls borrowed (x28 park)

    // ---- super_obj = super(cls, self) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lti_err
    str   x0, [sp, #96]
    mov   x0, x28
    bl    _incref
    mov   x0, x19
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x28
    bl    _t_tuple_setitem
    ldr   x0, [sp, #96]
    mov   x1, #1
    mov   x2, x19
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #240]                   // super builtin
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                    // park across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x24, [sp, #112]                   // super_obj NEW
    str   xzr, [sp, #112]
    cbz   x24, Lti_err
    mov   x28, xzr                          // cls park released

    // ---- init_m = super_obj.__init__ ----
    mov   x0, x24
    adrp  x1, a_init@PAGE
    add   x1, x1, a_init@PAGEOFF
    bl    _t_getattr_s
    mov   x25, x0                           // init_m NEW
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    cbz   x25, Lti_err

    // ---- super().__init__(text) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lti_err
    str   x0, [sp, #96]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    mov   x0, x25
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                    // result park (None)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x9, [sp, #112]
    cbnz  x9, Lti_chk_none
    b     Lti_err                           // super init failed (exc set)
Lti_chk_none:
    ldr   x0, [sp, #112]                    // drop the (unused) None
    bl    _t_decref
    str   xzr, [sp, #112]

    // ---- wire = (OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE) ----
    mov   x0, #6
    bl    _t_tuple_new
    cbz   x0, Lti_err
    str   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    // OPEN
    ldr   x0, [x10, #152]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #152]
    bl    _t_tuple_setitem
    // CLOSE
    ldr   x0, [x10, #160]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #1
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #160]
    bl    _t_tuple_setitem
    // KEY
    ldr   x0, [x10, #168]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #2
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #168]
    bl    _t_tuple_setitem
    // END_KEY
    ldr   x0, [x10, #176]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #3
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #176]
    bl    _t_tuple_setitem
    // VALUE
    ldr   x0, [x10, #184]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #4
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #184]
    bl    _t_tuple_setitem
    // END_VALUE
    ldr   x0, [x10, #192]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #5
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #192]
    bl    _t_tuple_setitem

    // ---- s = CONTROLS + wire ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #144]                   // CONTROLS
    ldr   x1, [sp, #104]
    bl    _t_num_add
    str   x0, [sp, #112]                    // park across cleanup
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x26, [sp, #112]                   // s NEW
    str   xzr, [sp, #112]
    cbz   x26, Lti_err

    // ---- tok = text.tokenizer ----
    mov   x0, x20
    adrp  x1, a_tokenizer@PAGE
    add   x1, x1, a_tokenizer@PAGEOFF
    bl    _t_getattr_s
    mov   x22, x0                           // tok NEW
    cbz   x22, Lti_err

    // ---- decoder = ByteLevelStream(tok, s) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lti_err
    str   x0, [sp, #96]
    mov   x0, x22
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    mov   x0, x26
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #1
    mov   x2, x26
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // ByteLevelStream
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                    // park across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x27, [sp, #112]                   // decoder NEW
    str   xzr, [sp, #112]
    cbz   x27, Lti_err

    // ---- self.decoder = decoder ----
    mov   x0, x19
    adrp  x1, a_decoder@PAGE
    add   x1, x1, a_decoder@PAGEOFF
    mov   x2, x27
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lti_err
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr

    // ---- parser = FlashToolOutput(codec) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lti_err
    str   x0, [sp, #96]
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x21
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // FlashToolOutput
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                    // park across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x23, [sp, #112]                   // parser NEW
    str   xzr, [sp, #112]
    cbz   x23, Lti_err

    // ---- self.parser = parser ----
    mov   x0, x19
    adrp  x1, a_parser@PAGE
    add   x1, x1, a_parser@PAGEOFF
    mov   x2, x23
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lti_err
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr

    // ---- success: drop locals, return None ----
    cbz   x22, Lti_s1
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lti_s1:
    cbz   x26, Lti_s2
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lti_s2:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #296]                   // None
    b     Lti_out

Lti_raise:
    cbz   x22, Ltr1
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Ltr1:
    cbz   x26, Ltr2
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Ltr2:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #200]                   // ValueError
    adrp  x1, m_textmsg@PAGE
    add   x1, x1, m_textmsg@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lti_out

Lti_err:
    ldr   x0, [sp, #96]
    cbz   x0, Lte1
    bl    _t_decref
    str   xzr, [sp, #96]
Lte1:
    ldr   x0, [sp, #104]
    cbz   x0, Lte2
    bl    _t_decref
    str   xzr, [sp, #104]
Lte2:
    cbz   x27, Lte3
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
Lte3:
    cbz   x26, Lte4
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Lte4:
    cbz   x23, Lte5
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
Lte5:
    cbz   x22, Lte6
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lte6:
    cbz   x25, Lte7
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
Lte7:
    cbz   x24, Lte8
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
Lte8:
    mov   x0, xzr                           // NULL + pending exception
    b     Lti_out

Lti_out:
    ldr   x9, [sp, #112]                    // parked stray result, if any
    cbz   x9, 1f
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #160
    ret

// ============================================================
// PyObject* _dd_tool_stream_close(x0 = self BORROWED) -> None
// self.parser.close()
// x19 self b  x20 parser NEW  x21 close_m NEW
// parks: [96] args  [104] result
// ============================================================
_dd_tool_stream_close:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #104]
    mov   x19, x0
    mov   x20, xzr
    mov   x21, xzr

    // ---- p = self.parser ----
    mov   x0, x19
    adrp  x1, a_parser@PAGE
    add   x1, x1, a_parser@PAGEOFF
    bl    _t_getattr_s
    mov   x20, x0                           // parser NEW
    cbz   x20, Ltc_err

    // ---- m = p.close ----
    mov   x0, x20
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                           // close_m NEW
    cbz   x21, Ltc_err

    // ---- m() ----
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Ltc_err
    str   x0, [sp, #96]
    mov   x0, x21
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                    // result park across cleanup
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x9, [sp, #104]
    str   xzr, [sp, #104]
    cbz   x9, Ltc_err                       // call failed
    mov   x0, x9
    bl    _t_decref                         // drop the (unused) result

    // ---- success: drop locals, return None ----
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #296]                   // None
    b     Ltc_out

Ltc_err:
    ldr   x0, [sp, #96]
    cbz   x0, Lce1
    bl    _t_decref
    str   xzr, [sp, #96]
Lce1:
    ldr   x0, [sp, #104]
    cbz   x0, Lce2
    bl    _t_decref
    str   xzr, [sp, #104]
Lce2:
    cbz   x21, Lce3
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lce3:
    cbz   x20, Lce4
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Lce4:
    mov   x0, xzr                           // NULL + pending exception
    b     Ltc_out

Ltc_out:
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_classes(x0 = modname BORROWED) -> tuple of 1 class
// Per class (docless): bases -> _calc_meta -> __prepare__ -> ns
// {__module__, __qualname__, __init__, close} -> meta(name, bases, ns).
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

// ===== class 0: GlmToolTokenStream(TokenStream) =====
Lcls_0:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lc_err
    str   x0, [sp, #128]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #112]                   // TokenStream
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
    // __init__ (wrapper fn; docless — no __doc__)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]                    // ns
    ldr   x1, [sp, #176]                    // '__init__' key NEW
    ldr   x2, [x10, #280]                   // _init (borrowed)
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // close (wrapper fn)
    adrp  x0, k_close@PAGE
    add   x0, x0, k_close@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]                    // ns
    ldr   x1, [sp, #176]                    // 'close' key NEW
    ldr   x2, [x10, #288]                   // _close (borrowed)
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
