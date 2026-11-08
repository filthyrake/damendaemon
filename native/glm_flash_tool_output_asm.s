// damendaemon/glm_flash_tool_output.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 41/370. 100% of the module:
//   class FlashToolOutput:
//       __init__(codec)   — isinstance gate + 6 attr inits (GlmOutput())
//       _usable()         — closed/failed gate -> RuntimeError
//       close()           — closed=True, pending='', buffer.close()
//       feed(text)        — byte budget, buffer fast path, OPEN boundary
//                            split, partial-opener suffix scan, BaseException
//                            handler (failed=True, close, bare raise)
//       finish(*,truncated=False) — bool flag gate, stop-marker scan over
//                            STOPS, tool_calls dict event, finally close
//   class FlashToolTokenStream(TokenStream):
//       __init__(text,codec) — FlashToolText isinstance gate + decoder/parser
//       close()             — parser.close()
// Fidelity notes:
//   - except BaseException + bare `raise`: PyErr_Fetch at handler entry
//     (clears the indicator so the except body can run calls), re-raise
//     via PyErr_Restore + NULL return. finally runs with the exception
//     still FETCHED (held), then Restore.
//   - `next((marker for marker in STOPS if value.endswith(marker)), None)`
//     materialized as an ordered index scan over the STOPS tuple — same
//     first-match result, documented divergence (no user code in the
//     genexp; STOPS is a plain tuple of strs).
//   - str slicing via PyUnicode_Substring with explicit non-negative
//     Py_ssize_t bounds (find/boundary/keep/len all int64 in registers;
//     self.bytes stays a Python int via PyNumber_Add — arbitrary range).
//   - `self.pending += text` = PyNumber_InPlaceAdd (str concat idiom);
//     boundary>=0 compare AFTER PyLong_AsLongLong with PyErr_Occurred
//     disambiguation (find legitimately returns -1).
//   - FlashToolOutput is a BASELESS class (empty bases tuple -> type meta).
//   - FlashToolTokenStream.__init__ resolves FlashToolOutput from _CLSREG
//     at call time (class 0 is registered after dd_classes returns).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString       8 PyObject_Call    16 PyTuple_New
//    24 PyTuple_SetItem    32 Py_DecRef       40 PyDict_New
//    48 PyDict_SetItem     56 PyUnicode_FromString  64 PyType_IsSubtype
//    72 PyObject_Size      80 PySequence_GetItem   88 PyErr_SetString
//    96 PyErr_Clear       104 type            112 TokenStream
//   120 FlashTools         128 GlmOutput      136 StringIO
//   144 MAX_TEXT_BYTES     152 CONTROLS       160 OPEN
//   168 _text              176 FlashToolText  184 ByteLevelStream
//   192 CLOSE              200 KEY            208 END_KEY
//   216 VALUE              224 END_VALUE      232 ValueError
//   240 TypeError          248 PyObject_IsTrue 256 RichCompareBool
//   264 PyLong_AsLongLong  272 PyUnicode_Substring 280 SetAttrString
//   288 IsInstance         296 PyNumber_Add   304 InPlaceAdd
//   312 PyErr_Fetch        320 PyErr_Restore  328 PyBool_Type
//   336 RuntimeError       344 PyList_New     352 PyList_SetItem
//   360 'content'          368 'tool_calls'   376 'utf-8'
//   384 STOPS              392 None           400 True
//   408 '' empty str       416 () empty tuple  424 int 0
//   432 _fto_init shell    440 _usable shell  448 _close shell (FTO)
//   456 _feed shell         464 _finish shell  472 _ts_init shell
//   480 _ts_close shell     488 False         496 PyErr_Occurred
//   504 PyLong_FromSsize_t  512 _CLSREG       520 key 'FlashToolOutput'
//   528 key 'FlashToolTokenStream'  536 super  544 DictGetItemWithError

.globl _dd_init
.globl _dd_classes
.globl _dd_fto_init
.globl _dd_fto_usable
.globl _dd_fto_close
.globl _dd_fto_feed
.globl _dd_fto_finish
.globl _dd_ts_init
.globl _dd_ts_close

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
s_prepare:  .asciz "__prepare__"
s_conflict: .asciz "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases"
k_module:   .asciz "__module__"
k_qualname: .asciz "__qualname__"
k_init:     .asciz "__init__"
k_usable:   .asciz "_usable"
k_close:    .asciz "close"
k_feed:     .asciz "feed"
k_finish:   .asciz "finish"
k_truncated: .asciz "truncated"
a_finish:   .asciz "finish"
a_init:     .asciz "__init__"
a_usable:   .asciz "_usable"
a_close:    .asciz "close"
a_feed:     .asciz "feed"
a_codec:    .asciz "codec"
a_ordinary: .asciz "ordinary"
a_pending:  .asciz "pending"
a_buffer:   .asciz "buffer"
a_bytes:    .asciz "bytes"
a_closed:   .asciz "closed"
a_failed:   .asciz "failed"
a_write:    .asciz "write"
a_getvalue: .asciz "getvalue"
a_find:     .asciz "find"
a_startswith: .asciz "startswith"
a_endswith: .asciz "endswith"
a_encode:   .asciz "encode"
a_channel:  .asciz "channel"
a_ended:    .asciz "ended"
a_parse:    .asciz "parse"
a_decoder: .asciz "decoder"
a_parser:   .asciz "parser"
a_tokenizer: .asciz "tokenizer"
m_codec:    .asciz "Flash tool output requires prepared declarations"
m_closed:   .asciz "Flash tool output is closed or failed"
m_limit:    .asciz "Flash tool completion exceeds byte limit"
m_channel:  .asciz "Flash tool calls require a completed reasoning channel"
m_flag:     .asciz "Flash truncation flag must be boolean"
m_incomp:   .asciz "incomplete Flash tool completion"
m_opener:   .asciz "incomplete Flash tool opener"
m_tsmsg:    .asciz "Flash tool stream requires explicit tool tokenizer"
cn_0:       .asciz "FlashToolOutput"
cn_1:       .asciz "FlashToolTokenStream"

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
_t_isinstance:     IFACE #288
_t_rcb:            IFACE #256
_t_aslong:         IFACE #264
_t_substring:      IFACE #272
_t_setattr_s:      IFACE #280
_t_num_add:        IFACE #296
_t_inplace_add:    IFACE #304
_t_fetch:          IFACE #312
_t_restore:        IFACE #320
_t_err_occurred:   IFACE #496
_t_fromssize:      IFACE #504
_t_dict_getwe:     IFACE #544
_t_istrue:         IFACE #248
_t_list_new:       IFACE #344
_t_list_setitem:   IFACE #352

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
    ldr   x0, [x10, #240]                   // TypeError
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
// PyObject* _dd_fto_init(x0=self, x1=codec) — BORROWED
// x19 self b  x20 codec b  x21 ordinary NEW (transient)
// parks: [96] args (transient)
// ============================================================
_dd_fto_init:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   xzr, xzr, [sp, #96]
    stp   xzr, xzr, [sp, #112]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr

    // ---- isinstance(codec, FlashTools) ----
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #120]                   // FlashTools
    bl    _t_isinstance
    cmn   w0, #1
    b.eq  Lfoi_err
    cbz   w0, Lfoi_raise

    // ---- self.codec = codec ----
    mov   x0, x19
    adrp  x1, a_codec@PAGE
    add   x1, x1, a_codec@PAGEOFF
    mov   x2, x20
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err

    // ---- self.ordinary = GlmOutput() ----
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfoi_err
    str   x0, [sp, #96]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #128]                   // GlmOutput
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x21, x0                           // ordinary NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x21, Lfoi_err
    mov   x0, x19
    adrp  x1, a_ordinary@PAGE
    add   x1, x1, a_ordinary@PAGEOFF
    mov   x2, x21
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_drop21_err
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr

    // ---- self.pending = '' / self.buffer = None ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #408]                   // ''
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #392]                   // None
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err

    // ---- self.bytes = 0 / self.closed = False / self.failed = False ----
    mov   x0, x19
    adrp  x1, a_bytes@PAGE
    add   x1, x1, a_bytes@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #424]                   // int 0
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #488]                   // False
    mov   x0, x19
    adrp  x1, a_closed@PAGE
    add   x1, x1, a_closed@PAGEOFF
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #488]                   // False
    mov   x0, x19
    adrp  x1, a_failed@PAGE
    add   x1, x1, a_failed@PAGEOFF
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfoi_err

    // ---- success: return None ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #392]                   // None
    b     Lfoi_out

Lfoi_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_codec@PAGE
    add   x1, x1, m_codec@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lfoi_out
Lfoi_drop21_err:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lfoi_err:
    mov   x0, xzr                           // NULL + pending exception
Lfoi_out:
    // drop leftover args-tuple park (error path only); preserve result in x0
    ldr   x9, [sp, #96]
    cbz   x9, Lfoi_o1
    str   x0, [sp, #112]                    // save result
    mov   x0, x9
    bl    _t_decref
    ldr   x0, [sp, #112]                    // restore result
    str   xzr, [sp, #96]
Lfoi_o1:
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_fto_usable(x0=self BORROWED) -> None or NULL+exc
// x19 self b
// ============================================================
_dd_fto_usable:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    mov   x19, x0

    // if self.closed or self.failed: raise RuntimeError(m_closed)
    mov   x0, x19
    adrp  x1, a_closed@PAGE
    add   x1, x1, a_closed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfu_err
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lfu_err
    cbnz  w0, Lfu_raise
    mov   x0, x19
    adrp  x1, a_failed@PAGE
    add   x1, x1, a_failed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfu_err
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lfu_err
    cbnz  w0, Lfu_raise

    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #392]                   // None
    b     Lfu_out
Lfu_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #336]                   // RuntimeError
    adrp  x1, m_closed@PAGE
    add   x1, x1, m_closed@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Lfu_out
Lfu_err:
    mov   x0, xzr
Lfu_out:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_fto_close(x0=self BORROWED) -> None or NULL+exc
// x19 self b  x20 buffer NEW  x21 close_m NEW
// parks: [24] args  [32] result
// ============================================================
_dd_fto_close:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   xzr, xzr, [sp, #48]
    mov   x19, x0
    mov   x20, xzr
    mov   x21, xzr

    // ---- self.closed = True ----
    mov   x0, x19
    adrp  x1, a_closed@PAGE
    add   x1, x1, a_closed@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #400]                   // True
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfc_err

    // ---- self.pending = '' ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #408]                   // ''
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfc_err

    // ---- if self.buffer is not None: self.buffer.close() ----
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    bl    _t_getattr_s
    mov   x20, x0                           // buffer NEW
    cbz   x20, Lfc_err
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #392]                   // None
    cmp   x20, x9
    b.eq  Lfc_succ                         // buffer is None -> skip
    mov   x0, x20
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                           // close_m NEW
    cbz   x21, Lfc_err
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfc_err
    str   x0, [sp, #48]
    mov   x0, x21
    ldr   x1, [sp, #48]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #56]
    ldr   x0, [sp, #48]
    bl    _t_decref
    str   xzr, [sp, #48]
    ldr   x9, [sp, #56]
    str   xzr, [sp, #56]
    cbz   x9, Lfc_err
    mov   x0, x9
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr

Lfc_succ:
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #392]                   // None
    b     Lfc_out
Lfc_err:
    ldr   x0, [sp, #48]
    cbz   x0, Lfc_e1
    bl    _t_decref
    str   xzr, [sp, #48]
Lfc_e1:
    cbz   x21, Lfc_e2
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
Lfc_e2:
    cbz   x20, Lfc_e3
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
Lfc_e3:
    mov   x0, xzr                           // NULL + pending exception
Lfc_out:
    ldr   x9, [sp, #56]
    cbz   x9, 1f
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #56]
1:  ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_fto_feed(x0=self, x1=text BORROWED) -> list or NULL
// x19 self b  x20 text b  x21 pending obj (live during suffix loop)
// x22 length  x23 plen  x24 keep  x25 boundary  x26 top  x27 temp int64
// parks: [104] args  [112] tempA  [120] tempB  [128] events  [136] result
// fetch: [144] [152] [160]
// except BaseException -> failed=True, close, bare raise (Fetch/Restore)
// ============================================================
_dd_fto_feed:
    stp   x29, x30, [sp, #-176]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #104]
    stp   xzr, xzr, [sp, #120]
    stp   xzr, xzr, [sp, #136]
    stp   xzr, xzr, [sp, #152]
    str   xzr, [sp, #144]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr
    mov   x26, xzr
    mov   x27, xzr

    // ---- self._usable() ----
    mov   x0, x19
    adrp  x1, a_usable@PAGE
    add   x1, x1, a_usable@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_pre_err
    str   x0, [sp, #112]                     // m NEW (bound method)
    mov   x0, #0                            // empty args: bound method already carries self
    bl    _t_tuple_new
    cbz   x0, Lfe_pre_drop112
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfe_pre_err                   // OUTSIDE try: propagate raw
    mov   x0, x9
    bl    _t_decref

    // ---- _text(text) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_err
    str   x0, [sp, #104]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #168]                    // _text
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #112]
    str   xzr, [sp, #112]
    cbz   x9, Lfe_err
    mov   x0, x9
    bl    _t_decref

    // ---- n = len(text.encode('utf-8')) ----
    mov   x0, x20
    adrp  x1, a_encode@PAGE
    add   x1, x1, a_encode@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // encode m NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop112_err
    str   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #376]                    // 'utf-8'
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #376]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]                     // b NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x0, Lfe_err
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfe_err
    mov   x27, x0                            // n
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]

    // ---- self.bytes = self.bytes + n ----
    mov   x0, x19
    adrp  x1, a_bytes@PAGE
    add   x1, x1, a_bytes@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // old bytes NEW
    mov   x0, x27
    bl    _t_fromssize
    cbz   x0, Lfe_drop112_err
    str   x0, [sp, #104]                     // nobj NEW
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    bl    _t_num_add
    str   x0, [sp, #120]                     // park nv across cleanup
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfe_err
    str   x9, [sp, #120]                     // re-park: x9 caller-saved
    mov   x0, x19
    adrp  x1, a_bytes@PAGE
    add   x1, x1, a_bytes@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfe_drop120_err
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    mov   x0, x9
    bl    _t_decref

    // ---- if self.bytes > MAX_TEXT_BYTES: raise ----
    mov   x0, x19
    adrp  x1, a_bytes@PAGE
    add   x1, x1, a_bytes@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // bytes obj NEW
    ldr   x0, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #144]                   // MAX_TEXT_BYTES
    mov   x2, #4                            // Py_GT
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lfe_drop112_err
    cmp   w0, #1
    b.eq  Lfe_over_limit
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    b     Lfe_bufcheck

Lfe_over_limit:
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_limit@PAGE
    add   x1, x1, m_limit@PAGEOFF
    bl    _t_seterr
    b     Lfe_except

    // ---- if self.buffer is not None: write + return [] ----
Lfe_bufcheck:
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // buffer NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #392]                   // None
    ldr   x10, [sp, #112]
    cmp   x10, x9
    b.eq  Lfe_drop_buf_none                 // buffer is None -> pending path
    // self.buffer.write(text)
    ldr   x0, [sp, #112]
    adrp  x1, a_write@PAGE
    add   x1, x1, a_write@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112_err
    str   x0, [sp, #120]                     // w NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop120_err
    str   x0, [sp, #104]
    mov   x0, x20
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    mov   x2, x20
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // park write result
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #128]
    str   xzr, [sp, #128]
    cbz   x9, Lfe_err
    mov   x0, x9
    bl    _t_decref
    // return []
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lfe_err
    b     Lfe_ret

Lfe_drop_buf_none:
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

    // ---- self.pending += text ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // p NEW
    ldr   x0, [sp, #112]
    mov   x1, x20
    bl    _t_inplace_add
    str   x0, [sp, #120]                     // p2 NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfe_err
    str   x9, [sp, #120]                     // re-park: x9 is caller-saved
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    ldr   x2, [sp, #120]
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfe_drop120_err
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    mov   x0, x9
    bl    _t_decref

    // ---- boundary = self.pending.find(OPEN) ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // p NEW
    ldr   x0, [sp, #112]
    adrp  x1, a_find@PAGE
    add   x1, x1, a_find@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112_err
    str   x0, [sp, #120]                     // f NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop120_err
    str   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // OPEN
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #160]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // bobj NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #128]
    str   xzr, [sp, #128]
    cbz   x0, Lfe_err
    bl    _t_aslong
    cmn   x0, #1
    b.ne  Lfe_find_ok
    mov   x25, #-1                          // keep -1 (err_occurred clobbers x0)
    bl    _t_err_occurred
    cbnz  x0, Lfe_bobj_err                  // real error
    mov   x0, x25                           // restore -1
Lfe_find_ok:
    mov   x25, x0                            // boundary int64
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]

    // ---- if boundary >= 0: opener split ----
    cmp   x25, #0
    b.lt  Lfe_suffix
    // events = self.ordinary.feed(self.pending[:boundary])
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]                     // p NEW
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x25
    bl    _t_substring
    str   x0, [sp, #120]                     // s NEW
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfe_err
    mov   x27, x9                           // s NEW in x27
    // ---- ordinary.feed(s) ----
    mov   x0, x19
    adrp  x1, a_ordinary@PAGE
    add   x1, x1, a_ordinary@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_s_err
    str   x0, [sp, #112]                     // ordinary NEW
    ldr   x0, [sp, #112]
    adrp  x1, a_feed@PAGE
    add   x1, x1, a_feed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_s112_err
    str   x0, [sp, #120]                     // feed m NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop_s112120_err
    str   x0, [sp, #104]
    mov   x0, x27
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    mov   x2, x27
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // events NEW (park)
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x9, [sp, #128]
    str   xzr, [sp, #128]
    cbz   x9, Lfe_err
    str   x9, [sp, #128]                     // events stays parked

    // ---- channel/ended gate ----
    mov   x0, x19
    adrp  x1, a_ordinary@PAGE
    add   x1, x1, a_ordinary@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_events_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    adrp  x1, a_channel@PAGE
    add   x1, x1, a_channel@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112_events_err
    str   x0, [sp, #120]                     // ch NEW
    ldr   x0, [sp, #120]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #360]                   // 'content'
    mov   x2, #3                            // Py_NE
    bl    _t_rcb
    cmn   w0, #1
    b.eq  Lfe_drop112120_events_err
    cmp   w0, #1
    b.eq  Lfe_channel_raise
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    adrp  x1, a_ended@PAGE
    add   x1, x1, a_ended@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112_events_err
    str   x0, [sp, #120]                     // en NEW
    ldr   x0, [sp, #120]
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lfe_drop112120_events_err
    cmp   w0, #1
    b.eq  Lfe_channel_raise_drop
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]

    // ---- self.buffer = StringIO() ----
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfe_drop_events_err
    str   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #136]                   // StringIO
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                     // sio NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #112]
    str   xzr, [sp, #112]
    cbz   x9, Lfe_drop_events_err
    str   x9, [sp, #112]                     // re-park: x9 caller-saved
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    ldr   x2, [sp, #112]
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfe_sio_events_err
    ldr   x9, [sp, #112]
    str   xzr, [sp, #112]
    mov   x0, x9
    bl    _t_decref

    // ---- self.buffer.write(self.pending[boundary:]) ----
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_events_err
    str   x0, [sp, #112]                     // buf NEW
    ldr   x0, [sp, #112]
    adrp  x1, a_write@PAGE
    add   x1, x1, a_write@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112_events_err
    str   x0, [sp, #120]                     // w NEW
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop112120_events_err
    str   x0, [sp, #104]                     // p NEW
    ldr   x0, [sp, #104]
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfe_dropall_events_err
    mov   x27, x0                            // plen
    ldr   x0, [sp, #104]
    mov   x1, x25
    mov   x2, x27
    bl    _t_substring
    str   x0, [sp, #136]                     // park s2 at [136]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfe_drop112120_events_err
    mov   x27, x9                           // s2 NEW — x27 reused (plen done)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop_all_s2_err
    str   x0, [sp, #104]
    mov   x0, x27
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    mov   x2, x27
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #136]                     // write result park at [136]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfe_drop_events_err
    mov   x0, x9
    bl    _t_decref

    // ---- self.pending = '' ; return events ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #408]
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfe_drop_events_err
    ldr   x0, [sp, #128]
    str   xzr, [sp, #128]
    b     Lfe_ret

    // ================= suffix scan: no opener yet =================
Lfe_suffix:
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    mov   x21, x0                           // p NEW — live across loop
    mov   x0, x21
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfe_drop21_err
    mov   x23, x0                           // plen
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // OPEN
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfe_drop21_err
    sub   x26, x0, #1                        // len(OPEN)-1
    // mn = min(plen, len(OPEN)-1); top = mn+1
    cmp   x23, x26
    csel  x26, x23, x26, le
    add   x26, x26, #1
    mov   x24, #0                            // keep
    mov   x22, #1                            // length
Lfe_loop:
    cmp   x22, x26
    b.ge  Lfe_loop_done
    // tail = p[-length:] = Substring(p, plen-length, plen)
    sub   x27, x23, x22
    mov   x0, x21
    mov   x1, x27
    mov   x2, x23
    bl    _t_substring
    cbz   x0, Lfe_drop21_err
    str   x0, [sp, #112]                     // tail NEW
    // sw = OPEN.startswith; call sw(tail)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // OPEN
    adrp  x1, a_startswith@PAGE
    add   x1, x1, a_startswith@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop21_112_err
    str   x0, [sp, #120]                     // sw NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop21_112120_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    ldr   x2, [sp, #112]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #136]                     // r park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x27, [sp, #136]                   // r -> callee-saved immediately
    str   xzr, [sp, #136]
    cbz   x27, Lfe_drop21_112120_err
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    // r truthy? -> keep = length
    mov   x0, x27
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lfe_drop21_r_err
    cmp   w0, #1
    b.ne  1f
    mov   x24, x22                          // keep = length
1:  mov   x0, x27
    bl    _t_decref
    add   x22, x22, #1
    b     Lfe_loop
Lfe_loop_done:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr

    // boundary = plen - keep
    sub   x25, x23, x24
    // events = ordinary.feed(p[:boundary]) if boundary else []
    cmp   x25, #0
    b.eq  Lfe_ev_empty
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x25
    bl    _t_substring
    str   x0, [sp, #136]                     // s park
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfe_err
    mov   x27, x9                           // s in x27
    mov   x0, x19
    adrp  x1, a_ordinary@PAGE
    add   x1, x1, a_ordinary@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_s_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    adrp  x1, a_feed@PAGE
    add   x1, x1, a_feed@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_s112_err
    str   x0, [sp, #120]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfe_drop_s112120_err
    str   x0, [sp, #104]
    mov   x0, x27
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    mov   x2, x27
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // events park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x9, [sp, #128]
    str   xzr, [sp, #128]
    cbz   x9, Lfe_err
    str   x9, [sp, #128]
    b     Lfe_ev_have
Lfe_ev_empty:
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lfe_err
    str   x0, [sp, #128]
Lfe_ev_have:
    // self.pending = self.pending[boundary:]
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_drop_events_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #112]
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfe_drop112_events_err
    mov   x27, x0                           // plen (reload)
    ldr   x0, [sp, #112]
    mov   x1, x25
    mov   x2, x27
    bl    _t_substring
    str   x0, [sp, #136]                     // s2 park
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfe_drop_events_err
    str   x9, [sp, #136]                     // re-park: x9 caller-saved
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    ldr   x2, [sp, #136]
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Lfe_drop_s2_events_err
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    mov   x0, x9
    bl    _t_decref
    // return events
    ldr   x0, [sp, #128]
    str   xzr, [sp, #128]
    b     Lfe_ret

// ================= drop labels =================
Lfe_drop_s2_events_err:
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    b     Lfe_drop_events_err
Lfe_drop21_r_err:
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    b     Lfe_drop21_err
Lfe_drop21_112120_err:
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
Lfe_drop21_112_err:
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
Lfe_drop21_err:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    b     Lfe_err
Lfe_drop_all_s2_err:
    cbz   x27, 1f
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  ldr   x0, [sp, #120]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #120]
3:  b     Lfe_drop_events_err
Lfe_dropall_events_err:
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  ldr   x0, [sp, #120]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #120]
3:  b     Lfe_drop_events_err
Lfe_sio_events_err:
    mov   x0, x9
    bl    _t_decref
    b     Lfe_drop_events_err
Lfe_channel_raise:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_channel@PAGE
    add   x1, x1, m_channel@PAGEOFF
    bl    _t_seterr
    b     Lfe_drop_events_except
Lfe_channel_raise_drop:
    b     Lfe_channel_raise
Lfe_drop112120_events_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  b     Lfe_drop_events_err
Lfe_drop112_events_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfe_drop_events_err
Lfe_drop_events_err:
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  b     Lfe_err
Lfe_drop_events_except:
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  b     Lfe_except
Lfe_drop_s_err:
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    b     Lfe_err
Lfe_drop_s112_err:
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfe_err
Lfe_drop_s112120_err:
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  ldr   x0, [sp, #120]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #120]
2:  b     Lfe_err
Lfe_dropnv_err:
    mov   x0, x9
    bl    _t_decref
    b     Lfe_err
Lfe_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfe_err
Lfe_drop120_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  b     Lfe_err
Lfe_bobj_err:
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  b     Lfe_err
Lfe_pre_drop112:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfe_pre_err

// ================= generic try-failure: sweep everything =================
Lfe_err:
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  ldr   x0, [sp, #120]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #120]
3:  ldr   x0, [sp, #128]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #128]
4:  ldr   x0, [sp, #136]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #136]
5:  cbz   x21, 6f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
6:  cbz   x27, 7f
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
7:  b     Lfe_except

// ================= except BaseException: failed=True, close, bare raise =================
Lfe_except:
    // PyErr_Fetch(&t, &v, &tb) — clears the indicator
    add   x0, sp, #144
    add   x1, sp, #152
    add   x2, sp, #160
    bl    _t_fetch
    // self.failed = True
    mov   x0, x19
    adrp  x1, a_failed@PAGE
    add   x1, x1, a_failed@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #400]                   // True
    bl    _t_setattr_s
    // self.close()
    mov   x0, x19
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfe_restore_null              // cannot close; still re-raise
    str   x0, [sp, #104]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfe_closem_104_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfe_restore_null
    mov   x0, x9
    bl    _t_decref
    b     Lfe_restore_null

Lfe_closem_104_err:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]

// re-raise: PyErr_Restore(t, v, tb); return NULL
Lfe_restore_null:
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #152]
    ldr   x2, [sp, #160]
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #160]
    bl    _t_restore
    mov   x0, xzr
    b     Lfe_out

// ================= pre-try failure (self._usable()): raw propagate =================
Lfe_pre_err:
    mov   x0, xzr
    b     Lfe_out

// ================= success exit =================
Lfe_ret:
    // x0 = result; parks empty by construction
    b     Lfe_out

Lfe_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #176
    ret

// ============================================================
// PyObject* _dd_fto_finish(x0=self, x1=truncated BORROWED)
//   -> list event or None or NULL+exc
// try: flag gate; buffer branch (STOPS scan, tool_calls dict);
//      pending gate; ordinary.finish(truncated=truncated)
// except BaseException: failed=True, bare raise
// finally: self.close()
// x19 self b  x20 truncated b  x21 value NEW  x22 stop NEW
// x23 i/size int64  x24 vlen int64  x25 temp int64
// parks: [104] args  [112] tempA  [120] tempB  [128] result park
// [136] tempC  fetch [144] [152] [160]
// ============================================================
_dd_fto_finish:
    stp   x29, x30, [sp, #-176]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    mov   x28, xzr
    stp   xzr, xzr, [sp, #104]
    stp   xzr, xzr, [sp, #120]
    stp   xzr, xzr, [sp, #136]
    stp   xzr, xzr, [sp, #152]
    str   xzr, [sp, #144]
    mov   x19, x0
    mov   x20, x1
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr

    // ---- self._usable() — OUTSIDE try ----
    mov   x0, x19
    adrp  x1, a_usable@PAGE
    add   x1, x1, a_usable@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_pre_err
    str   x0, [sp, #112]
    mov   x0, #0                            // bound method: empty args
    bl    _t_tuple_new
    cbz   x0, Lfi_pre_drop112
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfi_pre_err
    mov   x0, x9
    bl    _t_decref

    // ---- if type(truncated) is not bool: raise ----
    ldr   x23, [x20, #8]                    // ob_type
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #328]                   // PyBool_Type
    cmp   x23, x9
    b.ne  Lfi_flag_raise

    // ---- if self.buffer is not None: tool completion ----
    mov   x0, x19
    adrp  x1, a_buffer@PAGE
    add   x1, x1, a_buffer@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_err
    str   x0, [sp, #112]                     // buffer NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x9, [x10, #392]                   // None
    ldr   x10, [sp, #112]
    cmp   x10, x9
    b.eq  Lfi_no_buffer
    // value = self.buffer.getvalue()
    ldr   x0, [sp, #112]
    adrp  x1, a_getvalue@PAGE
    add   x1, x1, a_getvalue@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop112_err
    str   x0, [sp, #120]                     // gv m NEW
    ldr   x0, [sp, #120]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfi_drop112120_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #128]                     // value park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #128]
    str   xzr, [sp, #128]
    cbz   x9, Lfi_err
    mov   x21, x9                           // value NEW

    // stop = first marker in STOPS where value.endswith(marker)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #384]                    // STOPS
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfi_drop_value_err
    mov   x23, x0                            // n
    mov   x28, #0                            // i (NOT swept: x22 is object-slot)
    mov   x24, #0                            // vlen lazily? compute now
    mov   x0, x21
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfi_drop_value_err
    mov   x24, x0                            // vlen
Lfi_scan:
    cmp   x28, x23
    b.ge  Lfi_no_stop
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #384]                    // STOPS
    mov   x1, x28
    bl    _t_seq_getitem
    cbz   x0, Lfi_drop_value_err
    str   x0, [sp, #112]                     // marker NEW
    mov   x0, x21
    adrp  x1, a_endswith@PAGE
    add   x1, x1, a_endswith@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop_value_112_err
    str   x0, [sp, #120]                     // ew NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfi_drop_value_112120_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    ldr   x2, [sp, #112]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #136]                     // r park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfi_drop_value_112_err
    mov   x25, x9                            // hold r across istrue
    mov   x0, x9
    bl    _t_istrue
    cmn   w0, #1
    b.eq  Lfi_drop_value_112_r_err
    cmp   w0, #1
    b.ne  Lfi_scan_next
    // ---- match: stop = marker (steal [112] ref); drop r ----
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x22, [sp, #112]                    // stop NEW
    str   xzr, [sp, #112]
    b     Lfi_stop_found
Lfi_scan_next:
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    add   x28, x28, #1
    b     Lfi_scan
Lfi_stop_found:                             // stop x22 NEW, value x21 NEW, vlen x24
    // slen = len(stop)
    mov   x0, x22
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfi_drop_value_stop_err
    // s = value[:-slen] = Substring(value, 0, vlen-slen)
    sub   x2, x24, x0
    mov   x0, x21
    mov   x1, #0
    bl    _t_substring
    cbz   x0, Lfi_drop_value_stop_err
    str   x0, [sp, #112]                     // s NEW
    // codec = self.codec
    mov   x0, x19
    adrp  x1, a_codec@PAGE
    add   x1, x1, a_codec@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop_vs_112_err
    str   x0, [sp, #120]                     // codec NEW
    // parse = codec.parse
    ldr   x0, [sp, #120]
    adrp  x1, a_parse@PAGE
    add   x1, x1, a_parse@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop_vs_112120_err
    str   x0, [sp, #128]                     // parse m NEW
    // args = (s,)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lfi_drop_vs_112120128_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    ldr   x2, [sp, #112]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #104]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #136]                     // calls park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x9, [sp, #136]
    str   xzr, [sp, #136]
    cbz   x9, Lfi_err
    str   x9, [sp, #136]                     // calls stays parked
    // d = {'tool_calls': calls}
    bl    _t_dict_new
    cbz   x0, Lfi_drop_calls_err
    str   x0, [sp, #104]                     // d NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #104]                     // dict
    ldr   x1, [x10, #368]                    // 'tool_calls' key
    ldr   x2, [sp, #136]                     // calls
    bl    _t_dict_setitem
    cmn   w0, #1
    b.eq  Lfi_drop_d_calls_err
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    // result = [d]
    mov   x0, #1
    bl    _t_list_new
    cbz   x0, Lfi_drop_d_err
    str   x0, [sp, #136]                     // result list park
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #136]
    mov   x1, #0
    ldr   x2, [sp, #104]
    bl    _t_list_setitem
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    // drop value/stop locals
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    b     Lfi_succ_park_finally             // result already parked at [136]

// ---- scan exhausted: stop is None -> ValueError (except flow) ----
Lfi_no_stop:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_incomp@PAGE
    add   x1, x1, m_incomp@PAGEOFF
    bl    _t_seterr
    b     Lfi_err

// ---- buffer is None: pending gate then ordinary.finish ----
Lfi_no_buffer:
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    // if self.pending: raise
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_err
    str   x0, [sp, #112]                     // pending NEW
    ldr   x0, [sp, #112]
    bl    _t_size
    cmn   x0, #1
    b.eq  Lfi_drop112_err
    cbnz  x0, Lfi_opener_raise
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    // kwargs = {'truncated': truncated}
    bl    _t_dict_new
    cbz   x0, Lfi_err
    str   x0, [sp, #120]                     // kw NEW
    adrp  x0, k_truncated@PAGE
    add   x0, x0, k_truncated@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lfi_drop120_err
    str   x0, [sp, #112]                     // key NEW
    ldr   x0, [sp, #120]
    ldr   x1, [sp, #112]
    mov   x2, x20
    bl    _t_dict_setitem
    cmn   w0, #1
    b.eq  Lfi_drop112120_err
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    // ordinary = self.ordinary; fm = ordinary.finish
    mov   x0, x19
    adrp  x1, a_ordinary@PAGE
    add   x1, x1, a_ordinary@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop120_err
    str   x0, [sp, #112]                     // ordinary NEW
    ldr   x0, [sp, #112]
    adrp  x1, a_finish@PAGE
    add   x1, x1, a_finish@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_drop112120_err
    str   x0, [sp, #128]                     // fm NEW
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfi_drop112120128_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #128]
    ldr   x1, [sp, #104]
    ldr   x2, [sp, #120]
    bl    _t_call
    str   x0, [sp, #136]                     // result park
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #128]
    bl    _t_decref
    str   xzr, [sp, #128]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #120]
    bl    _t_decref
    str   xzr, [sp, #120]
    ldr   x9, [sp, #136]
    cbz   x9, Lfi_err
    b     Lfi_succ_park_finally             // result stays parked at [136]

Lfi_opener_raise:
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_opener@PAGE
    add   x1, x1, m_opener@PAGEOFF
    bl    _t_seterr
    b     Lfi_err

Lfi_flag_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_flag@PAGE
    add   x1, x1, m_flag@PAGEOFF
    bl    _t_seterr
    b     Lfi_err

// ---- success finally: result in x0 ----
Lfi_succ_finally:
    str   x0, [sp, #136]
Lfi_succ_park_finally:                      // result parked at [136]
    // finally: self.close()
    mov   x0, x19
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_finally_fail
    str   x0, [sp, #104]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfi_finally_fail104
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfi_finally_fail              // close raised in finally
    mov   x0, x9
    bl    _t_decref
    ldr   x0, [sp, #136]
    str   xzr, [sp, #136]
    b     Lfi_out
Lfi_finally_fail104:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
Lfi_finally_fail:                           // close's exception propagates
    ldr   x0, [sp, #136]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #136]
1:  mov   x0, xzr
    b     Lfi_out

// ---- drop labels (chain down to the sweep) ----
Lfi_drop_d_calls_err:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    b     Lfi_drop_calls_err
Lfi_drop_d_err:
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  b     Lfi_err
Lfi_drop_calls_err:
    ldr   x0, [sp, #136]
    bl    _t_decref
    str   xzr, [sp, #136]
    b     Lfi_err
Lfi_drop_vs_112120128_err:
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  ldr   x0, [sp, #120]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #120]
2:  ldr   x0, [sp, #112]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #112]
3:  b     Lfi_drop_vs_err
Lfi_drop_vs_112120_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  b     Lfi_drop_vs_err
Lfi_drop_vs_112_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfi_drop_vs_err
Lfi_drop_value_stop_err:                    // drop x22, x21
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Lfi_drop_vs_err:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    b     Lfi_err
Lfi_drop_value_112_r_err:
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    b     Lfi_drop_value_112_err
Lfi_drop_value_112120_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  b     Lfi_drop_value_112_err
Lfi_drop_value_112_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfi_drop_value_err
Lfi_drop_value_err:
    b     Lfi_drop_vs_err
Lfi_drop112120128_err:
    ldr   x0, [sp, #128]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #128]
1:  b     Lfi_drop112120_err
Lfi_drop112120_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  b     Lfi_err
Lfi_drop112_err:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfi_err
Lfi_drop120_err:
    ldr   x0, [sp, #120]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #120]
1:  b     Lfi_err
Lfi_pre_drop112:
    ldr   x0, [sp, #112]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #112]
1:  b     Lfi_pre_err

// ---- sweep everything, then except flow ----
Lfi_err:
    ldr   x0, [sp, #104]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #104]
1:  ldr   x0, [sp, #112]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #112]
2:  ldr   x0, [sp, #120]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #120]
3:  ldr   x0, [sp, #128]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #128]
4:  ldr   x0, [sp, #136]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #136]
5:  cbz   x21, 6f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
6:  cbz   x22, 7f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
7:  cbz   x25, 8f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
8:  b     Lfi_except

// ---- except BaseException: failed=True, finally close, bare raise ----
Lfi_except:
    add   x0, sp, #144
    add   x1, sp, #152
    add   x2, sp, #160
    bl    _t_fetch
    mov   x0, x19
    adrp  x1, a_failed@PAGE
    add   x1, x1, a_failed@PAGEOFF
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #400]                   // True
    bl    _t_setattr_s
    // finally: self.close()
    mov   x0, x19
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lfi_exc_restore
    str   x0, [sp, #104]
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lfi_exc_closem_err
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #120]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x9, [sp, #120]
    str   xzr, [sp, #120]
    cbz   x9, Lfi_exc_restore
    mov   x0, x9
    bl    _t_decref
    b     Lfi_exc_restore
Lfi_exc_closem_err:
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
Lfi_exc_restore:
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #152]
    ldr   x2, [sp, #160]
    str   xzr, [sp, #144]
    str   xzr, [sp, #152]
    str   xzr, [sp, #160]
    bl    _t_restore
    mov   x0, xzr
    b     Lfi_out

Lfi_pre_err:
    mov   x0, xzr
    b     Lfi_out

Lfi_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #176
    ret

// ============================================================
// PyObject* _dd_ts_init(x0=self, x1=text, x2=codec) — BORROWED
// x19 self b  x20 text b  x21 codec b  x22 tok NEW  x23 parser NEW
// x24 super_obj NEW  x25 init_m NEW  x26 s NEW  x27 decoder NEW
// parks: [96] args  [104] wire  [112] result
// ============================================================
_dd_ts_init:
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

    // ---- isinstance(text, FlashToolText) ----
    mov   x0, x20
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x1, [x10, #176]                   // FlashToolText
    bl    _t_isinstance
    cmn   w0, #1
    b.eq  Ltsi_err
    cbz   w0, Ltsi_raise

    // ---- cls = _CLSREG['FlashToolTokenStream'] ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #512]                   // _CLSREG
    ldr   x1, [x10, #528]                   // key
    bl    _t_dict_getwe
    cbnz  x0, 1f
    b     Ltsi_err
1:  mov   x28, x0                           // cls borrowed

    // ---- super_obj = super(cls, self) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Ltsi_err
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
    ldr   x0, [x10, #536]                   // super builtin
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x24, [sp, #112]                   // super_obj NEW
    str   xzr, [sp, #112]
    cbz   x24, Ltsi_err
    mov   x28, xzr

    // ---- init_m = super_obj.__init__ ----
    mov   x0, x24
    adrp  x1, a_init@PAGE
    add   x1, x1, a_init@PAGEOFF
    bl    _t_getattr_s
    mov   x25, x0
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    cbz   x25, Ltsi_err

    // ---- super().__init__(text) ----
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ltsi_err
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
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
    ldr   x9, [sp, #112]
    cbnz  x9, Ltsi_chk
    b     Ltsi_err
Ltsi_chk:
    mov   x0, x9
    bl    _t_decref
    str   xzr, [sp, #112]

    // ---- wire = (OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE) ----
    // fresh tuple per call (CPython builds the literal per evaluation)
    mov   x0, #6
    bl    _t_tuple_new
    cbz   x0, Ltsi_err
    str   x0, [sp, #104]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    // OPEN
    ldr   x0, [x10, #160]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #0
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #160]
    bl    _t_tuple_setitem
    // CLOSE
    ldr   x0, [x10, #192]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #1
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #192]
    bl    _t_tuple_setitem
    // KEY
    ldr   x0, [x10, #200]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #2
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #200]
    bl    _t_tuple_setitem
    // END_KEY
    ldr   x0, [x10, #208]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #3
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #208]
    bl    _t_tuple_setitem
    // VALUE
    ldr   x0, [x10, #216]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #4
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #216]
    bl    _t_tuple_setitem
    // END_VALUE
    ldr   x0, [x10, #224]
    bl    _incref
    ldr   x0, [sp, #104]
    mov   x1, #5
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #224]
    bl    _t_tuple_setitem


    // ---- s = CONTROLS + wire ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #152]                   // CONTROLS
    ldr   x1, [sp, #104]
    bl    _t_num_add
    str   x0, [sp, #112]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x26, [sp, #112]                   // s NEW
    str   xzr, [sp, #112]
    cbz   x26, Ltsi_err

    // ---- tok = text.tokenizer ----
    mov   x0, x20
    adrp  x1, a_tokenizer@PAGE
    add   x1, x1, a_tokenizer@PAGEOFF
    bl    _t_getattr_s
    mov   x22, x0                           // tok NEW
    cbz   x22, Ltsi_err

    // ---- decoder = ByteLevelStream(tok, s) ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Ltsi_err
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
    ldr   x0, [x10, #184]                   // ByteLevelStream
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x27, [sp, #112]                   // decoder NEW
    str   xzr, [sp, #112]
    cbz   x27, Ltsi_err


    // ---- self.decoder = decoder ----
    mov   x0, x19
    adrp  x1, a_decoder@PAGE
    add   x1, x1, a_decoder@PAGEOFF
    mov   x2, x27
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Ltsi_drop27_err
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr


    // ---- parser = _CLSREG['FlashToolOutput'](codec) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #512]                   // _CLSREG
    ldr   x1, [x10, #520]                   // key 'FlashToolOutput'
    bl    _t_dict_getwe
    cbnz  x0, 1f
    b     Ltsi_err
1:  bl    _incref                           // convert borrowed -> owned
    str   x0, [sp, #96]                     // cls NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Ltsi_drop96_err
    str   x0, [sp, #112]
    mov   x0, x21
    bl    _incref
    ldr   x0, [sp, #112]
    mov   x1, #0
    mov   x2, x21
    bl    _t_tuple_setitem
    ldr   x0, [sp, #96]
    ldr   x1, [sp, #112]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #104]                    // parser park
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x23, [sp, #104]                   // parser NEW
    str   xzr, [sp, #104]
    cbz   x23, Ltsi_err

    // ---- self.parser = parser ----
    mov   x0, x19
    adrp  x1, a_parser@PAGE
    add   x1, x1, a_parser@PAGEOFF
    mov   x2, x23
    bl    _t_setattr_s
    cmn   w0, #1
    b.eq  Ltsi_drop23_err
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr

    // ---- success ----
    cbz   x22, Ltsi_s1
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
Ltsi_s1:
    cbz   x26, Ltsi_s2
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
Ltsi_s2:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #392]                   // None
    b     Ltsi_out

Ltsi_raise:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #232]                   // ValueError
    adrp  x1, m_tsmsg@PAGE
    add   x1, x1, m_tsmsg@PAGEOFF
    bl    _t_seterr
    mov   x0, xzr
    b     Ltsi_out

Ltsi_drop23_err:
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    b     Ltsi_err
Ltsi_drop27_err:
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
    b     Ltsi_err
Ltsi_drop96_err:
    ldr   x0, [sp, #96]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #96]
1:  b     Ltsi_err
Ltsi_err:
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
3:  cbz   x27, 4f
    mov   x0, x27
    bl    _t_decref
    mov   x27, xzr
4:  cbz   x26, 5f
    mov   x0, x26
    bl    _t_decref
    mov   x26, xzr
5:  cbz   x23, 6f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
6:  cbz   x22, 7f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
7:  cbz   x25, 8f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
8:  cbz   x24, 9f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
9:  mov   x0, xzr                           // NULL + exception
    b     Ltsi_out

Ltsi_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #160
    ret

// ============================================================
// PyObject* _dd_ts_close(x0 = self BORROWED) -> None
// self.parser.close()
// ============================================================
_dd_ts_close:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   xzr, xzr, [sp, #48]
    stp   xzr, xzr, [sp, #56]
    mov   x19, x0
    mov   x20, xzr
    mov   x21, xzr

    mov   x0, x19
    adrp  x1, a_parser@PAGE
    add   x1, x1, a_parser@PAGEOFF
    bl    _t_getattr_s
    mov   x20, x0                           // parser NEW
    cbz   x20, Ltsc_err
    mov   x0, x20
    adrp  x1, a_close@PAGE
    add   x1, x1, a_close@PAGEOFF
    bl    _t_getattr_s
    mov   x21, x0                           // close_m NEW
    cbz   x21, Ltsc_err
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Ltsc_err
    str   x0, [sp, #48]
    mov   x0, x21
    ldr   x1, [sp, #48]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #56]
    ldr   x0, [sp, #48]
    bl    _t_decref
    str   xzr, [sp, #48]
    ldr   x9, [sp, #56]
    str   xzr, [sp, #56]
    cbz   x9, Ltsc_err
    mov   x0, x9
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #392]                   // None
    b     Ltsc_out
Ltsc_err:
    ldr   x0, [sp, #48]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #48]
1:  ldr   x0, [sp, #56]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #56]
2:  cbz   x21, 3f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
3:  cbz   x20, 4f
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
4:  mov   x0, xzr
Ltsc_out:
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_classes(x0 = modname BORROWED) -> tuple of 2 classes
// class 0 FlashToolOutput: BASELESS (empty bases -> type meta)
//   ns: __module__, __qualname__, __init__, _usable, close, feed, finish
// class 1 FlashToolTokenStream(TokenStream)
//   ns: __module__, __qualname__, __init__, close
// x19 meta(owned)  x28 result park (zeroed at entry)
// parks: [96] cls0  [100] cls1  [128] bases  [136] name  [144] ns
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

// ===== class 0: FlashToolOutput (baseless) =====
Lcls_0:
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lcc_err
    str   x0, [sp, #128]                     // bases = ()
    ldr   x0, [sp, #128]
    bl    _calc_meta
    cbz   x0, Lcc_err
    mov   x19, x0
    adrp  x0, cn_0@PAGE
    add   x0, x0, cn_0@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, s_prepare@PAGE
    add   x1, x1, s_prepare@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lc0_prep_fallback
    str   x0, [sp, #160]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lcc_err
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
    b     Lcc_err
Lc0_prep_fallback:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Lcc_err
    str   x0, [sp, #144]
Lc0_ns_have:
    // __module__
    adrp  x0, k_module@PAGE
    add   x0, x0, k_module@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
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
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __init__ (slot 432)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #432]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // _usable (slot 440)
    adrp  x0, k_usable@PAGE
    add   x0, x0, k_usable@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #440]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // close (slot 448)
    adrp  x0, k_close@PAGE
    add   x0, x0, k_close@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #448]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // feed (slot 456)
    adrp  x0, k_feed@PAGE
    add   x0, x0, k_feed@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #456]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // finish (slot 464)
    adrp  x0, k_finish@PAGE
    add   x0, x0, k_finish@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #464]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // meta(name, bases, ns)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lcc_err
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
    b     Lcc_err
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

// ===== class 1: FlashToolTokenStream(TokenStream) =====
Lcls_1:
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lcc_err
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
    cbz   x0, Lcc_err
    mov   x19, x0
    adrp  x0, cn_1@PAGE
    add   x0, x0, cn_1@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #136]
    mov   x0, x19
    adrp  x1, s_prepare@PAGE
    add   x1, x1, s_prepare@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lc1_prep_fallback
    str   x0, [sp, #160]
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lcc_err
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
    cbnz  x9, Lc1_ns_have
    b     Lcc_err
Lc1_prep_fallback:
    bl    _t_err_clear
    bl    _t_dict_new
    cbz   x0, Lcc_err
    str   x0, [sp, #144]
Lc1_ns_have:
    // __module__
    adrp  x0, k_module@PAGE
    add   x0, x0, k_module@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
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
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    ldr   x2, [sp, #136]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // __init__ (slot 472)
    adrp  x0, k_init@PAGE
    add   x0, x0, k_init@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #472]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // close (slot 480)
    adrp  x0, k_close@PAGE
    add   x0, x0, k_close@PAGEOFF
    bl    _t_fromstring
    cbz   x0, Lcc_err
    str   x0, [sp, #176]
    ldr   x0, [sp, #144]
    ldr   x1, [sp, #176]
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x2, [x10, #480]
    bl    _t_dict_setitem
    ldr   x0, [sp, #176]
    bl    _t_decref
    str   xzr, [sp, #176]
    // meta(name, bases, ns)
    mov   x0, #3
    bl    _t_tuple_new
    cbz   x0, Lcc_err
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
    cbnz  x9, Lc1_park
    b     Lcc_err
Lc1_park:
    ldr   x0, [sp, #176]
    str   x0, [sp, #104]
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

// ---- result tuple of 2 ----
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lcc_err
    str   x0, [sp, #128]
    ldr   x0, [sp, #96]
    bl    _incref
    ldr   x0, [sp, #128]
    mov   x1, #0
    ldr   x2, [sp, #96]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #128]
    mov   x1, #1
    ldr   x2, [sp, #104]
    bl    _t_tuple_setitem
    ldr   x28, [sp, #128]                   // result park
    str   xzr, [sp, #128]

Lcc_out:
    ldr   x0, [sp, #176]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #176]
1:  ldr   x0, [sp, #152]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #152]
2:  ldr   x0, [sp, #160]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #160]
3:  ldr   x0, [sp, #144]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #144]
4:  ldr   x0, [sp, #136]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #136]
5:  ldr   x0, [sp, #128]
    cbz   x0, 6f
    bl    _t_decref
    str   xzr, [sp, #128]
6:  cbz   x19, 7f
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
7:  ldr   x0, [sp, #104]
    cbz   x0, 8f
    bl    _t_decref
    str   xzr, [sp, #104]
8:  ldr   x0, [sp, #96]
    cbz   x0, 9f
    bl    _t_decref
    str   xzr, [sp, #96]
9:  mov   x0, x28
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #224
    ret

Lcc_err:
    ldr   x0, [sp, #176]
    cbz   x0, 1f
    bl    _t_decref
    str   xzr, [sp, #176]
1:  ldr   x0, [sp, #152]
    cbz   x0, 2f
    bl    _t_decref
    str   xzr, [sp, #152]
2:  ldr   x0, [sp, #160]
    cbz   x0, 3f
    bl    _t_decref
    str   xzr, [sp, #160]
3:  ldr   x0, [sp, #144]
    cbz   x0, 4f
    bl    _t_decref
    str   xzr, [sp, #144]
4:  ldr   x0, [sp, #136]
    cbz   x0, 5f
    bl    _t_decref
    str   xzr, [sp, #136]
5:  ldr   x0, [sp, #128]
    cbz   x0, 6f
    bl    _t_decref
    str   xzr, [sp, #128]
6:  cbz   x19, 7f
    mov   x0, x19
    bl    _t_decref
    mov   x19, xzr
7:  ldr   x0, [sp, #104]
    cbz   x0, 8f
    bl    _t_decref
    str   xzr, [sp, #104]
8:  ldr   x0, [sp, #96]
    cbz   x0, 9f
    bl    _t_decref
    str   xzr, [sp, #96]
9:  mov   x0, xzr                           // NULL + pending exception
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #224
    ret

