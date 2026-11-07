// damendaemon/token_observation.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 4/370. 100% of observe and report logic,
// including the float arithmetic (interval, mean gap, token rate)
// computed in arm64 FP registers rather than delegated.
//
// ABI table (offsets in bytes), installed via dd_init:
//     0 PyObject_GetAttrString    8 PyObject_SetAttrString
//    16 PyObject_IsTrue           24 PyLong_AsLongLong
//    32 PyLong_FromLongLong      40 PyFloat_AsDouble
//    48 PyFloat_FromDouble        56 PyDict_New
//    64 PyDict_SetItemString      72 Py_DecRef
//    80 None singleton            88 True   96 False
//   104 'server_sample_observation' str
//
// Live doubles stay in d8-d15 (callee-saved) across C calls.

.globl _dd_init
.globl _dd_observe
.globl _dd_report

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_count:     .asciz "count"
a_first:     .asciz "first"
a_last:      .asciz "last"
a_maxgap:    .asciz "max_gap"
a_seq:       .asciz "sequential"
k_scope:     .asciz "scope"
k_sampled:   .asciz "sampled_tokens"
k_firsts:    .asciz "first_sample_seconds"
k_lasts:     .asciz "last_sample_seconds"
k_seqt:      .asciz "sequential_token_timing"
k_mean:      .asciz "mean_gap_seconds"
k_maxg:      .asciz "max_gap_seconds"
k_rate:      .asciz "interval_tokens_per_second"

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
_t_setattr_s:
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
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_asdouble:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_fromdouble:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_dict_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_dict_setstr:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9

// helper: INCREF x0 (mortal objects only)
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_observe(x0=self, x1=offset, x2=sequential)
//   self.sequential = self.sequential and sequential
//   last is None ? first=offset : max_gap=max(max_gap,offset-last)
//   last=offset; count+=1     -> None
// Offset/sequential are borrowed (ctypes); parked in x20/x23.
// Float math in d8-d11 (callee-saved across C calls).
// ============================================================
_dd_observe:
    stp   x29, x30, [sp, #-96]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    // [x29,#64] parked value slot
    mov   x19, x0                          // self
    mov   x20, x1                          // offset (borrowed)
    mov   x23, x2                          // sequential (borrowed)

    // ---- self.sequential = self.sequential and sequential ----
    adrp  x1, a_seq@PAGE
    add   x1, x1, a_seq@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lobs_ret0
    mov   x21, x0                          // old sequential (owned)
    mov   x0, x21
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lobs_drop21
    b.eq  Lobs_seq_false
    mov   x0, x23
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lobs_drop21
    b.eq  Lobs_seq_false
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x22, [x9, #88]                   // True
    b     Lobs_set_seq
Lobs_seq_false:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x22, [x9, #96]                   // False
Lobs_set_seq:
    adrp  x1, a_seq@PAGE
    add   x1, x1, a_seq@PAGEOFF
    mov   x0, x19
    mov   x2, x22                          // immortal singleton, no incref
    bl    _t_setattr_s
    mov   x24, x0                          // rc (x0 dies at decref)
    mov   x0, x21
    bl    _t_decref
    cbnz  x24, Lobs_ret0

    // ---- last = self.last ----
    adrp  x1, a_last@PAGE
    add   x1, x1, a_last@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lobs_ret0
    mov   x21, x0                          // old last (owned)
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x10, [x9, #80]                   // None
    cmp   x21, x10
    b.eq  Lobs_first
    // ---- gap branch: max_gap = max(max_gap, offset - last) ----
    adrp  x1, a_maxgap@PAGE
    add   x1, x1, a_maxgap@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lobs_drop21
    mov   x22, x0                          // old max_gap (owned)
    mov   x0, x21
    bl    _t_asdouble
    fmov  d8, d0                           // last
    mov   x0, x20
    bl    _t_asdouble
    fmov  d9, d0                           // offset
    fsub  d10, d9, d8                      // gap
    mov   x0, x22
    bl    _t_asdouble
    fmov  d11, d0                          // old max_gap
    fcmp  d10, d11
    b.gt  1f
    fmov  d12, d11
    b     2f
1:  fmov  d12, d10
2:
    fmov  d0, d12
    bl    _t_fromdouble                    // new float
    cbz   x0, Lobs_drop22
    str   x0, [x29, #64]
    adrp  x1, a_maxgap@PAGE
    add   x1, x1, a_maxgap@PAGEOFF
    mov   x0, x19
    ldr   x2, [x29, #64]
    bl    _t_setattr_s
    cbnz  w0, Lobs_drop_parked_22
    mov   x0, x22
    bl    _t_decref                        // drop old max_gap
    b     Lobs_store_last
Lobs_first:
    // ---- first branch: self.first = offset ----
    // SetAttrString does not steal: it takes its own ref; pass borrowed
    str   x20, [x29, #64]
    adrp  x1, a_first@PAGE
    add   x1, x1, a_first@PAGEOFF
    mov   x0, x19
    ldr   x2, [x29, #64]
    bl    _t_setattr_s
    cbnz  w0, Lobs_drop21
Lobs_store_last:
    // ---- self.last = offset ----
    str   x20, [x29, #64]
    adrp  x1, a_last@PAGE
    add   x1, x1, a_last@PAGEOFF
    mov   x0, x19
    ldr   x2, [x29, #64]
    bl    _t_setattr_s
    cbnz  w0, Lobs_drop21
    mov   x0, x21
    bl    _t_decref                        // drop old last
    // ---- count += 1 ----
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lobs_ret0
    mov   x21, x0                          // old count (owned)
    mov   x0, x21
    bl    _t_aslong
    add   x0, x0, #1
    bl    _t_fromlong
    cbz   x0, Lobs_drop21
    str   x0, [x29, #64]
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    mov   x0, x19
    ldr   x2, [x29, #64]
    bl    _t_setattr_s
    cbnz  w0, Lobs_drop_parked_21
    mov   x0, x21
    bl    _t_decref                        // drop old count
    // return None
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #80]
    b     Lobs_ret
Lobs_drop_parked_22:
    ldr   x0, [x29, #64]
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    b     Lobs_drop21
Lobs_drop_parked_21:
    ldr   x0, [x29, #64]
    bl    _t_decref
Lobs_drop22:
    mov   x0, x22
    bl    _t_decref
Lobs_drop21:
    mov   x0, x21
    bl    _t_decref
Lobs_ret0:
    mov   x0, #0
Lobs_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #96
    ret

// ============================================================
// PyObject* _dd_report(x0=self) -> dict
//   interval = last - first if count > 1 else None
//   gaps = sequential and interval is not None
//   {scope, sampled_tokens, first/last_sample_seconds,
//    sequential_token_timing, mean_gap_seconds = interval/(count-1),
//    max_gap_seconds, interval_tokens_per_second = (count-1)/interval}
// Float math in d8-d10 (callee-saved). Getattr refs are handed to
// the dict (setitem re-adds its own) then dropped.
// ============================================================
_dd_report:
    stp   x29, x30, [sp, #-112]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    // x19 self  x20 count  x21 first  x22 last  x23 max_gap  x24 seq
    // x25 dict  [x29,#80] gaps  [x29,#88] count  [x29,#96] parked
    mov   x19, x0
    mov   x20, #0
    mov   x21, #0
    mov   x22, #0
    mov   x23, #0
    mov   x24, #0
    str   xzr, [x29, #80]
    str   xzr, [x29, #88]

    // ---- pull the five attributes ----
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lrep_err
    mov   x20, x0
    adrp  x1, a_first@PAGE
    add   x1, x1, a_first@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lrep_err
    mov   x21, x0
    adrp  x1, a_last@PAGE
    add   x1, x1, a_last@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lrep_err
    mov   x22, x0
    adrp  x1, a_maxgap@PAGE
    add   x1, x1, a_maxgap@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lrep_err
    mov   x23, x0
    adrp  x1, a_seq@PAGE
    add   x1, x1, a_seq@PAGEOFF
    mov   x0, x19
    bl    _t_getattr_s
    cbz   x0, Lrep_err
    mov   x24, x0

    // ---- count, sequential truth, interval ----
    mov   x0, x20
    bl    _t_aslong
    str   x0, [x29, #88]                   // count value
    mov   x0, x24
    bl    _t_istrue
    cmp   w0, #0
    b.lt  Lrep_err
    cset  x26, ne                          // seq truth
    // interval = last - first if count > 1 else invalid
    ldr   x9, [x29, #88]
    cmp   x9, #1
    b.le  Lrep_nointerval
    mov   x0, x22
    bl    _t_asdouble
    fmov  d8, d0                           // last
    mov   x0, x21
    bl    _t_asdouble
    fsub  d8, d8, d0                       // interval = last - first
    mov   x9, #1
    str   x9, [x29, #80]                   // interval valid
    b     Lrep_gaps
Lrep_nointerval:
    str   xzr, [x29, #80]
    fmov  d8, xzr
Lrep_gaps:
    ldr   x9, [x29, #80]
    and   x9, x9, x26
    str   x9, [x29, #80]                   // gaps = seq && interval

    // ---- dict ----
    mov   x0, #0
    bl    _t_dict_new
    cbz   x0, Lrep_err
    mov   x25, x0
    // scope
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #104]
    mov   x0, x2
    bl    _incref
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #104]
    mov   x0, x25
    adrp  x1, k_scope@PAGE
    add   x1, x1, k_scope@PAGEOFF
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // sampled_tokens = count
    ldr   x0, [x29, #88]
    bl    _t_fromlong
    cbz   x0, Lrep_err_drop_dict
    str   x0, [x29, #96]
    mov   x0, x25
    adrp  x1, k_sampled@PAGE
    add   x1, x1, k_sampled@PAGEOFF
    ldr   x2, [x29, #96]
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // first_sample_seconds
    mov   x0, x25
    adrp  x1, k_firsts@PAGE
    add   x1, x1, k_firsts@PAGEOFF
    mov   x2, x21
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // last_sample_seconds
    mov   x0, x25
    adrp  x1, k_lasts@PAGE
    add   x1, x1, k_lasts@PAGEOFF
    mov   x2, x22
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // sequential_token_timing
    mov   x0, x25
    adrp  x1, k_seqt@PAGE
    add   x1, x1, k_seqt@PAGEOFF
    mov   x2, x24
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // mean_gap_seconds = interval/(count-1) if gaps else None
    ldr   x9, [x29, #80]
    cbz   x9, Lrep_mean_none
    ldr   x9, [x29, #88]
    sub   x9, x9, #1
    scvtf d9, x9                           // (count-1) as double
    fdiv  d0, d8, d9
    bl    _t_fromdouble
    cbz   x0, Lrep_err_drop_dict
    str   x0, [x29, #96]
    mov   x0, x25
    adrp  x1, k_mean@PAGE
    add   x1, x1, k_mean@PAGEOFF
    ldr   x2, [x29, #96]
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    b     Lrep_maxgap
Lrep_mean_none:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #80]                    // None (immortal)
    mov   x0, x25
    adrp  x1, k_mean@PAGE
    add   x1, x1, k_mean@PAGEOFF
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
Lrep_maxgap:
    // max_gap_seconds = max_gap if gaps else None
    ldr   x9, [x29, #80]
    adrp  x10, g_iface@PAGE
    ldr   x10, [x10, g_iface@PAGEOFF]
    ldr   x2, [x10, #80]                   // None default
    cbnz  x9, 1f
    b     2f
1:  mov   x2, x23
2:  mov   x0, x25
    adrp  x1, k_maxg@PAGE
    add   x1, x1, k_maxg@PAGEOFF
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    // interval_tokens_per_second = (count-1)/interval if gaps and >0
    ldr   x9, [x29, #80]
    cbz   x9, Lrep_rate_none
    fcmp  d8, #0.0
    b.le  Lrep_rate_none
    ldr   x9, [x29, #88]
    sub   x9, x9, #1
    scvtf d9, x9
    fdiv  d0, d9, d8
    bl    _t_fromdouble
    cbz   x0, Lrep_err_drop_dict
    str   x0, [x29, #96]
    mov   x0, x25
    adrp  x1, k_rate@PAGE
    add   x1, x1, k_rate@PAGEOFF
    ldr   x2, [x29, #96]
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
    b     Lrep_done
Lrep_rate_none:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x2, [x9, #80]                    // None
    mov   x0, x25
    adrp  x1, k_rate@PAGE
    add   x1, x1, k_rate@PAGEOFF
    bl    _t_dict_setstr
    cbnz  w0, Lrep_err_drop_dict
Lrep_done:
    // drop the getattr refs (dict holds its own)
    mov   x0, x20
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x0, x22
    bl    _t_decref
    mov   x0, x23
    bl    _t_decref
    mov   x0, x24
    bl    _t_decref
    mov   x0, x25
    b     Lrep_ret
Lrep_err_drop_dict:
    mov   x0, x25
    bl    _t_decref
Lrep_err:
    cbz   x20, 9f
    mov   x0, x20
    bl    _t_decref
9:  cbz   x21, 9f
    mov   x0, x21
    bl    _t_decref
9:  cbz   x22, 9f
    mov   x0, x22
    bl    _t_decref
9:  cbz   x23, 9f
    mov   x0, x23
    bl    _t_decref
9:  cbz   x24, 9f
    mov   x0, x24
    bl    _t_decref
9:  mov   x0, #0
Lrep_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x25, x26, [sp, #64]
    ldp   x29, x30, [sp], #112
    ret
