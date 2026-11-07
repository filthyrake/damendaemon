// damendaemon/prefill.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 13/370. 100% of groups() as a RESUMABLE
// STATE MACHINE: the generator body
//     batch, assigned, count = [], [], 0
//     for prompt, slot in zip(prompts, slots):
//         if batch and count + len(prompt) > token_budget:
//             yield batch, assigned
//             batch, assigned, count = [], [], 0
//         batch.append(prompt); assigned.append(slot)
//         count += len(prompt)
//     if batch: yield batch, assigned
// becomes an iterator object whose state (it/batch/assigned/count/
// budget/prompts/slots/done) lives in instance attributes; each
// __next__ call runs one resumption of the loop in asm.
// Fidelity notes:
//  - zip(prompts, slots) is created LAZILY on first __next__ (a
//    generator function runs nothing at call time — TypeError from a
//    non-iterable prompts must surface on iteration, not on groups()).
//  - count and budget stay PyLong objects; compare/add via
//    PyNumber_Add + RichCompareBool (arbitrary precision, and TypeError
//    surfaces with CPython timing for exotic budget types).
//  - Yield hands out (batch, assigned) tuples; the lists are then
//    REPLACED with fresh ones (the original rebinds, never mutates
//    after yield).
//
// ABI table (byte offsets), installed via dd_init:
//     0 GetAttrString     8 SetAttrString    16 PyObject_Call
//    24 PyTuple_New      32 PyTuple_SetItem 40 Py_DecRef
//    48 PyList_New       56 PyLong_FromLongLong  64 PyNumber_Add
//    72 RichCompareBool  80 PyObject_Size   88 PyIter_Next
//    96 PyObject_IsTrue 104 PyErr_Occurred 112 PyErr_SetNone
//   120 PyUnicode_FromString 128 None  136 True  144 False
//   152 StopIteration   160 zip builtin   168 _GroupsIterator class

.globl _dd_init
.globl _dd_groups
.globl _dd_iter_next

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
a_prompts:  .asciz "prompts"
a_slots:    .asciz "slots"
a_budget:   .asciz "budget"
a_batch:    .asciz "batch"
a_assigned: .asciz "assigned"
a_count:    .asciz "count"
a_it:       .asciz "it"
a_done:     .asciz "done"
a_append:   .asciz "append"
a_pending:  .asciz "pending"
a_zip:      .asciz "zip"

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
_t_list_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_fromlong:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_num_add:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_rcb:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_size:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_iter_next:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_istrue:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #96]
    br    x9
_t_err_occurred:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #104]
    br    x9
_t_seterr_none:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #112]
    br    x9
_t_fromstring:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #120]
    br    x9

// helper: INCREF x0
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret

// ============================================================
// PyObject* _dd_groups(x0 = prompts, x1 = slots, x2 = token_budget)
// Constructs the iterator state object. Runs nothing that can raise
// observably (the generator body is deferred to first __next__).
// Register plan: x19 prompts  x20 slots  x21 budget (borrowed)
//   x22 inst NEW   x27 temp NEW
// ============================================================
_dd_groups:
    stp   x29, x30, [sp, #-128]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]
    stp   x25, x26, [sp, #64]
    stp   x27, x28, [sp, #80]
    stp   xzr, xzr, [sp, #96]               // args tuple / temp parks
    mov   x19, x0
    mov   x20, x1
    mov   x21, x2
    mov   x22, xzr                          // inst park

    // ---- inst = _GroupsIterator() ----
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lgr_err
    str   x0, [sp, #96]                      // () NEW
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #168]                   // _GroupsIterator class
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x22, x0                           // inst NEW
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbz   x22, Lgr_err

    // ---- setattr: prompts, slots, budget (borrowed; SetAttr does
    //      not steal) ----
    mov   x0, x22
    adrp  x1, a_prompts@PAGE
    add   x1, x1, a_prompts@PAGEOFF
    mov   x2, x19
    bl    _t_setattr_s
    cbnz  w0, Lgr_err
    mov   x0, x22
    adrp  x1, a_slots@PAGE
    add   x1, x1, a_slots@PAGEOFF
    mov   x2, x20
    bl    _t_setattr_s
    cbnz  w0, Lgr_err
    mov   x0, x22
    adrp  x1, a_budget@PAGE
    add   x1, x1, a_budget@PAGEOFF
    mov   x2, x21
    bl    _t_setattr_s
    cbnz  w0, Lgr_err

    // ---- it = None (lazy zip); done = False ----
    // (x10 is CALLER-SAVED — reload the iface pointer per use, never
    //  keep it across a call)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    mov   x0, x22
    adrp  x1, a_it@PAGE
    add   x1, x1, a_it@PAGEOFF
    ldr   x2, [x10, #128]                   // None
    bl    _t_setattr_s
    cbnz  w0, Lgr_err
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    mov   x0, x22
    adrp  x1, a_done@PAGE
    add   x1, x1, a_done@PAGEOFF
    ldr   x2, [x10, #144]                   // False
    bl    _t_setattr_s
    cbnz  w0, Lgr_err

    // ---- pending = None (in-flight pair across a yield) ----
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    mov   x0, x22
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    ldr   x2, [x10, #128]                   // None
    bl    _t_setattr_s
    cbnz  w0, Lgr_err

    // ---- batch = [] ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lgr_err
    str   x0, [sp, #96]
    mov   x0, x22
    adrp  x1, a_batch@PAGE
    add   x1, x1, a_batch@PAGEOFF
    ldr   x2, [sp, #96]
    bl    _t_setattr_s
    mov   w27, w0                          // park setattr result
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbnz  w27, Lgr_err

    // ---- assigned = [] ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lgr_err
    str   x0, [sp, #96]
    mov   x0, x22
    adrp  x1, a_assigned@PAGE
    add   x1, x1, a_assigned@PAGEOFF
    ldr   x2, [sp, #96]
    bl    _t_setattr_s
    mov   w27, w0                          // park setattr result
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbnz  w27, Lgr_err

    // ---- count = 0 ----
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lgr_err
    str   x0, [sp, #96]
    mov   x0, x22
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    ldr   x2, [sp, #96]
    bl    _t_setattr_s
    mov   w27, w0                          // park setattr result
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    cbnz  w27, Lgr_err

    mov   x0, x22                          // inst NEW
    b     Lgr_out

Lgr_err:
    cbz   x22, 1f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
1:  mov   x0, xzr

Lgr_out:
    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

// ============================================================
// PyObject* _dd_iter_next(x0 = self)  — one resumption of groups()
// Register plan:
//   x19 self (borrowed)      x20 it NEW        x21 pair NEW
//   x22 batch NEW            x23 assigned NEW  x24 count NEW
//   x25 budget NEW           x26 prompt (borrowed from pair)
//   x27 slot (borrowed from pair)
//   x28 raw flags/n (callee-saved; safe across bl)
// Frame parks (above the [80] save area):
//   [96] args tuple   [104] temp obj   [112] temp/result
// ============================================================
_dd_iter_next:
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
    mov   x20, xzr
    mov   x21, xzr
    mov   x22, xzr
    mov   x23, xzr
    mov   x24, xzr
    mov   x25, xzr

    // ---- done? ----
    mov   x0, x19
    adrp  x1, a_done@PAGE
    add   x1, x1, a_done@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    str   x0, [sp, #104]
    ldr   x0, [sp, #104]
    bl    _t_istrue
    mov   w28, w0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cmn   w28, #1
    b.eq  Lnx_err
    cmp   w28, #1
    b.eq  Lnx_stop

    // ---- it = self.it ----
    mov   x0, x19
    adrp  x1, a_it@PAGE
    add   x1, x1, a_it@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x20, x0                          // it NEW
    // lazy init: it is None -> zip(prompts, slots) NOW (not at groups()
    // time — generator bodies run on first next)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x10, [x10, #128]                 // None
    cmp   x20, x10
    b.ne  Lnx_have_it

    // ---- prompts/slots -> zip iter ----
    mov   x0, x19
    adrp  x1, a_prompts@PAGE
    add   x1, x1, a_prompts@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    str   x0, [sp, #104]                    // prompts NEW
    mov   x0, x19
    adrp  x1, a_slots@PAGE
    add   x1, x1, a_slots@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_drop104
    str   x0, [sp, #112]                    // slots NEW
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lnx_drop112
    str   x0, [sp, #96]                     // (prompts, slots) NEW
    ldr   x0, [sp, #104]
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    ldr   x2, [sp, #104]
    bl    _t_tuple_setitem
    ldr   x0, [sp, #112]
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #1
    ldr   x2, [sp, #112]
    bl    _t_tuple_setitem
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #160]                   // zip builtin
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    mov   x27, x0                           // zip iter NEW (keep)
    // drop tuple + prompts + slots (zip's TypeError, if any, is set)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    cbz   x27, Lnx_err
    // self.it = zipiter (setattr keeps its own ref; we keep ours)
    mov   x0, x19
    adrp  x1, a_it@PAGE
    add   x1, x1, a_it@PAGEOFF
    mov   x2, x27
    bl    _t_setattr_s
    mov   w28, w0
    cbnz  w28, Lnx_drop27
    // replace the stale None ref in x20 with the zip iter
    mov   x0, x20
    bl    _t_decref
    mov   x20, x27
    mov   x27, xzr
    b     Lnx_zip_ok
Lnx_drop27:
    mov   x0, x27
    bl    _t_decref
    b     Lnx_err
Lnx_zip_ok:

Lnx_have_it:
    // ---- budget = self.budget (constant across the iteration) ----
    mov   x0, x19
    adrp  x1, a_budget@PAGE
    add   x1, x1, a_budget@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x25, x0

    // ---- pending = self.pending (pair saved across a yield?) ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x21, x0                          // pending NEW (None or pair)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x10, [x10, #128]                 // None
    cmp   x21, x10
    b.eq  Lnx_no_pending
    // ---- resume: self.pending = None ----
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    mov   x2, x10                          // None (no call since load)
    bl    _t_setattr_s
    cbnz  w0, Lnx_err
    // ---- prompt/slot from the pending pair ----
    ldr   x26, [x21, #24]
    ldr   x27, [x21, #32]
    // ---- batch / assigned / count ----
    mov   x0, x19
    adrp  x1, a_batch@PAGE
    add   x1, x1, a_batch@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x22, x0
    mov   x0, x19
    adrp  x1, a_assigned@PAGE
    add   x1, x1, a_assigned@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x23, x0
    mov   x0, x19
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x24, x0
    b     Lnx_append                        // process the resumed pair
Lnx_no_pending:
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr

// ================= loop head =================
Lnx_loop:
    // ---- pair = next(it) ----
    mov   x0, x20
    bl    _t_iter_next
    mov   x21, x0                          // pair NEW or NULL
    cbnz  x21, Lnx_have_pair
    // NULL: exhausted or exception?
    bl    _t_err_occurred
    cbnz  x0, Lnx_err                      // exception set -> propagate
    b     Lnx_exhausted

// ================= have a (prompt, slot) pair =================
Lnx_have_pair:
    ldr   x26, [x21, #24]                   // prompt (borrowed)
    ldr   x27, [x21, #32]                   // slot (borrowed)
    // ---- batch / assigned / count from state ----
    mov   x0, x19
    adrp  x1, a_batch@PAGE
    add   x1, x1, a_batch@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x22, x0
    mov   x0, x19
    adrp  x1, a_assigned@PAGE
    add   x1, x1, a_assigned@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x23, x0
    mov   x0, x19
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x24, x0

    // ---- if batch and count + len(prompt) > budget ----
    mov   x0, x22
    bl    _t_istrue
    mov   w28, w0
    cmn   w28, #1
    b.eq  Lnx_err
    cbz   w28, Lnx_append
    // len(prompt)
    mov   x0, x26
    bl    _t_size
    cmp   x0, #0
    b.lt  Lnx_err
    mov   x28, x0                          // n (raw)
    // n_obj = long(n)
    mov   x0, x28
    bl    _t_fromlong
    cbz   x0, Lnx_err
    str   x0, [sp, #104]
    // total = count + n_obj   (park at [112] — x27 must survive: it
    // holds the slot for the append fallthrough)
    mov   x0, x24
    ldr   x1, [sp, #104]
    bl    _t_num_add
    str   x0, [sp, #112]                     // total NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Lnx_err
    // total > budget ?
    mov   x1, x25
    mov   x2, #4                           // Py_GT
    bl    _t_rcb
    mov   w28, w0
    ldr   x0, [sp, #112]
    bl    _t_decref
    str   xzr, [sp, #112]
    cmn   w28, #1
    b.eq  Lnx_err
    cmp   w28, #1
    b.eq  Lnx_flush

// ================= append path =================
Lnx_append:
    // ---- batch.append(prompt) ----
    mov   x0, x22
    adrp  x1, a_append@PAGE
    add   x1, x1, a_append@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    str   x0, [sp, #104]                    // append method NEW
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lnx_drop104
    str   x0, [sp, #96]
    mov   x0, x26
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x26
    bl    _t_tuple_setitem
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]                    // None result NEW (or NULL)
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Lnx_err
    bl    _t_decref
    str   xzr, [sp, #112]
    // ---- assigned.append(slot) ----
    mov   x0, x23
    adrp  x1, a_append@PAGE
    add   x1, x1, a_append@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    str   x0, [sp, #104]
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lnx_drop104
    str   x0, [sp, #96]
    mov   x0, x27
    bl    _incref
    ldr   x0, [sp, #96]
    mov   x1, #0
    mov   x2, x27
    bl    _t_tuple_setitem
    ldr   x0, [sp, #104]
    ldr   x1, [sp, #96]
    mov   x2, #0
    bl    _t_call
    str   x0, [sp, #112]
    ldr   x0, [sp, #96]
    bl    _t_decref
    str   xzr, [sp, #96]
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    ldr   x0, [sp, #112]
    cbz   x0, Lnx_err
    bl    _t_decref
    str   xzr, [sp, #112]
    // ---- count += len(prompt) ----
    mov   x0, x26
    bl    _t_size
    cmp   x0, #0
    b.lt  Lnx_err
    bl    _t_fromlong
    cbz   x0, Lnx_err
    str   x0, [sp, #104]                    // n_obj NEW
    mov   x0, x24
    ldr   x1, [sp, #104]
    bl    _t_num_add
    mov   x27, x0                           // new count NEW
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cbz   x27, Lnx_err
    // self.count = new count
    mov   x0, x19
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    mov   x2, x27
    bl    _t_setattr_s
    mov   w28, w0
    mov   x0, x27
    bl    _t_decref
    cbnz  w28, Lnx_err
    // ---- drop this iteration's refs; continue loop ----
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    b     Lnx_loop

// ================= yield flush: (batch, assigned) then reset =================
Lnx_flush:
    // ---- save the in-flight pair: the generator RESUMES WITH it ----
    // (self.pending = pair; setattr takes its own reference)
    mov   x0, x19
    adrp  x1, a_pending@PAGE
    add   x1, x1, a_pending@PAGEOFF
    mov   x2, x21
    bl    _t_setattr_s
    cbnz  w0, Lnx_err
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lnx_err
    mov   x27, x0                           // result tuple NEW
    mov   x0, x22
    bl    _incref
    mov   x0, x27
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    mov   x0, x23
    bl    _incref
    mov   x0, x27
    mov   x1, #1
    mov   x2, x23
    bl    _t_tuple_setitem
    // ---- reset: batch = [], assigned = [], count = 0 ----
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lnx_drop_result
    str   x0, [sp, #104]
    mov   x0, x19
    adrp  x1, a_batch@PAGE
    add   x1, x1, a_batch@PAGEOFF
    ldr   x2, [sp, #104]
    bl    _t_setattr_s
    mov   w28, w0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cbnz  w28, Lnx_drop_result
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lnx_drop_result
    str   x0, [sp, #104]
    mov   x0, x19
    adrp  x1, a_assigned@PAGE
    add   x1, x1, a_assigned@PAGEOFF
    ldr   x2, [sp, #104]
    bl    _t_setattr_s
    mov   w28, w0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cbnz  w28, Lnx_drop_result
    mov   x0, #0
    bl    _t_fromlong
    cbz   x0, Lnx_drop_result
    str   x0, [sp, #104]
    mov   x0, x19
    adrp  x1, a_count@PAGE
    add   x1, x1, a_count@PAGEOFF
    ldr   x2, [sp, #104]
    bl    _t_setattr_s
    mov   w28, w0
    ldr   x0, [sp, #104]
    bl    _t_decref
    str   xzr, [sp, #104]
    cbnz  w28, Lnx_drop_result
    // ---- drop iteration refs (pair stays for Lnx_out cleanup; the
    //      pending attr owns its own reference); return the tuple ----
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
    mov   x0, x27
    b     Lnx_out
Lnx_drop_result:
    mov   x0, x27
    bl    _t_decref
    b     Lnx_err

// ================= zip exhausted: final tail =================
Lnx_exhausted:
    // ---- batch = self.batch ----
    mov   x0, x19
    adrp  x1, a_batch@PAGE
    add   x1, x1, a_batch@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err
    mov   x22, x0
    // ---- if batch: yield (batch, assigned); done ----
    mov   x0, x22
    bl    _t_istrue
    mov   w28, w0
    cmn   w28, #1
    b.eq  Lnx_err_drop22
    cmp   w28, #1
    b.ne  Lnx_final_stop
    mov   x0, x19
    adrp  x1, a_assigned@PAGE
    add   x1, x1, a_assigned@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lnx_err_drop22
    mov   x23, x0
    mov   x0, #2
    bl    _t_tuple_new
    cbz   x0, Lnx_err_drop23
    mov   x27, x0                           // final tuple NEW
    mov   x0, x22
    bl    _incref
    mov   x0, x27
    mov   x1, #0
    mov   x2, x22
    bl    _t_tuple_setitem
    mov   x0, x23
    bl    _incref
    mov   x0, x27
    mov   x1, #1
    mov   x2, x23
    bl    _t_tuple_setitem
    // self.done = True (generator finished after the last yield)
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    mov   x0, x19
    adrp  x1, a_done@PAGE
    add   x1, x1, a_done@PAGEOFF
    ldr   x2, [x10, #136]                   // True
    bl    _t_setattr_s
    // drop list refs; tuple survives in x27
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
    mov   x0, x27
    b     Lnx_out

Lnx_err_drop23:
    mov   x0, x23
    bl    _t_decref
    b     Lnx_err_drop22
Lnx_err_drop22:
    mov   x0, x22
    bl    _t_decref
    b     Lnx_err

// ---- no tail batch: StopIteration ----
Lnx_final_stop:
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr

Lnx_stop:
    adrp  x9, g_iface@PAGE
    ldr   x10, [x9, g_iface@PAGEOFF]
    ldr   x0, [x10, #152]                   // StopIteration
    bl    _t_seterr_none
    mov   x0, xzr
    b     Lnx_out

// ---- error paths ----
Lnx_drop112:
    ldr   x0, [sp, #112]
    cbz   x0, Lnx_drop104
    bl    _t_decref
    str   xzr, [sp, #112]
Lnx_drop104:
    ldr   x0, [sp, #104]
    cbz   x0, Lnx_err
    bl    _t_decref
    str   xzr, [sp, #104]
    b     Lnx_err

Lnx_err:
    mov   x0, xzr

// ---- common cleanup/exit: x0 = result (NULL ok w/ exception) ----
Lnx_out:
    mov   x28, x0                          // park result across decrefs
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
3:  cbz   x25, 4f
    mov   x0, x25
    bl    _t_decref
    mov   x25, xzr
4:  cbz   x24, 5f
    mov   x0, x24
    bl    _t_decref
    mov   x24, xzr
5:  cbz   x23, 6f
    mov   x0, x23
    bl    _t_decref
    mov   x23, xzr
6:  cbz   x22, 7f
    mov   x0, x22
    bl    _t_decref
    mov   x22, xzr
7:  cbz   x21, 8f
    mov   x0, x21
    bl    _t_decref
    mov   x21, xzr
8:  cbz   x20, 9f
    mov   x0, x20
    bl    _t_decref
    mov   x20, xzr
9:  mov   x0, x28

    ldp   x27, x28, [sp, #80]
    ldp   x25, x26, [sp, #64]
    ldp   x23, x24, [sp, #48]
    ldp   x21, x22, [sp, #32]
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #128
    ret

