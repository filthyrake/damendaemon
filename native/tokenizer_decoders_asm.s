// damendaemon/tokenizer_decoders.py — hand-ported to arm64 assembly.
// asm-rewrite branch, module 2/370. 100% of decoder_steps and
// is_bytelevel logic; the Python bridge only builds the interface
// table and binds entry points (damendaemon/tokenizer_decoders_asm.py).
//
// ABI table (offsets in bytes), installed via dd_init:
//    0 json.loads obj    8 PyObject_GetAttrString  16 PyObject_Call
//   24 PyList_New       32 PyList_Append           40 PyList_GetItem
//   48 PyList_SetSlice   56 PyTuple_New            64 PyTuple_SetItem
//   72 PyDict_GetItemWithError  80 PyUnicode_CompareWithASCIIString
//   88 PyObject_GetItem  96 Py_DecRef              104 None singleton
//  112 key 'type'       120 key 'decoders'         128 True   136 False
//
// Reference protocol: py_object restype converts the returned pointer
// without stealing, so every success path returns a new reference and
// singletons are INCREF'd before returning.

.globl _dd_init
.globl _dd_decoder_steps
.globl _dd_is_bytelevel

.section __DATA,__data
g_iface: .quad 0

.section __TEXT,__cstring
c_getstate:  .asciz "__getstate__"
c_sequence:  .asciz "Sequence"
c_bytelevel: .asciz "ByteLevel"

.text

_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret

// tail-call trampolines into the interface table
_t_getattr_s:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #8]
    br    x9
_t_call:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #16]
    br    x9
_t_list_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #24]
    br    x9
_t_list_append:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #32]
    br    x9
_t_list_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #40]
    br    x9
_t_list_setslice:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #48]
    br    x9
_t_tuple_new:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #56]
    br    x9
_t_tuple_setitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #64]
    br    x9
_t_dict_getwe:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #72]
    br    x9
_t_cmp_ascii:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #80]
    br    x9
_t_obj_getitem:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #88]
    br    x9
_t_decref:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x9, [x9, #96]
    br    x9

// ============================================================
// PyObject* _dd_decoder_steps(x0 = decoder) -> list of step dicts
//   None -> []
//   pending = [json.loads(decoder.__getstate__())]
//   pop; type == 'Sequence' -> extend(reversed(decoders));
//   otherwise -> steps.append(step)
// Live state stays in callee-saved x19-x22 across all calls.
//   x19 steps   x20 pending   x21 owned temp   x22 owned temp
// ============================================================
_dd_decoder_steps:
    stp   x29, x30, [sp, #-64]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    stp   x21, x22, [sp, #32]
    stp   x23, x24, [sp, #48]

    // decoder is None -> fresh empty list
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x10, [x9, #104]               // None singleton
    cmp   x0, x10
    b.ne  1f
    mov   x0, #0
    bl    _t_list_new
    b     Lds_ret
1:
    // method = decoder.__getstate__  (owned)
    // PyObject_GetAttrString(v, name): x0 = decoder (unchanged), x1 = name
    adrp  x1, c_getstate@PAGE
    add   x1, x1, c_getstate@PAGEOFF
    bl    _t_getattr_s
    cbz   x0, Lds_err0                  // AttributeError propagates
    mov   x21, x0
    // x22 = empty args tuple (owned)
    mov   x0, #0
    bl    _t_tuple_new
    cbz   x0, Lds_drop_meth
    mov   x22, x0
    // x21 = state string = method()  (owned; reuses x21)
    mov   x0, x21
    mov   x1, x22
    mov   x2, #0
    bl    _t_call
    mov   x20, x0                       // x20 = state (saved BEFORE decrefs)
    mov   x0, x22
    bl    _t_decref                     // drop empty tuple
    mov   x0, x21
    bl    _t_decref                     // drop method
    cbz   x20, Lds_err0
    mov   x21, x20                      // x21 = state (owned)
    // args = (state,)  -> x22 (owned)
    mov   x0, #1
    bl    _t_tuple_new
    cbz   x0, Lds_drop_state
    mov   x22, x0
    mov   x0, x22                       // tuple
    mov   x1, #0                        // pos
    mov   x2, x21                       // item (stolen on success)
    bl    _t_tuple_setitem
    // x21 = json.loads(state)  (owned)
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #0]                  // json.loads callable
    mov   x1, x22                       // args
    mov   x2, #0
    bl    _t_call
    mov   x21, x0                       // x21 = step0 (owned)
    mov   x0, x22
    bl    _t_decref                      // drop args tuple
    cbz   x21, Lds_err0
    // pending = []; append(step0)
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lds_drop_step
    mov   x20, x0                       // x20 = pending
    mov   x0, x20
    mov   x1, x21
    bl    _t_list_append
    cbnz  w0, Lds_drop_both
    mov   x0, x21
    bl    _t_decref                      // pending owns step0 now
    mov   x0, #0
    bl    _t_list_new
    cbz   x0, Lds_drop_pending          // leak steps path; propagate
    mov   x19, x0                       // x19 = steps
    // ---------------- main loop ----------------
Lds_loop:
    ldr   x10, [x20, #16]               // n = len(pending)
    cbz   x10, Lds_done
    sub   x10, x10, #1                  // last
    // x21 = step = pending.pop()  (owned)
    mov   x0, x20
    mov   x1, x10
    bl    _t_list_getitem
    cbz   x0, Lds_ret
    mov   x21, x0
    ldr   x9, [x21]                      // INCREF
    add   x9, x9, #1
    str   x9, [x21]
    // del pending[last:n]  ==  pending.pop()
    mov   x0, x20                       // list
    mov   x1, x10                       // ilow = last
    add   x2, x10, #1                   // ihigh = n
    mov   x3, #0                        // v = NULL -> delete slice
    bl    _t_list_setslice
    cbnz  w0, Lds_drop_step
    // t = step.get('type')  (borrowed or NULL)
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #112]                // 'type' key
    mov   x0, x21
    bl    _t_dict_getwe
    cbz   x0, Lds_plain_step            // missing -> not a Sequence
    mov   x22, x0                       // x22 = t (borrowed)
    mov   x0, x22
    adrp  x1, c_sequence@PAGE
    add   x1, x1, c_sequence@PAGEOFF
    bl    _t_cmp_ascii
    cbnz  w0, Lds_plain_step            // != 'Sequence'
    // -------- Sequence branch --------
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #120]                // 'decoders' key
    mov   x0, x21
    bl    _t_obj_getitem                 // raises KeyError like ['decoders']
    cbz   x0, Lds_drop_step              // propagate, drop step
    mov   x22, x0                       // x22 = decoders (owned)
    ldr   x23, [x22, #16]               // m = len(decoders)
Lds_extend:
    cbz   x23, Lds_extend_done
    sub   x23, x23, #1                  // i = m-1 downto 0
    mov   x0, x22
    mov   x1, x23
    bl    _t_list_getitem               // borrowed
    cbz   x0, Lds_drop_dec_step
    mov   x1, x0
    mov   x0, x20
    bl    _t_list_append                 // pending.append(d[i])
    cbnz  w0, Lds_drop_dec_step          // propagate
    b     Lds_extend
Lds_extend_done:
    mov   x0, x22
    bl    _t_decref                      // drop decoders
    mov   x0, x21
    bl    _t_decref                      // step discarded
    b     Lds_loop
Lds_drop_dec_step:
    mov   x0, x22
    bl    _t_decref
    mov   x0, x21
    bl    _t_decref
    mov   x19, #0
    b     Lds_ret_null
Lds_drop_step:
    mov   x0, x21
    bl    _t_decref
    mov   x19, #0
    b     Lds_ret_null
Lds_drop_both:
    mov   x0, x21
    bl    _t_decref
Lds_drop_pending:
    mov   x0, x20
    bl    _t_decref
    mov   x19, #0
    b     Lds_ret_null
Lds_err0:
    mov   x19, #0
    b     Lds_ret_null
    // -------- plain step branch --------
Lds_plain_step:
    mov   x0, x19
    mov   x1, x21
    bl    _t_list_append                 // steps.append(step)
    cbnz  w0, Lds_drop_step              // propagate
    mov   x0, x21
    bl    _t_decref                      // drop our ref
    b     Lds_loop
Lds_done:
    mov   x0, x19
    b     Lds_ret
Lds_drop_state:
    mov   x0, x21
    bl    _t_decref
    mov   x19, #0
    b     Lds_ret_null
Lds_drop_meth:
    mov   x0, x21
    bl    _t_decref
    mov   x19, #0
    b     Lds_ret_null
Lds_ret_null:
    mov   x0, x19
Lds_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x21, x22, [sp, #32]
    ldp   x23, x24, [sp, #48]
    ldp   x29, x30, [sp], #64
    ret

// ============================================================
// PyObject* _dd_is_bytelevel(x0 = decoder) -> True / False
//   steps = decoder_steps(decoder)
//   len(steps) == 1 and steps[0].get('type') == 'ByteLevel'
// Returns INCREF'd True/False singletons.
// ============================================================
_dd_is_bytelevel:
    stp   x29, x30, [sp, #-48]!
    mov   x29, sp
    stp   x19, x20, [sp, #16]
    // x19 steps   x20 t
    bl    _dd_decoder_steps
    cbz   x0, Lbb_ret                   // propagate (x19 still 0)
    mov   x19, x0
    ldr   x10, [x19, #16]               // len
    cmp   x10, #1
    b.ne  Lbb_false
    mov   x0, x19
    mov   x1, #0
    bl    _t_list_getitem               // steps[0] borrowed
    cbz   x0, Lbb_false
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x1, [x9, #112]                // 'type' key
    bl    _t_dict_getwe
    cbz   x0, Lbb_false                 // .get -> None -> False
    mov   x20, x0
    mov   x0, x20
    adrp  x1, c_bytelevel@PAGE
    add   x1, x1, c_bytelevel@PAGEOFF
    bl    _t_cmp_ascii
    cbnz  w0, Lbb_false                 // != 'ByteLevel'
    b     Lbb_true
Lbb_true:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #128]                // True
    b     Lbb_finish
Lbb_false:
    adrp  x9, g_iface@PAGE
    ldr   x9, [x9, g_iface@PAGEOFF]
    ldr   x0, [x9, #136]                // False
Lbb_finish:
    mov   x20, x0                       // result: immortal singleton,
    mov   x0, x19                       // returned without a ref change
    bl    _t_decref                     // drop steps
    mov   x0, x20
    b     Lbb_ret
Lbb_ret:
    ldp   x19, x20, [sp, #16]
    ldp   x29, x30, [sp], #48
    ret
