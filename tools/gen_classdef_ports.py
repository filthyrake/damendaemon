#!/usr/bin/env python3
"""Generate arm64 asm class-def ports for damendaemon's pure class-statement
modules. Emits native/<mod>_asm.s (full __build_class__ semantics),
damendaemon/<mod>_asm.py (bridge), damendaemon/<mod>.py (shim).
Regenerate with: python3 tools/gen_classdef_ports.py"""
import os
"""Stamp out asm class-def ports for 14 damendaemon modules."""
import os

ROOT = '/Users/damenknight/damendaemon'

# if_mod: literal-string iface values are held by the bridge
def S(v):  return ('lit', v)     # literal (str/bool)
def N(v):  return ('name', v)    # imported name from IMPORTS

# each module: (modname, doc, imports [(module, [names])], classes [(name, [base exprs], docstr, [(key, expr)])])
# base exprs are imported names or ('prev', idx) meaning the class built earlier in this module.
MODS = [
 ('deepseek_pipeline_generation',
  '"""Explicit DeepSeek pipeline native single/batch text and tool generation."""',
  [('damendaemon.deepseek_batch_generation', ['BatchReferenceGenerator']),
   ('damendaemon.deepseek_layer_pipeline', ['DeepSeekLayerPipeline'])],
  [('DeepSeekPipelineReferenceGenerator', ['BatchReferenceGenerator'], None,
    [('model_type', N('DeepSeekLayerPipeline'))])]),

 ('queued_deepseek_host_serving',
  '"""Explicit bounded serialized native DeepSeek host HTTP admission."""',
  [('damendaemon.deepseek_host_serving', ['HostDeepSeekReferenceRuntime']),
   ('damendaemon.queued_native_host_serving', ['QueuedNativeHostReferenceRuntime'])],
  [('QueuedHostDeepSeekReferenceRuntime',
    ['QueuedNativeHostReferenceRuntime', 'HostDeepSeekReferenceRuntime'], None, [])]),

 ('queued_glm_flash_host_serving',
  '"""Explicit queued native Flash host HTTP admission."""',
  [('damendaemon.glm_flash_host_serving', ['HostFlashReferenceRuntime']),
   ('damendaemon.queued_native_host_serving', ['QueuedNativeHostReferenceRuntime'])],
  [('QueuedHostFlashReferenceRuntime',
    ['QueuedNativeHostReferenceRuntime', 'HostFlashReferenceRuntime'], None, [])]),

 ('queued_glm_host_serving',
  '"""Explicit queued native GLM host HTTP admission."""',
  [('damendaemon.glm_host_serving', ['HostGlmReferenceRuntime']),
   ('damendaemon.queued_native_host_serving', ['QueuedNativeHostReferenceRuntime'])],
  [('QueuedHostGlmReferenceRuntime',
    ['QueuedNativeHostReferenceRuntime', 'HostGlmReferenceRuntime'], None, [])]),

 ('cuda_deepseek_generation',
  '"""Explicit single-sequence prepared CUDA DeepSeek native text/tool generation."""',
  [('damendaemon.deepseek_generation', ['ReferenceGenerator']),
   ('damendaemon.cuda_deepseek_model', ['CudaDeepSeekModel'])],
  [('CudaDeepSeekGenerator', ['ReferenceGenerator'], None,
    [('model_type', N('CudaDeepSeekModel')), ('device_type', S('cuda'))])]),

 ('deepseek_host_serving',
  '"""Explicit native DeepSeek text/thinking/DSML host HTTP admission."""',
  [('damendaemon.deepseek_host_generation', ['HostDeepSeekGenerator']),
   ('damendaemon.native_host_serving', ['NativeHostReferenceRuntime'])],
  [('HostDeepSeekReferenceRuntime', ['NativeHostReferenceRuntime'], None,
    [('runner_type', N('HostDeepSeekGenerator')), ('model_type', S('deepseek_v4'))])]),

 ('glm_flash_host_serving',
  '"""Explicit Flash native host HTTP admission; text and tools only."""',
  [('damendaemon.glm_flash_host_generation', ['HostFlashGenerator']),
   ('damendaemon.native_host_serving', ['NativeHostReferenceRuntime'])],
  [('HostFlashReferenceRuntime', ['NativeHostReferenceRuntime'], None,
    [('runner_type', N('HostFlashGenerator')), ('model_type', S('glm5_next'))])]),

 ('glm_host_serving',
  '"""Explicit GLM native host HTTP admission."""',
  [('damendaemon.glm_host_generation', ['HostGlmGenerator']),
   ('damendaemon.native_host_serving', ['NativeHostReferenceRuntime'])],
  [('HostGlmReferenceRuntime', ['NativeHostReferenceRuntime'], None,
    [('runner_type', N('HostGlmGenerator')), ('model_type', S('glm_moe_dsa'))])]),

 ('glm_flash_host_image_serving',
  '"""Explicit native Flash image/tool HTTP owners for prepared CPU host stages."""',
  [('damendaemon.glm_flash_host_image_generation', ['HostFlashImageGenerator']),
   ('damendaemon.native_host_serving', ['NativeHostReferenceRuntime']),
   ('damendaemon.queued_native_host_serving', ['QueuedNativeHostReferenceRuntime'])],
  [('HostFlashImageReferenceRuntime', ['NativeHostReferenceRuntime'], None,
    [('runner_type', N('HostFlashImageGenerator')), ('model_type', S('glm5_next')),
     ('supports_images', S(True))]),
   ('QueuedHostFlashImageReferenceRuntime',
    ['QueuedNativeHostReferenceRuntime', ('prev', 0)], None, [])]),

 ('glm_flash_text',
  '"""Pinned ordinary Flash text adapter; tools and media remain unsupported.\n\nThe GLM-5.3-Flash template at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a\nhas the same ordinary-text branches and tokenizer bytes as the separately pinned\nGLM-5.3 contract. Its media branches differ. Reuse only the verified ordinary\nformat and tokenizer validation, not the media/tool branches. Checkpoint license:\ndocs/licenses/glm53-flash-MIT.txt. Official prompt/token-ID cases are pinned in\ntests/fixtures/glm53_flash_text_contract.json.\n"""',
  [('damendaemon.glm_text', ['GlmText', 'render_chat'])],
  [('FlashText', ['GlmText'],
    'Explicit Flash family binding over the verified shared ordinary format.', [])]),

 ('glm_flash_tool_generation',
  '\"\"\"Explicit Flash binding over the verified shared bounded tool lifecycle.\"\"\"',
  [('damendaemon.glm_flash_generation', ['FlashReferenceGenerator']),
   ('damendaemon.glm_flash_tool_text', ['FlashToolText']),
   ('damendaemon.glm_flash_tool_output', ['FlashToolTokenStream']),
   ('damendaemon.glm_tool_generation', ['ToolGenerationMixin'])],
  [('FlashToolReferenceGenerator', ['ToolGenerationMixin', 'FlashReferenceGenerator'], None,
    [('text_type', N('FlashToolText')), ('stream_type', N('FlashToolTokenStream'))])]),

 ('glm_flash_pipeline_generation',
  '"""Explicit native Flash text/tool generation over complete CPU layer pipelines."""',
  [('damendaemon.glm_flash_layer_pipeline', ['FlashLayerPipeline']),
   ('damendaemon.glm_flash_batch_generation', ['FlashBatchReferenceGenerator']),
   ('damendaemon.glm_flash_tool_generation', ['FlashToolReferenceGenerator'])],
  [('FlashPipelineReferenceGenerator', ['FlashBatchReferenceGenerator'], None,
    [('model_type', N('FlashLayerPipeline'))]),
   ('FlashPipelineToolReferenceGenerator', ['FlashToolReferenceGenerator'], None,
    [('model_type', N('FlashLayerPipeline'))])]),

 ('glm_flash_pipeline_image_generation',
  '"""Explicit bounded Flash image/image-tool generation over CPU layer pipelines."""',
  [('damendaemon.glm_flash_layer_pipeline', ['FlashLayerPipeline']),
   ('damendaemon.glm_flash_image_generation', ['FlashImageReferenceGenerator']),
   ('damendaemon.glm_flash_image_tool_generation', ['FlashImageToolReferenceGenerator'])],
  [('FlashPipelineImageReferenceGenerator', ['FlashImageReferenceGenerator'], None,
    [('model_type', N('FlashLayerPipeline'))]),
   ('FlashPipelineImageToolReferenceGenerator', ['FlashImageToolReferenceGenerator'], None,
    [('model_type', N('FlashLayerPipeline'))])]),

 ('glm_pipeline_generation',
  '"""Explicit prepared CPU pipeline generation; ordinary model admission is unchanged."""',
  [('damendaemon.glm_layer_pipeline', ['GlmLayerPipeline']),
   ('damendaemon.glm_batch_generation', ['BatchReferenceGenerator']),
   ('damendaemon.glm_tool_generation', ['GlmToolReferenceGenerator'])],
  [('PipelineReferenceGenerator', ['BatchReferenceGenerator'],
    'Single/equal-length/ragged text generation over one complete pipeline.',
    [('model_type', N('GlmLayerPipeline'))]),
   ('PipelineToolReferenceGenerator', ['GlmToolReferenceGenerator'],
    'Pinned GLM native-tool grammar; calls are returned, never executed.',
    [('model_type', N('GlmLayerPipeline'))])]),

 ('cuda_glm_generation',
  '"""Explicit single-sequence prepared CUDA GLM text/tool generation."""',
  [('damendaemon.glm_generation', ['ReferenceGenerator']),
   ('damendaemon.glm_tool_generation', ['ToolGenerationMixin']),
   ('damendaemon.glm_tool_text', ['GlmToolText']),
   ('damendaemon.glm_tool_output', ['GlmToolTokenStream']),
   ('damendaemon.cuda_glm_model', ['CudaGlmModel'])],
  [('CudaGlmGenerator', ['ReferenceGenerator'], None,
    [('model_type', N('CudaGlmModel')), ('device_type', S('cuda'))]),
   ('CudaGlmToolGenerator', [('prev', 0)], None,
    [('text_type', N('GlmToolText')), ('stream_type', N('GlmToolTokenStream')),
     ('generate_tools', N('ToolGenerationMixin.generate_tools'))])]),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IF = dict(getattr=0, call=8, tuple_new=16, tuple_setitem=24, decref=32,
          dict_new=40, dict_setitem=48, fromstring=56, issub=64, size=72,
          getitem=80, seterr=88, err_clear=96, type=104, true=112,
          typeerr=120, imp0=128)

P_RES = [96, 104, 112, 120]
P_BASES, P_NAME, P_NS, P_ARGS, P_PREP = 128, 136, 144, 152, 160
P_MOD, P_TMP, P_TMP2 = 168, 176, 184
FRAME = 208


def esc(s):
    out = []
    for ch in s:
        if ch == '"': out.append('\\"')
        elif ch == '\\': out.append('\\\\')
        elif ch == '\n': out.append('\\n')
        elif 32 <= ord(ch) < 127: out.append(ch)
        else: out.append('\\%03o' % ord(ch))
    return ''.join(out)


def tram(name, off):
    return (f'_t_{name}:\n    adrp  x9, g_iface@PAGE\n'
            f'    ldr   x9, [x9, g_iface@PAGEOFF]\n'
            f'    ldr   x9, [x9, #{off}]\n    br    x9\n')


def trampolines():
    out = ['''
.globl _dd_init
.globl _dd_classes

.section __DATA,__data
g_iface: .quad 0

.text
_dd_init:
    adrp  x9, g_iface@PAGE
    str   x0, [x9, g_iface@PAGEOFF]
    ret
''']
    for n, k in [('getattr_s', 'getattr'), ('call', 'call'),
                 ('tuple_new', 'tuple_new'), ('tuple_setitem', 'tuple_setitem'),
                 ('decref', 'decref'), ('dict_new', 'dict_new'),
                 ('dict_setitem', 'dict_setitem'), ('fromstring', 'fromstring'),
                 ('issub', 'issub'), ('size', 'size'),
                 ('seq_getitem', 'getitem'), ('seterr', 'seterr'),
                 ('err_clear', 'err_clear')]:
        out.append(tram(n, IF[k]))
    out.append('''
_incref:
    ldr   x9, [x0]
    add   x9, x9, #1
    str   x9, [x0]
    ret
''')
    return ''.join(out)


CALC_META = '''
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
    ldr   x0, [x10, {IF_TYPE}]
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
    ldr   x0, [x10, {IF_TYPEERR}]
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
'''.replace('{IF_TYPE}', str(IF['type'])).replace('{IF_TYPEERR}', str(IF['typeerr']))
print('gen core loaded')


def emit_class(ci, cls, imp_offsets):
    name, bases, doc, attrs = cls
    L = f'Lc{ci}_'
    ERR = 'Lerr'   # shared function-level error label
    A = []
    a = A.append
    a(f'// ===== class {name} =====')
    # bases tuple
    a(f'    mov   x0, #{len(bases)}')
    a('    bl    _t_tuple_new')
    a('    cbz   x0, ' + ERR)
    a(f'    str   x0, [sp, #{P_BASES}]')
    for bi, b in enumerate(bases):
        if isinstance(b, tuple) and b[0] == 'prev':
            src = f'[sp, #{P_RES[b[1]]}]'
            a(f'    ldr   x0, {src}')
            a('    bl    _incref')
            a(f'    ldr   x0, [sp, #{P_BASES}]')
            a(f'    mov   x1, #{bi}')
            a(f'    ldr   x2, {src}')
            a('    bl    _t_tuple_setitem')
        else:
            bname = b[1] if isinstance(b, tuple) else b
            off = imp_offsets[bname]
            a('    adrp  x9, g_iface@PAGE')
            a('    ldr   x10, [x9, g_iface@PAGEOFF]')
            a(f'    ldr   x0, [x10, #{off}]')
            a('    bl    _incref')
            a(f'    ldr   x0, [sp, #{P_BASES}]')
            a(f'    mov   x1, #{bi}')
            a(f'    ldr   x2, [x10, #{off}]')
            a('    bl    _t_tuple_setitem')
    # meta = _calc_meta(bases); x19 owned
    a(f'    ldr   x0, [sp, #{P_BASES}]')
    a('    bl    _calc_meta')
    a('    cbz   x0, ' + ERR)               # park in x19
    a('    mov   x19, x0')
    # name
    a(f'    adrp  x0, cn_{ci}@PAGE')
    a(f'    add   x0, x0, cn_{ci}@PAGEOFF')
    a('    bl    _t_fromstring')
    a('    cbz   x0, ' + ERR)
    a(f'    str   x0, [sp, #{P_NAME}]')
    # prep = getattr(meta, '__prepare__') -> ns
    a('    mov   x0, x19')
    a('    adrp  x1, s_prepare@PAGE')
    a('    add   x1, x1, s_prepare@PAGEOFF')
    a('    bl    _t_getattr_s')
    a(f'    cbz   x0, {L}prep_fallback')     # AttributeError path
    a(f'    str   x0, [sp, #{P_PREP}]')
    a('    mov   x0, #2')
    a('    bl    _t_tuple_new')
    a('    cbz   x0, ' + ERR)
    a(f'    str   x0, [sp, #{P_ARGS}]')
    a(f'    ldr   x0, [sp, #{P_NAME}]')
    a('    bl    _incref')
    a(f'    ldr   x0, [sp, #{P_ARGS}]')
    a('    mov   x1, #0')
    a(f'    ldr   x2, [sp, #{P_NAME}]')
    a('    bl    _t_tuple_setitem')
    a(f'    ldr   x0, [sp, #{P_BASES}]')
    a('    bl    _incref')
    a(f'    ldr   x0, [sp, #{P_ARGS}]')
    a('    mov   x1, #1')
    a(f'    ldr   x2, [sp, #{P_BASES}]')
    a('    bl    _t_tuple_setitem')
    a(f'    ldr   x0, [sp, #{P_PREP}]')
    a(f'    ldr   x1, [sp, #{P_ARGS}]')
    a('    mov   x2, #0')
    a('    bl    _t_call')
    a(f'    str   x0, [sp, #{P_NS}]')
    a(f'    ldr   x0, [sp, #{P_ARGS}]')
    a('    bl    _t_decref')
    a(f'    str   xzr, [sp, #{P_ARGS}]')
    a(f'    ldr   x0, [sp, #{P_PREP}]')
    a('    bl    _t_decref')
    a(f'    str   xzr, [sp, #{P_PREP}]')
    a(f'    ldr   x9, [sp, #{P_NS}]')
    a(f'    cbnz  x9, {L}ns_have')
    a('    b     ' + ERR)                    # prepare call failed
    a(f'{L}prep_fallback:')
    a('    bl    _t_err_clear')               # AttributeError cleared
    a('    bl    _t_dict_new')
    a('    cbz   x0, ' + ERR)
    a(f'    str   x0, [sp, #{P_NS}]')
    a(f'{L}ns_have:')
    # ns['__module__'] = modname; ns['__qualname__'] = name
    for key_label, val_expr in [
        ('k_module', f'[sp, #{P_MOD}]'),
        ('k_qualname', f'[sp, #{P_NAME}]'),
    ]:
        a('    adrp  x0, ' + key_label + '@PAGE')
        a('    add   x0, x0, ' + key_label + '@PAGEOFF')
        a('    bl    _t_fromstring')
        a('    cbz   x0, ' + ERR)
        a(f'    str   x0, [sp, #{P_TMP}]')
        a(f'    ldr   x0, [sp, #{P_NS}]')
        a(f'    ldr   x1, [sp, #{P_TMP}]')
        a(f'    ldr   x2, {val_expr}')
        a('    bl    _t_dict_setitem')
        a(f'    ldr   x0, [sp, #{P_TMP}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_TMP}]')
    # doc
    if doc is not None:
        a(f'    adrp  x0, cd_{ci}@PAGE')
        a(f'    add   x0, x0, cd_{ci}@PAGEOFF')
        a('    bl    _t_fromstring')
        a('    cbz   x0, ' + ERR)
        a(f'    str   x0, [sp, #{P_TMP}]')
        a('    adrp  x0, k_doc@PAGE')
        a('    add   x0, x0, k_doc@PAGEOFF')
        a('    bl    _t_fromstring')
        a('    cbz   x0, ' + ERR)
        a(f'    str   x0, [sp, #{P_TMP2}]')
        a(f'    ldr   x0, [sp, #{P_NS}]')
        a(f'    ldr   x1, [sp, #{P_TMP2}]')
        a(f'    ldr   x2, [sp, #{P_TMP}]')
        a('    bl    _t_dict_setitem')
        a(f'    ldr   x0, [sp, #{P_TMP2}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_TMP2}]')
        a(f'    ldr   x0, [sp, #{P_TMP}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_TMP}]')
    # attrs
    for ai, (k, v) in enumerate(attrs):
        key = v[1].split('.')[-1] if v[0] == 'name' and '.' in v[1] else v[1]
        a(f'    adrp  x0, ak_{ci}_{ai}@PAGE')
        a(f'    add   x0, x0, ak_{ci}_{ai}@PAGEOFF')
        a('    bl    _t_fromstring')
        a(f'    cbz   x0, {ERR}')
        a(f'    str   x0, [sp, #{P_TMP}]')  # key NEW
        if v[0] == 'name':
            off = imp_offsets[key]
            a('    adrp  x9, g_iface@PAGE')
            a('    ldr   x10, [x9, g_iface@PAGEOFF]')
            a(f'    ldr   x0, [sp, #{P_NS}]')
            a(f'    ldr   x1, [sp, #{P_TMP}]')
            a(f'    ldr   x2, [x10, #{off}]')
            a('    bl    _t_dict_setitem')
        elif v[0] == 'lit' and v[1] is True:
            a('    adrp  x9, g_iface@PAGE')
            a('    ldr   x10, [x9, g_iface@PAGEOFF]')
            a(f'    ldr   x0, [sp, #{P_NS}]')
            a(f'    ldr   x1, [sp, #{P_TMP}]')
            a(f'    ldr   x2, [x10, #{IF["true"]}]')
            a('    bl    _t_dict_setitem')
        else:  # string literal
            a(f'    adrp  x0, av_{ci}_{ai}@PAGE')
            a(f'    add   x0, x0, av_{ci}_{ai}@PAGEOFF')
            a('    bl    _t_fromstring')
            a('    cbz   x0, ' + ERR)
            a(f'    str   x0, [sp, #{P_TMP2}]')
            a(f'    ldr   x0, [sp, #{P_NS}]')
            a(f'    ldr   x1, [sp, #{P_TMP}]')
            a(f'    ldr   x2, [sp, #{P_TMP2}]')
            a('    bl    _t_dict_setitem')
            a(f'    ldr   x0, [sp, #{P_TMP2}]')
            a('    bl    _t_decref')
            a(f'    str   xzr, [sp, #{P_TMP2}]')
        a(f'    ldr   x0, [sp, #{P_TMP}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_TMP}]')
    # cls_obj = meta(name, bases, ns)
    a('    mov   x0, #3')
    a('    bl    _t_tuple_new')
    a('    cbz   x0, ' + ERR)
    a(f'    str   x0, [sp, #{P_ARGS}]')
    for idx, src in enumerate([f'[sp, #{P_NAME}]', f'[sp, #{P_BASES}]', f'[sp, #{P_NS}]']):
        a(f'    ldr   x0, {src}')
        a('    bl    _incref')
        a(f'    ldr   x0, [sp, #{P_ARGS}]')
        a(f'    mov   x1, #{idx}')
        a(f'    ldr   x2, {src}')
        a('    bl    _t_tuple_setitem')
    a('    mov   x0, x19')
    a(f'    ldr   x1, [sp, #{P_ARGS}]')
    a('    mov   x2, #0')
    a('    bl    _t_call')
    a(f'    str   x0, [sp, #{P_TMP}]')  # result NEW or NULL
    a(f'    ldr   x0, [sp, #{P_ARGS}]')
    a('    bl    _t_decref')
    a(f'    str   xzr, [sp, #{P_ARGS}]')
    a(f'    ldr   x9, [sp, #{P_TMP}]')
    a(f'    cbnz  x9, {L}park')
    a('    b     ' + ERR)
    a(f'{L}park:')
    a(f'    ldr   x0, [sp, #{P_TMP}]')
    a(f'    str   x0, [sp, #{P_RES[ci]}]')
    a(f'    str   xzr, [sp, #{P_TMP}]')
    # drop ns, name, bases, meta
    for p in (P_NS, P_NAME, P_BASES):
        a(f'    ldr   x0, [sp, #{p}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{p}]')
    a('    mov   x0, x19')
    a('    bl    _t_decref')
    a('    mov   x19, xzr')
    return A


def dd_classes_body(nclasses):
    A = []
    a = A.append
    a(f'// ============================================================')
    a(f'// _dd_classes(x0 = modname BORROWED) -> tuple of N classes')
    a(f'// x19 meta(owned, per-class)   x28 result park across cleanup')
    a(f'// ============================================================')
    a('_dd_classes:')
    a(f'    stp   x29, x30, [sp, #-{FRAME + 16}]!')
    a('    mov   x29, sp')
    a('    stp   x19, x20, [sp, #16]')
    a('    stp   x21, x22, [sp, #32]')
    a('    stp   x23, x24, [sp, #48]')
    a('    stp   x25, x26, [sp, #64]')
    a('    stp   x27, x28, [sp, #80]')
    for off in P_RES:
        a(f'    str   xzr, [sp, #{off}]')
    for p in (P_BASES, P_NAME, P_NS, P_ARGS, P_PREP, P_TMP, P_TMP2):
        a(f'    str   xzr, [sp, #{p}]')
    a('    mov   x19, xzr')
    a(f'    str   x0, [sp, #{P_MOD}]')          # modname borrowed
    a(f'    adrp  x9, g_iface@PAGE')
    a(f'    ldr   x10, [x9, g_iface@PAGEOFF]')
    a('// ---- per-class construction ----')
    for ci in range(nclasses):
        a(f'Lcls_{ci}:')
        A.extend('    ' + ln if ln else ln for ln in [''])
        # NOTE: caller splices emit_class output here
    a('// ---- result tuple ----')
    a(f'    mov   x0, #{nclasses}')
    a('    bl    _t_tuple_new')
    a('    cbz   x0, Lerr')
    a(f'    str   x0, [sp, #{P_TMP}]')
    for ci in range(nclasses):
        a(f'    ldr   x0, [sp, #{P_RES[ci]}]')
        a('    bl    _incref')
        a(f'    ldr   x0, [sp, #{P_TMP}]')
        a(f'    mov   x1, #{ci}')
        a(f'    ldr   x2, [sp, #{P_RES[ci]}]')
        a('    bl    _t_tuple_setitem')
    for ci in range(nclasses):
        a(f'    ldr   x0, [sp, #{P_RES[ci]}]')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_RES[ci]}]')
    a(f'    ldr   x0, [sp, #{P_TMP}]')          # result tuple
    a(f'    str   xzr, [sp, #{P_TMP}]')
    a('    b     Lexit')
    a('Lerr:')
    a('    mov   x0, xzr')
    a('Lexit:')
    a('    mov   x28, x0')                      # park result
    for p in (P_TMP, P_TMP2, P_PREP, P_ARGS, P_NS, P_NAME, P_BASES):
        a(f'    ldr   x0, [sp, #{p}]')
        a('    cbz   x0, 1f')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{p}]')
        a('1:')
    for ci in range(nclasses):
        a(f'    ldr   x0, [sp, #{P_RES[ci]}]')
        a('    cbz   x0, 2f')
        a('    bl    _t_decref')
        a(f'    str   xzr, [sp, #{P_RES[ci]}]')
        a('2:')
    a('    cbz   x19, 3f')
    a('    mov   x0, x19')
    a('    bl    _t_decref')
    a('    mov   x19, xzr')
    a('3:  mov   x0, x28')
    a(f'    ldp   x27, x28, [sp, #80]')
    a('    ldp   x25, x26, [sp, #64]')
    a('    ldp   x23, x24, [sp, #48]')
    a('    ldp   x21, x22, [sp, #32]')
    a('    ldp   x19, x20, [sp, #16]')
    a(f'    ldp   x29, x30, [sp], #{FRAME + 16}')
    a('    ret')
    return A


def emit_s(modname, mod, imports, classes):
    # import order: flat list of module-imported names, then derived attrs
    names = [n for _, ns in imports for n in ns]
    extra = []
    for _, _, _, attrs in classes:
        for k, v in attrs:
            if v[0] == 'name' and '.' in v[1]:
                holder, attr = v[1].split('.')
                if attr not in names:
                    extra.append((holder, attr))
    imp_names = names + [attr for _, attr in extra]
    imp_offsets = {}
    for i, n in enumerate(imp_names):
        imp_offsets[n] = IF['imp0'] + 8 * i
    niface = IF['imp0'] // 8 + len(imp_names)

    hdr = ('// GENERATED by tools/gen_classdef_ports.py — do not edit by hand.\n'
           f'// damendaemon/{modname}.py — arm64 asm port (class-def module).\n'
           '// 100% of the module logic: the class statements, realized as\n'
           '// full __build_class__ semantics (metaclass calculation via\n'
           '// PyType_IsSubtype + conflict TypeError, __prepare__ with\n'
           '// AttributeError fallback, ordered namespace population, then\n'
           '// meta(name, bases, ns)).\n')
    out = [hdr, trampolines(), CALC_META]
    # class bodies spliced into dd_classes
    body = dd_classes_body(len(classes))
    # strip splice marker lines, find insertion point after Lcls_{ci}:
    text_lines = body
    final = []
    ci_iter = iter(range(len(classes)))
    splice_at = {f'Lcls_{i}': i for i in range(len(classes))}
    for ln in text_lines:
        final.append(ln)
        if ln.startswith('Lcls_'):
            idx = int(ln.split('_')[1].rstrip(':'))
            final.extend(emit_class(idx, classes[idx], imp_offsets))
    out.append('\n'.join(final) + '\n')

    # cstring section
    cs = ['\n.section __TEXT,__cstring']
    cs.append('s_prepare:  .asciz "__prepare__"')
    cs.append('s_conflict: .asciz "metaclass conflict: the metaclass of a '
              'derived class must be a (non-strict) subclass of the '
              'metaclasses of all its bases"')
    cs.append('k_module:   .asciz "__module__"')
    cs.append('k_qualname: .asciz "__qualname__"')
    cs.append('k_doc:      .asciz "__doc__"')
    for i, (name, bases, doc, attrs) in enumerate(classes):
        cs.append(f'cn_{i}: .asciz "{esc(name)}"')
        if doc is not None:
            cs.append(f'cd_{i}: .asciz "{esc(doc)}"')
        for ai, (k, v) in enumerate(attrs):
            cs.append(f'ak_{i}_{ai}: .asciz "{esc(k)}"')
            if v[0] == 'lit' and isinstance(v[1], str):
                cs.append(f'av_{i}_{ai}: .asciz "{esc(v[1])}"')
    out.append('\n'.join(cs) + '\n')
    return ''.join(out), imp_names, niface


BRIDGE = '''"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch (class-def module): every class statement lives in
native/{modname}_asm.s as full __build_class__ semantics — metaclass
calculation (PyType_IsSubtype winner selection, conflict TypeError),
__prepare__ with AttributeError fallback, ordered namespace population
(__module__/__qualname__/__doc__/attrs), then meta(name, bases, ns).
This bridge only compiles/caches the dylib, resolves the interface
table (C API + imported names), and binds the resulting classes.
"""
import ctypes
import os
import subprocess
import sys

{imports}

_HOLD = ({holds},)  # keep iface entries alive


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', '{modname}_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'lib{modname}_asm.dylib')
    os.makedirs(cache, exist_ok=True)
    if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(source):
        subprocess.run([os.environ.get('CC', 'clang'), '-arch', 'arm64', '-shared', '-O2',
                        '-mmacosx-version-min=11.0', '-undefined', 'dynamic_lookup',
                        '-o', target, source], check=True)
    return target


def _address(dll, name):
    return ctypes.cast(getattr(dll, name), ctypes.c_void_p).value


_lib = ctypes.PyDLL(_build())
_lib.dd_init.argtypes = [ctypes.c_void_p]
_lib.dd_classes.restype = ctypes.py_object
_lib.dd_classes.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

_iface = (ctypes.c_void_p * {niface})(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyUnicode_FromString'),
    _address(_python, 'PyType_IsSubtype'),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PySequence_GetItem'),
    _address(_python, 'PyErr_SetString'),
    _address(_python, 'PyErr_Clear'),
    id(type),
    id(True),
    id(TypeError),
{iface_names}
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
{binds}
'''


def emit_bridge(modname, imports, classes, imp_names):
    # imports section (relative for damendaemon-internal modules)
    imps = []
    for mod, ns in imports:
        rel = '.' + mod.split('.', 1)[1] if mod.startswith('damendaemon.') else mod
        imps.append(f'from {rel} import ' + ', '.join(ns))
    # derived attr values (e.g. ToolGenerationMixin.generate_tools)
    derived = []
    for _, _, _, attrs in classes:
        for k, v in attrs:
            if v[0] == 'name' and '.' in v[1]:
                holder, attr = v[1].split('.')
                derived.append((holder, attr))
    for holder, attr in derived:
        imps.append(f'_derived_{attr} = getattr({holder}, "{attr}")')
    # _HOLD order must match imp_names: flat names, then derived attrs
    flat = [n for _, ns in imports for n in ns]
    hold_exprs = flat + [f'_derived_{attr}' for _, attr in derived]
    holds = ', '.join(hold_exprs)
    iface_names = '\n'.join(f'    id({e}),' for e in hold_exprs)
    binds = '\n'.join(f'{name} = _classes[{i}]' for i, (name, _, _, _) in enumerate(classes))
    txt = BRIDGE.format(modname=modname, imports='\n'.join(imps), holds=holds,
                        niface=len(hold_exprs) + 16, iface_names=iface_names,
                        binds=binds)
    return txt


def emit_shim(modname, mod_doc, imports, classes):
    names = [n for _, ns in imports for n in ns]
    cnames = [c[0] for c in classes]
    allnames = cnames + names
    lines = [mod_doc, '', f'from .{modname}_asm import (' + ', '.join(allnames) + ')', '']
    return '\n'.join(lines) + '\n' if mod_doc.endswith('"""') else '\n'.join(lines)


def main():
    import subprocess as sp
    for spec in MODS:
        modname, mod_doc, imports, classes = spec
        s_text, imp_names, niface = emit_s(modname, mod_doc, imports, classes)
        open(os.path.join(ROOT, 'native', f'{modname}_asm.s'), 'w').write(s_text)
        bridge = emit_bridge(modname, imports, classes, imp_names)
        open(os.path.join(ROOT, 'damendaemon', f'{modname}_asm.py'), 'w').write(bridge)
        shim = emit_shim(modname, mod_doc, imports, classes)
        open(os.path.join(ROOT, 'damendaemon', f'{modname}.py'), 'w').write(shim)
        r = sp.run(['clang', '-arch', 'arm64', '-c', '-o', '/tmp/genc.o',
                    os.path.join(ROOT, 'native', f'{modname}_asm.s')],
                   capture_output=True, text=True)
        status = 'OK' if r.returncode == 0 else 'FAIL'
        print(f'{status} {modname}' + ('' if r.returncode == 0 else '\n' + r.stderr[:500]))
    print('done')

if __name__ == '__main__':
    main()



