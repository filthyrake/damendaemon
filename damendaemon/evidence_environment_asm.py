"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 43 — report_environment (the dict comprehension
over sorted(REPORT_KEYS)) lives in native/evidence_environment_asm.s.
This bridge only compiles/caches the dylib, resolves the interface table,
holds the module-level data (REPORT_KEYS, sorted builtin), and supplies
the logic-free protocol shell.
"""
import ctypes
import os
import subprocess
import sys

REPORT_KEYS = frozenset('''
DAMENDAEMON_DEVICE DAMENDAEMON_KERNELS DAMENDAEMON_QUANTIZATION
DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE
DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4 DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT
DAMENDAEMON_EXPERIMENTAL_VISION DAMENDAEMON_EXPERIMENTAL_XPU
DAMENDAEMON_GEMMA_WEIGHT_DTYPE DAMENDAEMON_GPT_OSS_ATTENTION DAMENDAEMON_GPT_OSS_PREFILL
DAMENDAEMON_HYBRID_METAL_SCAN DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB
DAMENDAEMON_HYBRID_PRECISION DAMENDAEMON_HYBRID_PREFILL
DAMENDAEMON_KQUANT_DENSE_CACHE_MIB DAMENDAEMON_KQUANT_PREFILL_F32 DAMENDAEMON_KV_CACHE_TOKENS
DAMENDAEMON_LLAMA_PRECISION DAMENDAEMON_LOGPROBS_KERNELS
DAMENDAEMON_MAMBA2_COMPENSATED DAMENDAEMON_MAMBA2_SSD DAMENDAEMON_MAMBA_FUSED_DT
DAMENDAEMON_METAL_WEIGHT_DTYPE DAMENDAEMON_PARTITION_PACKED_PREFILL DAMENDAEMON_PARTITION_REBIND
DAMENDAEMON_PREFILL_GRAPHS DAMENDAEMON_PREFILL_GRAPH_MAX_TOKENS
DAMENDAEMON_Q4_STAGE_ADJACENT DAMENDAEMON_Q4_STAGE_RUNTIME_ROPE DAMENDAEMON_Q4_STAGE_RUNTIME_ROWS
DAMENDAEMON_QWEN3_PRECISION DAMENDAEMON_ROCM_MEMORY_GIB DAMENDAEMON_SPECULATIVE_TOKENS
DAMENDAEMON_T5_PREFILL_GRAPHS
'''.split())

_HOLD = (REPORT_KEYS, sorted)


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'evidence_environment_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libevidence_environment_asm.dylib')
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
_lib.dd_report.restype = ctypes.py_object
_lib.dd_report.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _report_environment(environment):
    """Protocol shell — logic in native/evidence_environment_asm.s."""
    return _lib.dd_report(environment)


# ABI table shared with native/evidence_environment_asm.s — byte offset =
# index*8, exactly as listed in the .s header (12 entries, 0..88):
_iface = (ctypes.c_void_p * 12)(
    _address(_python, 'PyObject_Call'),                 # 0
    _address(_python, 'PyDict_New'),                    # 8
    _address(_python, 'PyDict_SetItem'),                # 16
    _address(_python, 'Py_DecRef'),                      # 24
    _address(_python, 'PyObject_GetIter'),              # 32
    _address(_python, 'PyIter_Next'),                    # 40
    _address(_python, 'PyErr_Occurred'),                # 48
    id(sorted),                                          # 56
    id(REPORT_KEYS),                                     # 64
    _address(_python, 'PyDict_GetItemWithError'),        # 72
    _address(_python, 'PyTuple_SetItem'),                # 80
    _address(_python, 'PyTuple_New'),                    # 88
)
_lib.dd_init(ctypes.byref(_iface))
