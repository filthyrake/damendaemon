"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch (class-def module): every class statement lives in
native/deepseek_host_serving_asm.s as full __build_class__ semantics — metaclass
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

from .deepseek_host_generation import HostDeepSeekGenerator
from .native_host_serving import NativeHostReferenceRuntime

_HOLD = (HostDeepSeekGenerator, NativeHostReferenceRuntime,)  # keep iface entries alive


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'deepseek_host_serving_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libdeepseek_host_serving_asm.dylib')
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

_iface = (ctypes.c_void_p * 18)(
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
    id(HostDeepSeekGenerator),
    id(NativeHostReferenceRuntime),
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
HostDeepSeekReferenceRuntime = _classes[0]
