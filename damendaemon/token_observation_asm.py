"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 4: observe and report logic (including all
float arithmetic — gap tracking, mean gap, token rate) lives in
native/token_observation_asm.s. The dataclass field declarations below
are declarative state (defaults), matching the bridge-signature
pattern; all method logic is assembly. GIL held via PyDLL.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'token_observation_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libtoken_observation_asm.dylib')
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
_lib.dd_observe.restype = ctypes.py_object
_lib.dd_observe.argtypes = [ctypes.py_object] * 3
_lib.dd_report.restype = ctypes.py_object
_lib.dd_report.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/token_observation_asm.s (offsets/8):
#  0 GetAttrString   8 SetAttrString  16 PyObject_IsTrue
# 24 PyLong_AsLongLong  32 PyLong_FromLongLong  40 PyFloat_AsDouble
# 48 PyFloat_FromDouble  56 PyDict_New  64 PyDict_SetItemString
# 72 Py_DecRef  80 None  88 True  96 False  104 scope string
_iface = (ctypes.c_void_p * 14)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_SetAttrString'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyLong_AsLongLong'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyFloat_AsDouble'),
    _address(_python, 'PyFloat_FromDouble'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItemString'),
    _address(_python, 'Py_DecRef'),
    id(None),
    id(True),
    id(False),
    id('server_sample_observation'),
)
_lib.dd_init(ctypes.byref(_iface))


def observe(self, offset, *, sequential=True):
    return _lib.dd_observe(self, offset, sequential)


def report(self):
    return _lib.dd_report(self)
