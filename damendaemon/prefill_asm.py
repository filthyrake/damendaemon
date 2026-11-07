"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 13: groups() logic lives in
native/prefill_asm.s as a RESUMABLE STATE MACHINE — the generator body
(zip pull, budget compare, yield-and-reset, exhaustion tail) runs one
resumption per __next__ call, with all state in instance attributes and
lazy zip construction so a non-iterable `prompts` raises on first
iteration, exactly like a generator function. This bridge only
compiles/caches the dylib, resolves the interface table, and supplies
the protocol shell (__iter__/__next__) around the asm entry point.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'prefill_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libprefill_asm.dylib')
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
_lib.dd_groups.restype = ctypes.py_object
_lib.dd_groups.argtypes = [ctypes.py_object] * 3
_lib.dd_iter_next.restype = ctypes.py_object
_lib.dd_iter_next.argtypes = [ctypes.py_object]


class _GroupsIterator:
    """Protocol shell; all decision logic lives in assembly."""

    def __iter__(self):
        return self

    def __next__(self):
        return _lib.dd_iter_next(self)


_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/prefill_asm.s (offsets/8):
#  0 GetAttrString  1 SetAttrString  2 PyObject_Call  3 PyTuple_New
#  4 PyTuple_SetItem  5 Py_DecRef  6 PyList_New  7 PyLong_FromLongLong
#  8 PyNumber_Add  9 PyObject_RichCompareBool  10 PyObject_Size
# 11 PyIter_Next  12 PyObject_IsTrue  13 PyErr_Occurred  14 PyErr_SetNone
# 15 PyUnicode_FromString  16 None  17 True  18 False  19 StopIteration
# 20 zip  21 _GroupsIterator
_iface = (ctypes.c_void_p * 22)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_SetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyList_New'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyNumber_Add'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PyIter_Next'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyErr_Occurred'),
    _address(_python, 'PyErr_SetNone'),
    _address(_python, 'PyUnicode_FromString'),
    id(None),
    id(True),
    id(False),
    id(StopIteration),
    id(zip),
    id(_GroupsIterator),
)
_lib.dd_init(ctypes.byref(_iface))


def groups(prompts, slots, token_budget):
    """Bounded prefill batch grouping — logic in assembly."""
    return _lib.dd_groups(prompts, slots, token_budget)
