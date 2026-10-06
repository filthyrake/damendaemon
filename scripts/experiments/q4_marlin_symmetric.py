"""Isolated implicit-zero-point Q4 Marlin adapter; requires experimental ABI 2."""
from .q4_marlin_direct import DirectLibrary


class SymmetricLibrary(DirectLibrary):
    abi = 2

    def zero_points(self, n, k, device):
        return None
