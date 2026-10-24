"""Re-export of the runtime thermal pacer for experiment harnesses.

The implementation lives in damendaemon.thermal_pacing so `serve` can
opt in (DAMENDAEMON_THERMAL_SOFT_STOP) without importing scripts.
"""
from damendaemon.thermal_pacing import (  # noqa: F401
    ThermalPacer, activate, activate_from_env, read_sensors)

__all__ = ['ThermalPacer', 'activate', 'activate_from_env', 'read_sensors']
