"""vnstock_forecast.builtin.signal_based.technical – technical analysis module."""

from vnstock_forecast.builtin.signal_based.registry import (
    get_all_techniques,
    get_technique,
    register,
)
from vnstock_forecast.builtin.signal_based.technical import (
    BaseTechnique,
    SignalBasedBacktestBot,
)

__all__ = [
    "BaseTechnique",
    "SignalBasedBacktestBot",
    "register",
    "get_technique",
    "get_all_techniques",
]
