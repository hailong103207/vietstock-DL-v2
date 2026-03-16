"""vnstock_forecast.builtin.forecast – analysis, strategies, profiling & forecasting."""

from .profile import SignalProfile
from .profiler import Profiler
from .registry import get_all_techniques, get_technique, register
from .signal import Signal, SignalDirection, TradePlan
from .visualization import (
    HLine,
    IndicatorLine,
    PlotOverlays,
    Rectangle,
    SignalSnapshot,
    SignalStore,
    TrendLine,
    VLine,
    plot_signal,
)

__all__ = [
    "Signal",
    "SignalDirection",
    "TradePlan",
    "register",
    "get_technique",
    "get_all_techniques",
    "SignalProfile",
    "Profiler",
    # Visualization
    "SignalSnapshot",
    "PlotOverlays",
    "IndicatorLine",
    "HLine",
    "VLine",
    "Rectangle",
    "TrendLine",
    "plot_signal",
    "SignalStore",
]
