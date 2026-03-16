"""vnstock_forecast.builtin.forecast.visualization – plot & persist signals."""

from .plotter import plot_signal
from .snapshot import (
    HLine,
    IndicatorLine,
    PlotOverlays,
    Rectangle,
    SignalSnapshot,
    TrendLine,
    VLine,
)
from .store import SignalStore

__all__ = [
    # Snapshot data-structures
    "SignalSnapshot",
    "PlotOverlays",
    "IndicatorLine",
    "HLine",
    "VLine",
    "Rectangle",
    "TrendLine",
    # Plotter
    "plot_signal",
    # Store
    "SignalStore",
    # PDF Report
    "PDFProfileReport",
]


def __getattr__(name: str):
    if name == "PDFProfileReport":
        from .pdf_report import PDFProfileReport

        return PDFProfileReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
