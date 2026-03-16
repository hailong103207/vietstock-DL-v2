"""MACD (Moving Average Convergence Divergence) indicator."""

from __future__ import annotations

import pandas as pd

from vnstock_forecast.builtin.forecast import PlotOverlays


def compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Tính MACD, Signal line và Histogram.

    Args:
        closes: Chuỗi giá đóng cửa.
        fast: Chu kỳ EMA nhanh (mặc định 12).
        slow: Chu kỳ EMA chậm (mặc định 26).
        signal_period: Chu kỳ Signal line EMA (mặc định 9).

    Returns:
        Tuple (macd_line, signal_line, histogram).
    """
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_overlays(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    panel: int = 2,
) -> PlotOverlays:
    """Trả về plot overlays cho MACD indicator.

    Bao gồm:
    - MACD line (xanh dương).
    - Signal line (cam).
    - Histogram (xám, dạng bar).
    - HLine zero (đường 0).

    Args:
        closes:        Chuỗi giá đóng cửa (DatetimeIndex).
        fast:          Chu kỳ EMA nhanh.
        slow:          Chu kỳ EMA chậm.
        signal_period: Chu kỳ Signal EMA.
        panel:         Panel subplot (mặc định 2).

    Returns:
        ``PlotOverlays`` chứa indicator lines + hline zero.
    """
    from vnstock_forecast.builtin.forecast.visualization.snapshot import (
        HLine,
        IndicatorLine,
        PlotOverlays,
    )

    macd_line, signal_line, histogram = compute_macd(closes, fast, slow, signal_period)

    label = f"MACD({fast},{slow},{signal_period})"

    return PlotOverlays(
        indicators=[
            IndicatorLine(
                name=f"{label} MACD",
                data=macd_line,
                color="#2196F3",
                panel=panel,
                ylabel="MACD",
                linewidth=1.2,
            ),
            IndicatorLine(
                name=f"{label} Signal",
                data=signal_line,
                color="#FF9800",
                panel=panel,
                ylabel="MACD",
                linewidth=1.0,
                linestyle="--",
            ),
            IndicatorLine(
                name=f"{label} Hist",
                data=histogram,
                color="#9E9E9E",
                panel=panel,
                ylabel="MACD",
                type="bar",
                alpha=0.5,
            ),
        ],
        hlines=[
            HLine(
                value=0,
                color="#616161",
                linestyle="-",
                linewidth=0.5,
                panel=panel,
            ),
        ],
    )
