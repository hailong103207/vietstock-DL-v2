"""RSI (Relative Strength Index) indicator."""

from __future__ import annotations

import pandas as pd

from vnstock_forecast.builtin.signal_based import PlotOverlays


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính RSI (Relative Strength Index).

    Args:
        closes: Chuỗi giá đóng cửa.
        period: Chu kỳ RSI (mặc định 14).

    Returns:
        pd.Series chứa giá trị RSI.
    """
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_overlays(
    closes: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    panel: int = 2,
) -> PlotOverlays:
    """Trả về plot overlays cho RSI indicator.

    Bao gồm:
    - RSI line trên subplot riêng.
    - HLine oversold & overbought.

    Args:
        closes:     Chuỗi giá đóng cửa (DatetimeIndex).
        period:     Chu kỳ RSI.
        oversold:   Ngưỡng quá bán.
        overbought: Ngưỡng quá mua.
        panel:      Panel subplot (mặc định 2; panel 1 = volume).

    Returns:
        ``PlotOverlays`` chứa indicator lines + hlines.
    """
    from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
        HLine,
        IndicatorLine,
        PlotOverlays,
    )

    rsi = compute_rsi(closes, period)

    return PlotOverlays(
        indicators=[
            IndicatorLine(
                name=f"RSI({period})",
                data=rsi,
                color="#7B1FA2",
                panel=panel,
                ylabel="RSI",
                linewidth=1.2,
            ),
        ],
        hlines=[
            HLine(
                value=oversold,
                color="#4CAF50",
                linestyle="--",
                linewidth=0.8,
                label=f"Oversold ({oversold:.0f})",
                panel=panel,
            ),
            HLine(
                value=overbought,
                color="#F44336",
                linestyle="--",
                linewidth=0.8,
                label=f"Overbought ({overbought:.0f})",
                panel=panel,
            ),
        ],
    )
