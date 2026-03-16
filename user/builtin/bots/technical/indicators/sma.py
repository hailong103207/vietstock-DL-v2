"""SMA (Simple Moving Average) indicator."""

from __future__ import annotations

import pandas as pd

from vnstock_forecast.builtin.forecast.visualization.snapshot import PlotOverlays


def compute_sma(closes: pd.Series, period: int = 20) -> pd.Series:
    """
    Tính Simple Moving Average.

    Args:
        closes: Chuỗi giá đóng cửa.
        period: Chu kỳ SMA (mặc định 20).

    Returns:
        pd.Series chứa giá trị SMA.
    """
    return closes.rolling(period).mean()


def sma_overlays(
    closes: pd.Series,
    period: int = 20,
    panel: int = 0,
    color: str = "#FF9800",
) -> PlotOverlays:
    """Trả về plot overlays cho SMA indicator.

    SMA được vẽ overlay lên biểu đồ giá (panel 0).

    Args:
        closes: Chuỗi giá đóng cửa (DatetimeIndex).
        period: Chu kỳ SMA.
        panel:  Panel (mặc định 0 = overlay lên giá).
        color:  Màu đường SMA.

    Returns:
        ``PlotOverlays`` chứa SMA line.
    """
    from vnstock_forecast.builtin.forecast.visualization.snapshot import (
        IndicatorLine,
        PlotOverlays,
    )

    sma = compute_sma(closes, period)

    return PlotOverlays(
        indicators=[
            IndicatorLine(
                name=f"SMA({period})",
                data=sma,
                color=color,
                panel=panel,
                linewidth=1.3,
                linestyle="-",
            ),
        ],
    )
