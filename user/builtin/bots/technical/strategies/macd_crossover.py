"""MACD Crossover – phát hiện tín hiệu dựa trên MACD cắt Signal line."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vnstock_forecast.builtin.signal_based.registry import register
from vnstock_forecast.builtin.signal_based.signal import (
    Signal,
    SignalDirection,
    TradePlan,
)
from vnstock_forecast.builtin.signal_based.technical import BaseTechnique
from vnstock_forecast.engine.backtest.context import StepContext

from ..indicators.macd import compute_macd, macd_overlays


def _compute_macd(
    closes: pd.Series,
    fast: int,
    slow: int,
    signal_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Tính MACD. Delegate to indicators module."""
    return compute_macd(closes, fast, slow, signal_period)


@register("macd_crossover")
class MACDCrossover(BaseTechnique):
    """
    Kỹ thuật MACD Crossover.

    Tín hiệu:

    - **BUY**: MACD line cắt lên trên Signal line (histogram từ âm sang dương).
    - **SELL**: MACD line cắt xuống dưới Signal line (histogram từ dương sang âm).

    Params::

        fast_period:    EMA nhanh (mặc định 12).
        slow_period:    EMA chậm (mặc định 26).
        signal_period:  Signal line EMA (mặc định 9).
        sl_pct:         Stop loss % cho BUY (mặc định 7%).
        tp_pct:         Take profit % cho BUY (mặc định 10%).
    """

    name = "macd_crossover"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        sl_pct: float = 0.07,
        tp_pct: float = 0.10,
        max_holding_days=15,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.required_lookback = slow_period + signal_period + 5
        self.max_holding_days = max_holding_days

    @property
    def params(self) -> dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
        }

    def build_overlays(self, df):
        """MACD line + Signal line + Histogram."""
        return macd_overlays(
            df["Close"], self.fast_period, self.slow_period, self.signal_period
        )

    def analyze_step(self, ctx: StepContext, symbol: str) -> list[Signal]:
        """Phân tích MACD tại bar hiện tại."""
        needed = (
            max(self.required_lookback + 10, self.snapshot_lookback)
            if self.attach_snapshot
            else self.required_lookback + 10
        )
        df = ctx.history(symbol, lookback=needed)
        if len(df) < self.required_lookback:
            return []

        closes = df["Close"]
        macd_line, signal_line, histogram = _compute_macd(
            closes, self.fast_period, self.slow_period, self.signal_period
        )

        if len(histogram) < 2:
            return []
        if histogram.isna().iloc[-1] or histogram.isna().iloc[-2]:
            return []

        prev_hist = histogram.iloc[-2]
        curr_hist = histogram.iloc[-1]
        price = ctx.price(symbol)
        signals: list[Signal] = []

        # BUY: histogram từ âm/zero sang dương (MACD cắt lên Signal)
        if prev_hist <= 0 and curr_hist > 0:
            signals.append(
                Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    timestamp=ctx.timestamp,
                    trade_plan=TradePlan(
                        entry=price,
                        stop_loss=round(price * (1 - self.sl_pct), 2),
                        take_profit=round(price * (1 + self.tp_pct), 2),
                        max_holding_days=self.max_holding_days,
                    ),
                    confidence=0.55,
                    reason=f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) "  # noqa: E501
                    f"bullish crossover: hist {prev_hist:.2f} → {curr_hist:.2f}",
                    tags={"macd_bullish"},
                    metadata={
                        "macd": float(macd_line.iloc[-1]),
                        "signal": float(signal_line.iloc[-1]),
                        "histogram": float(curr_hist),
                    },
                )
            )
            if self.attach_snapshot:
                signals[-1].snapshot = self.build_snapshot(
                    df,
                    signals[-1],
                    resolution=ctx.primary_resolution,
                )

        # SELL: histogram từ dương/zero sang âm (MACD cắt xuống Signal)
        if prev_hist >= 0 and curr_hist < 0:
            signals.append(
                Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    timestamp=ctx.timestamp,
                    confidence=0.55,
                    reason=f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) "  # noqa: E501
                    f"bearish crossover: hist {prev_hist:.2f} → {curr_hist:.2f}",
                    tags={"macd_bearish"},
                    metadata={
                        "macd": float(macd_line.iloc[-1]),
                        "signal": float(signal_line.iloc[-1]),
                        "histogram": float(curr_hist),
                    },
                )
            )
            if self.attach_snapshot:
                signals[-1].snapshot = self.build_snapshot(
                    df,
                    signals[-1],
                    resolution=ctx.primary_resolution,
                )

        return signals

    def analyze_batch(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        """Phân tích MACD trên toàn bộ DataFrame."""
        if len(df) < self.required_lookback:
            return []

        closes = df["Close"]
        macd_line, signal_line, histogram = _compute_macd(
            closes, self.fast_period, self.slow_period, self.signal_period
        )
        signals: list[Signal] = []

        for i in range(1, len(histogram)):
            if histogram.isna().iloc[i] or histogram.isna().iloc[i - 1]:
                continue

            prev_hist = histogram.iloc[i - 1]
            curr_hist = histogram.iloc[i]
            price = float(closes.iloc[i])
            raw_ts = df.index[i]
            ts = pd.to_datetime(raw_ts, unit="s", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(raw_ts)
            timestamp = ts.floor("us").to_pydatetime()

            # BUY
            if prev_hist <= 0 and curr_hist > 0:
                sig = Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    timestamp=timestamp,
                    trade_plan=TradePlan(
                        entry=price,
                        stop_loss=round(price * (1 - self.sl_pct), 2),
                        take_profit=round(price * (1 + self.tp_pct), 2),
                    ),
                    confidence=0.55,
                    reason="MACD bullish crossover",
                    metadata={"histogram": float(curr_hist)},
                )
                if self.attach_snapshot:
                    sig.snapshot = self.build_snapshot(df.iloc[: i + 1], sig)
                signals.append(sig)

            # SELL
            if prev_hist >= 0 and curr_hist < 0:
                sig = Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    timestamp=timestamp,
                    confidence=0.55,
                    reason="MACD bearish crossover",
                    metadata={"histogram": float(curr_hist)},
                )
                if self.attach_snapshot:
                    sig.snapshot = self.build_snapshot(df.iloc[: i + 1], sig)
                signals.append(sig)

        return signals
