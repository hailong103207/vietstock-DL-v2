"""RSI Crossover – phát hiện tín hiệu dựa trên RSI vượt ngưỡng."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vnstock_forecast.builtin.forecast.registry import register
from vnstock_forecast.builtin.forecast.signal import Signal, SignalDirection, TradePlan
from vnstock_forecast.builtin.forecast.technical import BaseTechnique
from vnstock_forecast.engine.backtest.context import StepContext

from ..indicators.rsi import compute_rsi, rsi_overlays


def _compute_rsi(closes: pd.Series, period: int) -> pd.Series:
    """Tính RSI (Relative Strength Index). Delegate to indicators module."""
    return compute_rsi(closes, period)


@register("rsi_crossover")
class RSICrossover(BaseTechnique):
    """
    Kỹ thuật RSI Crossover.

    Tín hiệu:

    - **BUY**: RSI vượt lên trên ngưỡng oversold (mặc định 30) từ dưới.
    - **SELL**: RSI vượt xuống dưới ngưỡng overbought (mặc định 70) từ trên.

    Params::

        period:     Chu kỳ RSI (mặc định 14).
        oversold:   Ngưỡng quá bán (mặc định 30).
        overbought: Ngưỡng quá mua (mặc định 70).
        sl_pct:     Stop loss % cho BUY signal (mặc định 7%).
        tp_pct:     Take profit % cho BUY signal (mặc định 10%).
    """

    name = "rsi_crossover"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        sl_pct: float = 0.07,
        tp_pct: float = 0.10,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.required_lookback = period + 5

    @property
    def params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
        }

    def build_overlays(self, df):
        """RSI line + oversold/overbought hlines."""
        return rsi_overlays(df["Close"], self.period, self.oversold, self.overbought)

    def analyze_step(self, ctx: StepContext, symbol: str) -> list[Signal]:
        """Phân tích RSI tại bar hiện tại."""
        lookback = (
            max(self.required_lookback + 5, self.snapshot_lookback)
            if self.attach_snapshot
            else self.required_lookback + 5
        )
        df = ctx.history(symbol, lookback=lookback)
        if len(df) < self.required_lookback:
            return []

        closes = df["Close"]
        rsi = _compute_rsi(closes, self.period)

        if len(rsi) < 2 or rsi.isna().iloc[-1] or rsi.isna().iloc[-2]:
            return []

        prev_rsi = rsi.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        price = ctx.price(symbol)
        signals: list[Signal] = []

        # BUY: RSI vượt lên trên oversold
        if prev_rsi <= self.oversold and curr_rsi > self.oversold:
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
                    ),
                    confidence=0.6,
                    reason=f"RSI({self.period}) vượt lên {self.oversold}: "
                    f"{prev_rsi:.1f} → {curr_rsi:.1f}",
                    tags={"oversold_bounce"},
                    metadata={"rsi_prev": prev_rsi, "rsi_curr": curr_rsi},
                )
            )
            if self.attach_snapshot:
                signals[-1].snapshot = self.build_snapshot(
                    df,
                    signals[-1],
                    resolution=ctx.primary_resolution,
                )

        # SELL: RSI vượt xuống dưới overbought
        if prev_rsi >= self.overbought and curr_rsi < self.overbought:
            signals.append(
                Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    timestamp=ctx.timestamp,
                    confidence=0.6,
                    reason=f"RSI({self.period}) rơi khỏi {self.overbought}: "
                    f"{prev_rsi:.1f} → {curr_rsi:.1f}",
                    tags={"overbought_reversal"},
                    metadata={"rsi_prev": prev_rsi, "rsi_curr": curr_rsi},
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
        """Phân tích RSI trên toàn bộ DataFrame."""
        if len(df) < self.required_lookback:
            return []

        closes = df["Close"]
        rsi = _compute_rsi(closes, self.period)
        signals: list[Signal] = []

        for i in range(1, len(rsi)):
            if rsi.isna().iloc[i] or rsi.isna().iloc[i - 1]:
                continue

            prev_rsi = rsi.iloc[i - 1]
            curr_rsi = rsi.iloc[i]
            price = float(closes.iloc[i])
            timestamp = pd.Timestamp(df.index[i]).to_pydatetime()

            # BUY
            if prev_rsi <= self.oversold and curr_rsi > self.oversold:
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
                    confidence=0.6,
                    reason=f"RSI({self.period}) vượt lên {self.oversold}",
                    metadata={"rsi_prev": prev_rsi, "rsi_curr": curr_rsi},
                )
                if self.attach_snapshot:
                    sig.snapshot = self.build_snapshot(df.iloc[: i + 1], sig)
                signals.append(sig)

            # SELL
            if prev_rsi >= self.overbought and curr_rsi < self.overbought:
                sig = Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    timestamp=timestamp,
                    confidence=0.6,
                    reason=f"RSI({self.period}) rơi khỏi {self.overbought}",
                    metadata={"rsi_prev": prev_rsi, "rsi_curr": curr_rsi},
                )
                if self.attach_snapshot:
                    sig.snapshot = self.build_snapshot(df.iloc[: i + 1], sig)
                signals.append(sig)

        return signals
