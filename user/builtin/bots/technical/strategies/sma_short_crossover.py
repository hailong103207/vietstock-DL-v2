"""SMA Short Crossover – dual-SMA crossover với confidence động và TP/SL thích nghi."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from vnstock_forecast.builtin.signal_based.registry import register
from vnstock_forecast.builtin.signal_based.signal import (
    Signal,
    SignalDirection,
    TradePlan,
)
from vnstock_forecast.builtin.signal_based.technical import BaseTechnique
from vnstock_forecast.engine.backtest.context import StepContext

from ..confirmations import (
    apply_confirmations,
    check_breakout_resistance,
    check_no_large_fvg,
    check_volume_surge,
)
from ..indicators.resistance import compute_resistance_levels, compute_support_levels
from ..indicators.sma import compute_sma, sma_overlays


@register("sma_short_crossover")
class SMAShortCrossover(BaseTechnique):
    """
    Dual-SMA Crossover với confidence động và TP/SL thích nghi.

    Tín hiệu chính
    --------------
    ``SMA(short)`` cắt lên trên ``SMA(long)``.
    Confidence ban đầu = 0.5, được điều chỉnh dựa trên độ mạnh của
    tín hiệu (khoảng cách & độ dốc SMA ngắn):

    .. code-block::

        normalized_gap  = (sma_short - sma_long) / close
        slope           = (sma_short[-1] - sma_short[-2]) / sma_short[-2]
        strength        = (normalized_gap + slope) / 2
        primary_boost   = clip(strength / MAX_STRENGTH, 0, 1) * 0.1
        base_confidence = 0.5 + primary_boost          # ∈ [0.5, 0.6]

    Tín hiệu xác nhận (mỗi cái +≤0.1, tổng max 0.8)
    -----------------------------------------------
    - Volume surge      (+≤0.1)
    - Breakout resistance (+0.1)

    TP / SL
    -------
    - ``tp = max(nearest_resistance_above, 1.1 * entry)``
    - ``sl = max(nearest_support_below,   entry - (tp - entry) / 3)``

    Params
    ------
    short_period : int   Chu kỳ SMA ngắn (mặc định 10).
    long_period  : int   Chu kỳ SMA dài  (mặc định 30).
    sr_lookback  : int   Số nến để tính support/resistance (mặc định 60).
    vol_lookback : int   Số nến để tính trung bình volume  (mặc định 20).
    surge_threshold : float  Ngưỡng volume surge (mặc định 1.5×).
    max_strength : float Giá trị strength chuẩn hóa tương ứng boost=0.1 (mặc định 0.03)
    """

    name = "sma_short_crossover"

    def __init__(
        self,
        short_period: int = 10,
        long_period: int = 30,
        sr_lookback: int = 60,
        vol_lookback: int = 20,
        surge_threshold: float = 1.5,
        max_strength: float = 0.03,
        min_confidence: float = 0.5,
    ) -> None:
        if short_period >= long_period:
            raise ValueError(
                f"short_period ({short_period}) phải nhỏ hơn long_period ({long_period})."  # noqa: E501
            )
        self.short_period = short_period
        self.long_period = long_period
        self.sr_lookback = sr_lookback
        self.vol_lookback = vol_lookback
        self.surge_threshold = surge_threshold
        self.max_strength = max_strength
        self.min_confidence = min_confidence

        self.required_lookback = max(long_period + 2, sr_lookback, vol_lookback + 1)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "short_period": self.short_period,
            "long_period": self.long_period,
            "sr_lookback": self.sr_lookback,
            "vol_lookback": self.vol_lookback,
            "surge_threshold": self.surge_threshold,
            "max_strength": self.max_strength,
            "min_confidence": self.min_confidence,
        }

    # ------------------------------------------------------------------
    #  Overlay (snapshot)
    # ------------------------------------------------------------------

    def build_overlays(self, df: pd.DataFrame):
        """Hai SMA overlay lên biểu đồ giá."""
        short_overlay = sma_overlays(df["Close"], self.short_period, color="#FF9800")
        long_overlay = sma_overlays(df["Close"], self.long_period, color="#2196F3")
        # Gộp indicator lists
        from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
            PlotOverlays,
        )

        return PlotOverlays(
            indicators=short_overlay.indicators + long_overlay.indicators
        )

    # ------------------------------------------------------------------
    #  Core logic helpers
    # ------------------------------------------------------------------

    def _compute_tp_sl(
        self,
        entry: float,
        df: pd.DataFrame,
    ) -> tuple[float, float]:
        """
        Tính TP và SL dựa trên nearest resistance / support.

        Returns:
            ``(take_profit, stop_loss)``
        """
        window = min(self.sr_lookback, len(df))
        sub = df.iloc[-window:]

        res_levels = compute_resistance_levels(
            sub["Close"], sub["Volume"], window=window, num_levels=5
        )
        sup_levels = compute_support_levels(
            sub["Close"], sub["Volume"], window=window, num_levels=5
        )

        # Nearest resistance TRÊN entry
        above = [(lvl, vol) for lvl, vol in res_levels if lvl > entry]
        nearest_resistance: Optional[float] = (
            min(above, key=lambda x: x[0])[0] if above else None
        )

        # Nearest support DƯỚI entry (ot sắp xếp giảm dần → lấy phần tử đầu)
        below = [(lvl, vol) for lvl, vol in sup_levels if lvl < entry]
        nearest_support: Optional[float] = (
            max(below, key=lambda x: x[0])[0] if below else None
        )

        # TP = max(nearest_resistance, 1.1 * entry)
        min_tp = round(entry * 1.1, 2)
        tp = round(
            max(nearest_resistance, min_tp) if nearest_resistance else min_tp,
            2,
        )

        # SL = max(nearest_support, entry - (tp - entry) / 3)
        formula_sl = round(entry - (tp - entry) / 3, 2)
        sl = round(
            min(nearest_support, formula_sl) if nearest_support else formula_sl,
            2,
        )

        return tp, sl

    # ------------------------------------------------------------------
    #  Analyze step (live / backtest)
    # ------------------------------------------------------------------

    def analyze_step(self, ctx: StepContext, symbol: str) -> list[Signal]:
        """Phân tích dual-SMA crossover tại bar hiện tại."""
        lookback = (
            max(self.required_lookback, self.snapshot_lookback)
            if self.attach_snapshot
            else self.required_lookback
        )
        df = ctx.history(symbol, lookback=lookback)
        if len(df) < self.required_lookback:
            return []

        signals = self._detect(df, symbol, ctx.timestamp, ctx.price(symbol))

        if self.attach_snapshot and signals:
            for sig in signals:
                sig.snapshot = self.build_snapshot(
                    df, sig, resolution=ctx.primary_resolution
                )

        return signals

    # ------------------------------------------------------------------
    #  Analyze batch (profiler)
    # ------------------------------------------------------------------

    def analyze_batch(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        """Phân tích trên toàn bộ DataFrame (dành cho profiler)."""
        if len(df) < self.required_lookback:
            return []

        signals: list[Signal] = []
        for i in range(self.required_lookback, len(df)):
            sub = df.iloc[: i + 1]
            raw_ts = df.index[i]
            ts = pd.to_datetime(raw_ts, unit="s", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(raw_ts)
            timestamp = ts.floor("us").to_pydatetime()
            price = float(df["Close"].iloc[i])
            batch_signals = self._detect(sub, symbol, timestamp, price)
            if self.attach_snapshot:
                for sig in batch_signals:
                    sig.snapshot = self.build_snapshot(sub, sig)
            signals.extend(batch_signals)

        return signals

    # ------------------------------------------------------------------
    #  Internal: core detection logic (shared by step & batch)
    # ------------------------------------------------------------------

    def _detect(
        self,
        df: pd.DataFrame,
        symbol: str,
        timestamp,
        price: float,
    ) -> list[Signal]:
        closes = df["Close"]
        sma_short = compute_sma(closes, self.short_period)
        sma_long = compute_sma(closes, self.long_period)

        # Cần ít nhất 2 bars đã tính xong
        if sma_short.isna().iloc[-1] or sma_short.isna().iloc[-2]:
            return []
        if sma_long.isna().iloc[-1] or sma_long.isna().iloc[-2]:
            return []

        prev_short = float(sma_short.iloc[-2])
        curr_short = float(sma_short.iloc[-1])
        prev_long = float(sma_long.iloc[-2])
        curr_long = float(sma_long.iloc[-1])

        signals: list[Signal] = []

        # ── BUY: sma_short cắt lên trên sma_long ────────────────────
        if prev_short <= prev_long and curr_short > curr_long:
            # 1) Tính base confidence từ tín hiệu chính
            normalized_gap = (curr_short - curr_long) / price if price > 0 else 0.0
            slope = (curr_short - prev_short) / prev_short if prev_short > 0 else 0.0
            strength = (normalized_gap + slope) / 2
            primary_boost = min(strength / self.max_strength, 1.0) * 0.1
            primary_boost = max(0.0, primary_boost)
            base_conf = 0.5 + primary_boost  # ∈ [0.5, 0.6]

            # 2) Tín hiệu xác nhận
            confirmations = [
                check_volume_surge(
                    df,
                    lookback=self.vol_lookback,
                    surge_threshold=self.surge_threshold,
                ),
                check_breakout_resistance(
                    df,
                    current_price=price,
                    lookback=self.sr_lookback,
                ),
                check_no_large_fvg(
                    df,
                    current_price=price,
                    lookback=self.sr_lookback,
                ),
            ]
            final_conf, conf_reasons = apply_confirmations(base_conf, confirmations)
            # Cap tối đa 0.8 theo thiết kế
            final_conf = min(final_conf, 0.8)

            # 3) TP / SL
            tp, sl = self._compute_tp_sl(price, df)

            # 4) Reason
            reason_parts = [
                f"SMA({self.short_period})={curr_short:.0f} cắt lên "
                f"SMA({self.long_period})={curr_long:.0f}",
                f"gap={normalized_gap * 100:.2f}% slope={slope * 100:.3f}%",
            ]
            if conf_reasons:
                reason_parts += conf_reasons

            signals.append(
                Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    timestamp=timestamp,
                    trade_plan=TradePlan(
                        entry=price,
                        stop_loss=sl,
                        take_profit=tp,
                    ),
                    confidence=final_conf,
                    reason=" | ".join(reason_parts),
                    tags={"sma_short_crossover", "dual_sma", "bullish"},
                    metadata={
                        "sma_short": curr_short,
                        "sma_long": curr_long,
                        "normalized_gap": normalized_gap,
                        "slope": slope,
                        "primary_boost": primary_boost,
                        "confirmation_reasons": conf_reasons,
                    },
                )
            )

        # ── SELL: sma_short cắt xuống dưới sma_long ─────────────────
        elif prev_short >= prev_long and curr_short < curr_long:
            signals.append(
                Signal(
                    technique=self.name,
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    timestamp=timestamp,
                    confidence=0.5,
                    reason=(
                        f"SMA({self.short_period})={curr_short:.0f} cắt xuống "
                        f"SMA({self.long_period})={curr_long:.0f}"
                    ),
                    tags={"sma_short_crossover", "dual_sma", "bearish"},
                    metadata={
                        "sma_short": curr_short,
                        "sma_long": curr_long,
                    },
                )
            )

        return signals
