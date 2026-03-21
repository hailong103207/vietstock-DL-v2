"""BaseTechnique – lớp cơ sở cho mọi kỹ thuật phân tích kỹ thuật."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from vnstock_forecast.builtin.signal_based.signal import Signal

if TYPE_CHECKING:
    from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
        PlotOverlays,
        SignalSnapshot,
    )
    from vnstock_forecast.engine.backtest.context import StepContext


class BaseTechnique(ABC):
    """
    Lớp cơ sở cho tất cả kỹ thuật phân tích kỹ thuật.

    Mỗi technique cần implement:

    - ``analyze_step(ctx, symbol)`` – phân tích tại 1 bar (incremental).
      Được gọi mỗi bước bởi ``SignalBasedBacktestBot``.

    Tùy chọn override:

    - ``analyze_batch(df)`` – phân tích toàn bộ DataFrame OHLCV một lần.
      Hữu ích cho profiler hoặc batch analysis.
    - ``required_lookback`` – số bars tối thiểu cần để phân tích.
    - ``params`` – dict tham số kỹ thuật (dùng cho profiling / logging).

    Lifecycle trong SignalBasedBacktestBot::

        for symbol in ctx.symbols:
            for technique in self.techniques:
                signals = technique.analyze_step(ctx, symbol)
                # → bot tổng hợp signals

    Example::

        class RSICrossover(BaseTechnique):
            name = "RSI_Crossover"
            required_lookback = 15

            def __init__(self, period=14, oversold=30, overbought=70):
                self.period = period
                self.oversold = oversold
                self.overbought = overbought

            @property
            def params(self):
                return {"period": self.period, "oversold": self.oversold,
                        "overbought": self.overbought}

            def analyze_step(self, ctx, symbol):
                df = ctx.history(symbol, lookback=self.required_lookback + 5)
                # ... tính RSI, phát hiện tín hiệu ...
                return signals
    """

    name: str = "BaseTechnique"

    #: Số bars tối thiểu cần có trước khi technique có thể phân tích.
    required_lookback: int = 1

    #: Bật tính năng đính kèm ``SignalSnapshot`` vào mỗi signal.
    #: Khi ``True``, ``analyze_step`` / ``analyze_batch`` sẽ tự động gắn
    #: snapshot (OHLCV + indicator overlays) vào ``signal.snapshot``.
    attach_snapshot: bool = False

    #: Số bars OHLCV lưu vào snapshot. Chỉ ảnh hưởng khi ``attach_snapshot=True``.
    snapshot_lookback: int = 60

    #: Ngưỡng confidence tối thiểu để technique này phát ra signal.
    #: Signal có confidence < min_confidence bị loại ngay tại bước thu thập,
    #: trước khi bot xử lý. Mặc định 0.0 (chấp nhận tất cả).
    min_confidence: float = 0.0

    @property
    def params(self) -> dict[str, Any]:
        """
        Tham số kỹ thuật hiện tại. Override để liệt kê params cụ thể.

        Dùng bởi profiler để log / so sánh cấu hình.
        """
        return {}

    @abstractmethod
    def analyze_step(self, ctx: StepContext, symbol: str) -> list[Signal]:
        """
        Phân tích tại bar hiện tại (incremental mode).

        Được gọi mỗi bước bởi ``SignalBasedBacktestBot.on_step()``.
        Chỉ sử dụng dữ liệu từ ``ctx`` – không future data leak.

        Args:
            ctx:    StepContext chứa dữ liệu thị trường & tài khoản.
            symbol: Mã cổ phiếu cần phân tích.

        Returns:
            Danh sách ``Signal`` nếu phát hiện tín hiệu. ``[]`` nếu không.
        """
        ...

    def analyze_batch(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        """
        Phân tích toàn bộ DataFrame OHLCV một lần (batch mode).

        Mặc định raise NotImplementedError. Override nếu technique hỗ trợ
        batch analysis (hữu ích cho profiler).

        Args:
            df:     DataFrame OHLCV đầy đủ (DatetimeIndex + OHLCV columns).
            symbol: Mã cổ phiếu.

        Returns:
            Danh sách ``Signal`` phát hiện trên toàn bộ dữ liệu.
        """
        raise NotImplementedError(
            f"{self.name} chưa implement analyze_batch(). "
            "Override phương thức này để hỗ trợ batch analysis."
        )

    # ------------------------------------------------------------------
    #  Visualization helpers
    # ------------------------------------------------------------------

    def build_overlays(self, df: pd.DataFrame) -> PlotOverlays:
        """Trả về plot overlays (indicator lines, hlines…) cho technique này.

        Override trong subclass để cung cấp dữ liệu vẽ indicator.
        Strategy gọi hàm này khi cần gắn snapshot vào signal.

        Args:
            df: DataFrame OHLCV (DatetimeIndex) đã cắt đến thời điểm
                signal (không future leak).

        Returns:
            ``PlotOverlays`` chứa indicator lines, hlines, v.v.
        """
        from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
            PlotOverlays,
        )

        return PlotOverlays()

    def build_snapshot(
        self,
        df: pd.DataFrame,
        signal: Signal,
        overlays: Optional[PlotOverlays] = None,
        lookback: Optional[int] = None,
        resolution: str = "D",
    ) -> SignalSnapshot:
        """Tạo ``SignalSnapshot`` từ OHLCV + signal + overlays.

        Phương thức tiện ích – strategy chỉ cần gọi::

            signal.snapshot = self.build_snapshot(df, signal)

        và snapshot sẽ chứa đủ dữ liệu để ``plot_signal()`` render.

        Args:
            df:         DataFrame OHLCV (DatetimeIndex) đến thời điểm signal.
            signal:     ``Signal`` vừa tạo.
            overlays:   ``PlotOverlays`` tuỳ chỉnh. ``None`` → gọi
                        ``self.build_overlays(df)``.
            lookback:   Số bars giữ lại. ``None`` → ``self.snapshot_lookback``.
            resolution: Resolution OHLCV (``"D"``, ``"60"``…).

        Returns:
            ``SignalSnapshot`` sẵn sàng gắn vào ``signal.snapshot``.
        """
        from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
            SignalSnapshot,
        )

        if overlays is None:
            overlays = self.build_overlays(df)

        lookback = lookback or self.snapshot_lookback

        # Cắt OHLCV đến thời điểm signal, giữ lại lookback bars
        ohlcv = df.copy()
        if signal.timestamp is not None:
            sig_ts = pd.Timestamp(signal.timestamp)
            if isinstance(ohlcv.index, pd.DatetimeIndex):
                mask = ohlcv.index <= sig_ts
            else:
                mask = ohlcv.index <= int(sig_ts.timestamp())
            ohlcv = ohlcv[mask]
        if len(ohlcv) > lookback:
            ohlcv = ohlcv.tail(lookback)

        # Align indicator data với cùng index
        indicators = []
        for ind in overlays.indicators:
            aligned_ind = ind.__class__(
                name=ind.name,
                data=ind.data.reindex(ohlcv.index),
                color=ind.color,
                linestyle=ind.linestyle,
                linewidth=ind.linewidth,
                panel=ind.panel,
                ylabel=ind.ylabel,
                secondary_y=ind.secondary_y,
                type=ind.type,
                alpha=ind.alpha,
            )
            indicators.append(aligned_ind)

        # Tính time_limit từ TradePlan.max_holding_days
        time_limit = None
        if signal.trade_plan and signal.trade_plan.max_holding_days:
            # Ước lượng: 1 ngày giao dịch ≈ 1.5 ngày lịch
            cal_days = int(signal.trade_plan.max_holding_days * 1.5) + 2
            time_limit = signal.timestamp + timedelta(days=cal_days)

        entry_time = signal.metadata.get("entry_time", signal.timestamp)
        if entry_time is not None:
            entry_time = pd.Timestamp(entry_time).to_pydatetime()

        exit_time = signal.metadata.get("exit_time")
        if exit_time is not None:
            exit_time = pd.Timestamp(exit_time).to_pydatetime()

        exit_price = signal.metadata.get("exit_price")
        if exit_price is not None:
            exit_price = float(exit_price)

        return SignalSnapshot(
            ohlcv=ohlcv,
            entry=signal.trade_plan.entry if signal.trade_plan else None,
            entry_time=entry_time,
            stop_loss=signal.trade_plan.stop_loss if signal.trade_plan else None,
            take_profit=signal.trade_plan.take_profit if signal.trade_plan else None,
            exit_price=exit_price,
            exit_time=exit_time,
            signal_time=signal.timestamp,
            time_limit=time_limit,
            resolution=resolution,
            symbol=signal.symbol,
            hlines=list(overlays.hlines),
            vlines=list(overlays.vlines),
            rectangles=list(overlays.rectangles),
            trendlines=list(overlays.trendlines),
            indicators=indicators,
        )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
