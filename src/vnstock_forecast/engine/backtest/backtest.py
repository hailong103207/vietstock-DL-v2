"""BacktestEngine – core engine chạy backtest bar-by-bar."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .bot_base import BotBase
from .context import StepContext
from .core import EngineCore
from .portfolio import CloseReason, Portfolio, TradeEvent
from .report import BacktestReport


class BacktestEngine(EngineCore):
    """
    Engine backtest chạy bar-by-bar, tương thích mọi khung thời gian.

    Mỗi bar engine sẽ:

    1. Kiểm tra SL/TP tự động cho tất cả vị thế mở.
    2. Xây dựng ``StepContext`` (sổ lệnh, tiền, dữ liệu thị trường).
    3. Gọi ``bot.on_step(ctx)`` → nhận danh sách ``Action``.
    4. Thực thi các Action (mua/bán).
    5. Ghi lại equity curve.

    Usage::

        engine = BacktestEngine(initial_cash=100_000_000)
        report = engine.run(
            bot=my_bot,
            data={"VNM": df_vnm, "VHM": df_vhm},
            start="2023-01-01",
            end="2024-12-31",
        )
        report.print_summary()

    Data format::

        # Single-resolution (cũ, vẫn hoạt động)
        data: dict[str, pd.DataFrame]
        - Key   = symbol (vd: "VNM", "VHM")
        - Value = DataFrame OHLCV với DatetimeIndex

        # Multi-resolution (mới)
        data: dict[str, dict[str, pd.DataFrame]]
        - Outer key = resolution (vd: "D", "60", "15")
        - Inner key = symbol
        - Vòng lặp bar-by-bar được lái bởi ``primary_resolution``
          (mặc định = key đầu tiên)

        DataFrame OHLCV bắt buộc có:
            * DatetimeIndex (hoặc Unix-timestamp index – auto convert)
            * Cột: Open, High, Low, Close, Volume

    Để load từ parquet store::

        from vnstock_forecast.data.query import query_ohlcv_grouped
        grouped = query_ohlcv_grouped(symbols=["VNM"], resolutions=["D", "60"])
        # Single: data = grouped["D"]
        # Multi:  data = {"D": grouped["D"], "60": grouped["60"]}
    """

    def run(
        self,
        bot: BotBase,
        data: "dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]",
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
        primary_resolution: Optional[str] = None,
    ) -> BacktestReport:
        """
        Chạy backtest.

        Args:
            bot:   Đối tượng kế thừa ``BotBase``.
            data:  Một trong hai dạng:

                   * ``{symbol: DataFrame}`` – single-resolution (cũ, vẫn hoạt động).
                   * ``{resolution: {symbol: DataFrame}}`` – multi-resolution.
                     Ví dụ: ``{"D": {"VNM": df_daily}, "60": {"VNM": df_hourly}}``.

            start: Ngày bắt đầu (inclusive). ``None`` = từ đầu dữ liệu.
            end:   Ngày kết thúc (inclusive). ``None`` = đến cuối dữ liệu.
            primary_resolution:
                   Resolution dùng để lái vòng lặp bar-by-bar và xác định
                   giá thực hiện (SL/TP, mua/bán).
                   - Single-resolution: bỏ qua tham số này.
                   - Multi-resolution: mặc định = key đầu tiên trong ``data``.

        Returns:
            ``BacktestReport`` chứa toàn bộ kết quả.
        """
        multi_data, primary_resolution = self._normalize_multi_data(
            data, primary_resolution
        )
        multi_data = self._prepare_data(multi_data)
        primary_data = multi_data[primary_resolution]
        symbols = list(primary_data.keys())

        all_timestamps = self._collect_timestamps(primary_data, start, end)
        if not all_timestamps:
            raise ValueError("Không có dữ liệu trong khoảng thời gian chỉ định.")

        portfolio = Portfolio(
            self.initial_cash, self.commission_rate, self.settlement_days
        )
        events: list[TradeEvent] = []
        equity_curve: list[tuple[datetime, float]] = []

        # --- on_start ------------------------------------------------
        first_ts = all_timestamps[0]
        first_prices = self._prices_at(primary_data, first_ts)
        bot.on_start(
            StepContext(
                self._to_dt(first_ts),
                portfolio,
                multi_data,
                first_prices,
                symbols,
                primary_resolution,
            )
        )

        # --- Iterate (bắt đầu từ bar thứ 2 để luôn có ≥1 bar lịch sử) ---
        for i in range(1, len(all_timestamps)):
            ts = all_timestamps[i]
            timestamp = self._to_dt(ts)
            current_prices = self._prices_at(primary_data, ts)

            # 1) SL / TP tự động (có thể tắt để bot tự xử lý bằng SELL action)
            if self.auto_manage_sl_tp:
                sl_tp_events = self._check_all_sl_tp(
                    portfolio, primary_data, symbols, ts, timestamp, current_prices
                )
                events.extend(sl_tp_events)

            # 2) Build context
            ctx = StepContext(
                timestamp,
                portfolio,
                multi_data,
                current_prices,
                symbols,
                primary_resolution,
            )

            # 3) Gọi bot
            actions = bot.on_step(ctx)

            # 4) Thực thi actions
            action_events = self._execute_actions(
                actions, portfolio, current_prices, timestamp
            )
            events.extend(action_events)

            # 5) Ghi equity
            equity_curve.append((timestamp, portfolio.equity(current_prices)))

        # --- Đóng tất cả vị thế còn mở cuối kỳ ----------------------
        last_ts = all_timestamps[-1]
        last_dt = self._to_dt(last_ts)
        last_prices = self._prices_at(primary_data, last_ts)

        for pos in list(portfolio.open_positions):
            price = last_prices.get(pos.symbol, pos.entry_price)
            closed = portfolio.close_position(
                pos.id, price, last_dt, CloseReason.END_OF_DATA
            )
            events.append(
                TradeEvent(
                    timestamp=last_dt,
                    action="end_of_data",
                    symbol=closed.symbol,
                    price=price,
                    quantity=closed.quantity,
                    position_id=closed.id,
                    equity=portfolio.equity(last_prices),
                    reason="Đóng cuối kỳ backtest",
                )
            )

        # --- on_end --------------------------------------------------
        final_ctx = StepContext(
            last_dt, portfolio, multi_data, last_prices, symbols, primary_resolution
        )
        bot.on_end(final_ctx)

        return BacktestReport(
            bot_name=bot.name,
            symbols=symbols,
            start=self._to_dt(all_timestamps[0]),
            end=last_dt,
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            portfolio=portfolio,
            events=events,
            equity_curve=equity_curve,
        )
