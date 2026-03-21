"""EngineCore – logic dùng chung cho backtest/live engine."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .bot_base import Action, ActionType
from .portfolio import CloseReason, Portfolio, TradeEvent

logger = logging.getLogger(__name__)


class EngineCore:
    """Core chứa các helper dùng chung cho nhiều loại engine."""

    def __init__(
        self,
        initial_cash: float = 100_000_000.0,
        commission_rate: float = 0.0015,
        settlement_days: int = 3,
        auto_manage_sl_tp: bool = True,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.settlement_days = settlement_days
        self.auto_manage_sl_tp = auto_manage_sl_tp

    @staticmethod
    def _normalize_multi_data(
        data: dict,
        primary_resolution: Optional[str],
    ) -> "tuple[dict[str, dict[str, pd.DataFrame]], str]":
        """
        Chuẩn hóa ``data`` sang dạng ``{resolution: {symbol: DataFrame}}``.

        * Nếu ``data`` là ``{symbol: DataFrame}`` (dạng cũ) → bọc thành
          ``{resolution: data}`` với ``resolution = primary_resolution or 'primary'``.
        * Nếu ``data`` đã là ``{resolution: {symbol: DataFrame}}`` → giữ nguyên.
        """
        if not data:
            raise ValueError("data không được rỗng")

        first_value = next(iter(data.values()))
        if isinstance(first_value, pd.DataFrame):
            # Dạng cũ: {symbol: DataFrame}
            resolution = primary_resolution or "D"
            return {resolution: data}, resolution  # type: ignore[return-value]
        else:
            # Dạng mới: {resolution: {symbol: DataFrame}}
            if primary_resolution is None:
                primary_resolution = next(iter(data))
            elif primary_resolution not in data:
                raise ValueError(
                    f"primary_resolution='{primary_resolution}' không có trong data. "
                    f"Có: {list(data)}"
                )
            return data, primary_resolution  # type: ignore[return-value]

    @staticmethod
    def _prepare_data(
        multi_data: "dict[str, dict[str, pd.DataFrame]]",
    ) -> "dict[str, dict[str, pd.DataFrame]]":
        """Validate và chuẩn hóa tất cả DataFrames trong multi_data."""
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        prepared: dict[str, dict[str, pd.DataFrame]] = {}

        for resolution, sym_data in multi_data.items():
            if not sym_data:
                raise ValueError(f"Resolution '{resolution}' không có symbol nào")
            prepared[resolution] = {}
            for symbol, df in sym_data.items():
                df = df.copy()

                # Auto-convert index sang DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index, unit="s")
                    except Exception:
                        df.index = pd.to_datetime(df.index)

                df = df.sort_index()

                missing = required_cols - set(df.columns)
                if missing:
                    raise ValueError(
                        f"DataFrame của '{symbol}' (resolution='{resolution}')"
                        f" thiếu cột: {missing}"
                    )

                prepared[resolution][symbol] = df

        return prepared

    @staticmethod
    def _collect_timestamps(
        data: dict[str, pd.DataFrame],
        start: Optional[str | datetime],
        end: Optional[str | datetime],
    ) -> list:
        """Thu thập tất cả timestamps duy nhất, lọc theo khoảng thời gian."""
        all_ts = sorted(set().union(*(df.index for df in data.values())))

        if start is not None:
            start_ts = pd.Timestamp(start)
            all_ts = [t for t in all_ts if t >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            all_ts = [t for t in all_ts if t <= end_ts]

        return all_ts

    @staticmethod
    def _prices_at(
        data: dict[str, pd.DataFrame],
        ts: pd.Timestamp,
    ) -> dict[str, float]:
        """Giá Close mới nhất của tất cả symbols tại hoặc trước *ts*."""
        prices: dict[str, float] = {}
        for symbol, df in data.items():
            available = df[df.index <= ts]
            if not available.empty:
                prices[symbol] = float(available.iloc[-1]["Close"])
        return prices

    @staticmethod
    def _to_dt(ts: pd.Timestamp) -> datetime:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts  # type: ignore[return-value]

    def _check_all_sl_tp(
        self,
        portfolio: Portfolio,
        data: dict[str, pd.DataFrame],
        symbols: list[str],
        ts: pd.Timestamp,
        timestamp: datetime,
        current_prices: dict[str, float],
    ) -> list[TradeEvent]:
        """Kiểm tra SL/TP cho tất cả symbols tại bar hiện tại."""
        events: list[TradeEvent] = []

        for symbol in symbols:
            df = data[symbol]
            bar_data = df[df.index == ts]
            if bar_data.empty:
                continue

            bar = bar_data.iloc[-1]
            closed_positions = portfolio.check_sl_tp(
                symbol,
                float(bar["High"]),
                float(bar["Low"]),
                float(bar["Close"]),
                timestamp,
            )

            for pos in closed_positions:
                assert pos.exit_price is not None
                assert pos.close_reason is not None
                events.append(
                    TradeEvent(
                        timestamp=timestamp,
                        action=pos.close_reason.value,
                        symbol=pos.symbol,
                        price=pos.exit_price,
                        quantity=pos.quantity,
                        position_id=pos.id,
                        equity=portfolio.equity(current_prices),
                        reason=f"Auto {pos.close_reason.value}",
                    )
                )

        return events

    def _execute_actions(
        self,
        actions: list[Action],
        portfolio: Portfolio,
        current_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeEvent]:
        """Thực thi danh sách Action từ bot."""
        events: list[TradeEvent] = []

        for action in actions:
            try:
                if action.type == ActionType.BUY:
                    events.extend(
                        self._exec_buy(action, portfolio, current_prices, timestamp)
                    )
                elif action.type == ActionType.SELL:
                    events.extend(
                        self._exec_sell(action, portfolio, current_prices, timestamp)
                    )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "[%s] Không thể %s %s: %s",
                    timestamp,
                    action.type.value,
                    action.symbol,
                    exc,
                )

        return events

    @staticmethod
    def _exec_buy(
        action: Action,
        portfolio: Portfolio,
        current_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeEvent]:
        # Resolve giá nếu bot không chỉ định
        if action.price is None:
            if action.symbol not in current_prices:
                raise ValueError(f"Không có giá cho '{action.symbol}'")
            action.price = current_prices[action.symbol]

        pos = portfolio.open_position(action, timestamp)

        return [
            TradeEvent(
                timestamp=timestamp,
                action="buy",
                symbol=action.symbol,
                price=action.price,
                quantity=action.quantity,
                position_id=pos.id,
                equity=portfolio.equity(current_prices),
                reason=action.reason,
            )
        ]

    @staticmethod
    def _exec_sell(
        action: Action,
        portfolio: Portfolio,
        current_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        price = action.price or current_prices.get(action.symbol)
        if price is None:
            raise ValueError(f"Không có giá cho '{action.symbol}'")

        # Tìm vị thế cần đóng
        if action.position_id:
            pos_ids = [action.position_id]
        else:
            # Bán các vị thế đã qua T+N (FIFO) – không bán lô chưa đến hạn
            pos_ids = [
                p.id for p in portfolio.sellable_positions(action.symbol, timestamp)
            ]

        for pid in pos_ids:
            # Nếu bán 1 vị thế cụ thể và có chỉ định quantity thì bán một phần
            sell_qty = (
                action.quantity
                if (action.position_id and action.quantity and action.quantity > 0)
                else None
            )
            pos = portfolio.close_position(
                pid, price, timestamp, CloseReason.MANUAL, action, quantity=sell_qty
            )
            events.append(
                TradeEvent(
                    timestamp=timestamp,
                    action="sell",
                    symbol=pos.symbol,
                    price=price,
                    quantity=pos.quantity,
                    position_id=pos.id,
                    equity=portfolio.equity(current_prices),
                    reason=action.reason,
                )
            )
        return events
