"""LiveEngine – chạy bot ở thời điểm hiện tại với dữ liệu local cập nhật mới nhất."""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from vnstock_forecast.builtin.signal_based.interactive.signal_based_portfolio import (
    SignalBasedPortfolio,
)
from vnstock_forecast.builtin.signal_based.signal import Signal, SignalDirection
from vnstock_forecast.engine.backtest.bot_base import Action, ActionType
from vnstock_forecast.engine.backtest.core import EngineCore
from vnstock_forecast.engine.backtest.portfolio import CloseReason, TradeEvent


@dataclass
class InteractiveRunResult:
    """Kết quả của một lần gọi ``InteractiveEngine.run_current``."""

    timestamp: datetime
    primary_resolution: str
    symbols: list[str]
    actions: list[Action]
    events: list[TradeEvent]
    equity: float
    skipped: bool = False
    reason: str = ""


class InteractiveEngine(EngineCore):
    """
    Live-like engine chạy theo snapshot hiện tại của dữ liệu market.

    Luồng ``run_current``:

    1. Cập nhật OHLCV mới nhất từ nguồn dữ liệu.
    2. Query lại dữ liệu grouped theo resolution/symbol.
    3. Tạo ``StepContext`` tại bar mới nhất của ``primary_resolution``.
    4. Chạy auto SL/TP (nếu bật), gọi ``bot.on_step``, thực thi actions.
    5. Ghi lại event/equity vào state nội bộ để có thể save/load/resume.

    Engine giữ trạng thái danh mục giữa các lần gọi ``run_current``.
    """

    def __init__(
        self,
        initial_cash: float = 100_000_000.0,
        commission_rate: float = 0.0015,
        settlement_days: int = 3,
        auto_manage_sl_tp: bool = True,
        lookback_days: int = 365,
    ) -> None:
        super().__init__(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            settlement_days=settlement_days,
            auto_manage_sl_tp=auto_manage_sl_tp,
        )
        self.lookback_days = lookback_days

        self.portfolio = SignalBasedPortfolio(
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            settlement_days=self.settlement_days,
        )
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []

        self.last_timestamp: Optional[datetime] = None
        self.last_primary_resolution: Optional[str] = None
        self.last_symbols: list[str] = []
        self.started: bool = False
        self.bot_name: Optional[str] = None

    @staticmethod
    def _to_list(value: list[str] | str) -> list[str]:
        return [value] if isinstance(value, str) else list(value)

    @staticmethod
    def _to_ts(value: int | datetime | pd.Timestamp) -> pd.Timestamp:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        return pd.to_datetime(value, unit="s")

    @staticmethod
    def _to_datetime(value: int | datetime | pd.Timestamp) -> datetime:
        ts = InteractiveEngine._to_ts(value)
        return ts.to_pydatetime()

    @staticmethod
    def _to_action_list(actions: list[Action] | Action | None) -> list[Action]:
        if actions is None:
            return []
        if isinstance(actions, Action):
            return [actions]
        return list(actions)

    def execute_actions(
        self,
        actions: list[Action] | Action | None,
        *,
        timestamp: int | datetime | pd.Timestamp,
        current_prices: dict[str, float],
        primary_resolution: str = "D",
        sl_tp_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> InteractiveRunResult:
        """
        Thực thi actions được truyền từ bên ngoài trên trạng thái engine hiện tại.

        Args:
            actions: Một action hoặc danh sách action cần thực thi.
            timestamp: Thời điểm thực thi.
            current_prices: Giá hiện tại theo symbol (dùng để resolve price và tính equity).
            primary_resolution: Resolution ngữ cảnh của lần thực thi.
            sl_tp_data: Dữ liệu OHLCV tại primary resolution để auto SL/TP theo bar hiện tại.

        Returns:
            ``InteractiveRunResult`` chứa actions/events/equity sau khi thực thi.
        """
        dt = self._to_datetime(timestamp)
        action_list = self._to_action_list(actions)

        # Theo pattern BacktestEngine:
        # 1) auto SL/TP trước, 2) execute actions, 3) ghi equity.
        step_events: list[TradeEvent] = []
        primary_symbols: list[str] = []

        action_events = self._execute_actions(
            action_list,
            self.portfolio,
            current_prices,
            dt,
        )
        step_events.extend(action_events)

        equity = self.portfolio.equity(current_prices)
        # self.equity_curve.append((dt, equity))
        self.events.extend(step_events)

        self.last_timestamp = dt
        self.last_primary_resolution = primary_resolution
        self.last_symbols = primary_symbols or sorted(set(current_prices))

        return InteractiveRunResult(
            timestamp=dt,
            primary_resolution=primary_resolution,
            symbols=self.last_symbols,
            actions=action_list,
            events=step_events,
            equity=equity,
        )

    def execute_signal(
        self,
        signal: Signal,
        *,
        quantity: float,
        current_prices: dict[str, float],
        timestamp: int | datetime | pd.Timestamp,
        primary_resolution: str = "D",
        position_id: Optional[str] = None,
    ) -> InteractiveRunResult:
        """Thực thi trực tiếp một signal thành action tương ứng."""
        if signal.direction == SignalDirection.BUY and quantity <= 0:
            raise ValueError("quantity phải > 0")
        dt = self._to_datetime(timestamp)
        events: list[TradeEvent] = []

        if signal.direction == SignalDirection.BUY:
            entry_price = (
                signal.trade_plan.entry
                if signal.trade_plan is not None
                else current_prices.get(signal.symbol)
            )
            if entry_price is None:
                raise ValueError(f"Không có giá cho '{signal.symbol}'")

            action = Action(
                type=ActionType.BUY,
                symbol=signal.symbol,
                quantity=quantity,
                price=entry_price,
                stop_loss=(
                    signal.trade_plan.stop_loss
                    if signal.trade_plan is not None
                    else None
                ),
                take_profit=(
                    signal.trade_plan.take_profit
                    if signal.trade_plan is not None
                    else None
                ),
                max_holding_days=(
                    signal.trade_plan.max_holding_days
                    if signal.trade_plan is not None
                    else None
                ),
                reason=signal.reason,
            )
            context_from_ts: Optional[int] = None
            context_to_ts: Optional[int] = None
            if signal.snapshot is not None and not signal.snapshot.ohlcv.empty:
                start_ts = pd.Timestamp(signal.snapshot.ohlcv.index.min())
                end_ts = pd.Timestamp(signal.snapshot.ohlcv.index.max())
                context_from_ts = int(start_ts.timestamp())
                context_to_ts = int(end_ts.timestamp())

            pos = self.portfolio.open_position(
                action,
                dt,
                signal=signal,
                resolution=primary_resolution,
                context_from_ts=context_from_ts,
                context_to_ts=context_to_ts,
            )
            events.append(
                TradeEvent(
                    timestamp=dt,
                    action="buy",
                    symbol=action.symbol,
                    price=entry_price,
                    quantity=quantity,
                    position_id=pos.id,
                    equity=self.portfolio.equity(current_prices),
                    reason=signal.reason,
                )
            )
            action_list: list[Action] = [action]
        else:
            exit_price = current_prices.get(signal.symbol)
            if exit_price is None:
                raise ValueError(f"Không có giá cho '{signal.symbol}'")

            action = Action(
                type=ActionType.SELL,
                symbol=signal.symbol,
                quantity=quantity,
                price=exit_price,
                position_id=position_id,
                reason=signal.reason,
            )

            if position_id is not None:
                pos_ids = [position_id]
            else:
                pos_ids = [
                    p.id for p in self.portfolio.sellable_positions(signal.symbol, dt)
                ]

            sell_qty = quantity if (position_id and quantity > 0) else None
            for pid in pos_ids:
                closed = self.portfolio.close_position(
                    pid,
                    exit_price,
                    dt,
                    reason=CloseReason.MANUAL,
                    exit_action=action,
                    quantity=sell_qty,
                    signal=signal,
                )
                events.append(
                    TradeEvent(
                        timestamp=dt,
                        action="sell",
                        symbol=closed.symbol,
                        price=exit_price,
                        quantity=closed.initial_quantity,
                        position_id=closed.id,
                        equity=self.portfolio.equity(current_prices),
                        reason=signal.reason,
                    )
                )
            action_list = [action]

        equity = self.portfolio.equity(current_prices)
        self.events.extend(events)
        self.last_timestamp = dt
        self.last_primary_resolution = primary_resolution
        self.last_symbols = sorted(set(current_prices))

        return InteractiveRunResult(
            timestamp=dt,
            primary_resolution=primary_resolution,
            symbols=self.last_symbols,
            actions=action_list,
            events=events,
            equity=equity,
        )

    def update_position_risk(
        self,
        *,
        position_id: str,
        timestamp: int | datetime | pd.Timestamp,
        current_prices: dict[str, float],
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        note: str = "",
        primary_resolution: str = "D",
    ) -> InteractiveRunResult:
        """Cập nhật SL/TP của position mở thông qua portfolio interactive."""
        dt = self._to_datetime(timestamp)
        updated = self.portfolio.update_position_risk(
            position_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=dt,
            note=note,
        )

        event = TradeEvent(
            timestamp=dt,
            action="update_risk",
            symbol=updated.symbol,
            price=updated.entry_price,
            quantity=0.0,
            position_id=updated.id,
            equity=self.portfolio.equity(current_prices),
            reason=note,
        )
        self.events.append(event)

        equity = self.portfolio.equity(current_prices)
        self.last_timestamp = dt
        self.last_primary_resolution = primary_resolution
        self.last_symbols = sorted(set(current_prices))

        return InteractiveRunResult(
            timestamp=dt,
            primary_resolution=primary_resolution,
            symbols=self.last_symbols,
            actions=[],
            events=[event],
            equity=equity,
        )

    def execute_action(
        self,
        action: Action,
        *,
        timestamp: int | datetime | pd.Timestamp,
        current_prices: dict[str, float],
        primary_resolution: str = "D",
        sl_tp_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> InteractiveRunResult:
        """Thực thi một action đơn, tương thích API với ``execute_actions``."""
        return self.execute_actions(
            action,
            timestamp=timestamp,
            current_prices=current_prices,
            primary_resolution=primary_resolution,
            sl_tp_data=sl_tp_data,
        )

    @staticmethod
    def _apply_event_to_replay(
        event: TradeEvent,
        *,
        cash: float,
        commission_rate: float,
        open_lots_by_id: dict[str, dict[str, float | str]],
        open_ids_by_symbol: dict[str, list[str]],
    ) -> float:
        event_symbol = event.symbol
        event_qty = float(event.quantity)
        event_price = float(event.price)

        if event.action == "buy":
            cost = event_price * event_qty
            cash -= cost + cost * commission_rate
            open_lots_by_id[event.position_id] = {
                "symbol": event_symbol,
                "quantity": event_qty,
            }
            open_ids_by_symbol[event_symbol].append(event.position_id)
            return cash

        if event.action in {
            "sell",
            "manual",
            "stop_loss",
            "take_profit",
            "end_of_data",
        }:
            remaining = event_qty

            if event.position_id in open_lots_by_id:
                lot = open_lots_by_id[event.position_id]
                lot_qty = float(lot["quantity"])
                sold = min(lot_qty, remaining)
                remaining -= sold
                lot_qty -= sold
                cash += sold * event_price * (1 - commission_rate)
                if lot_qty <= 0:
                    del open_lots_by_id[event.position_id]
                    if event_symbol in open_ids_by_symbol:
                        open_ids_by_symbol[event_symbol] = [
                            pid
                            for pid in open_ids_by_symbol[event_symbol]
                            if pid != event.position_id
                        ]
                else:
                    lot["quantity"] = lot_qty

            if remaining > 0 and event_symbol in open_ids_by_symbol:
                while remaining > 0 and open_ids_by_symbol[event_symbol]:
                    lot_id = open_ids_by_symbol[event_symbol][0]
                    lot = open_lots_by_id.get(lot_id)
                    if lot is None:
                        open_ids_by_symbol[event_symbol].pop(0)
                        continue

                    lot_qty = float(lot["quantity"])
                    sold = min(lot_qty, remaining)
                    remaining -= sold
                    lot_qty -= sold
                    cash += sold * event_price * (1 - commission_rate)

                    if lot_qty <= 0:
                        del open_lots_by_id[lot_id]
                        open_ids_by_symbol[event_symbol].pop(0)
                    else:
                        lot["quantity"] = lot_qty

        return cash

    def build_equity_curve(
        self,
        historical_data: "dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]",
        *,
        primary_resolution: Optional[str] = None,
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
        events: Optional[list[TradeEvent]] = None,
        update_state: bool = True,
    ) -> list[tuple[datetime, float]]:
        """
        Dựng lại ``equity_curve`` từ dữ liệu lịch sử và lịch sử giao dịch.

        Hàm này replay ``events`` theo timeline của ``historical_data`` rồi
        mark-to-market danh mục ở mỗi timestamp.
        """
        multi_data, resolved_primary = self._normalize_multi_data(
            historical_data,
            primary_resolution,
        )
        multi_data = self._prepare_data(multi_data)
        primary_data = multi_data[resolved_primary]
        all_timestamps = self._collect_timestamps(primary_data, start, end)

        if not all_timestamps:
            if update_state:
                self.equity_curve = []
            return []

        events_to_replay = sorted(
            list(self.events if events is None else events),
            key=lambda e: e.timestamp,
        )

        open_lots_by_id: dict[str, dict[str, float | str]] = {}
        open_ids_by_symbol: dict[str, list[str]] = defaultdict(list)
        replay_cash = float(self.initial_cash)
        replay_curve: list[tuple[datetime, float]] = []
        event_idx = 0

        for ts in all_timestamps:
            dt = self._to_dt(ts)

            while event_idx < len(events_to_replay):
                event = events_to_replay[event_idx]
                if pd.Timestamp(event.timestamp) > ts:
                    break

                replay_cash = self._apply_event_to_replay(
                    event,
                    cash=replay_cash,
                    commission_rate=self.commission_rate,
                    open_lots_by_id=open_lots_by_id,
                    open_ids_by_symbol=open_ids_by_symbol,
                )
                event_idx += 1

            prices = self._prices_at(primary_data, ts)
            market_value = 0.0
            for lot in open_lots_by_id.values():
                symbol = str(lot["symbol"])
                quantity = float(lot["quantity"])
                market_value += prices.get(symbol, 0.0) * quantity

            replay_curve.append((dt, replay_cash + market_value))

        if update_state:
            self.equity_curve = replay_curve

        self.last_primary_resolution = resolved_primary
        self.last_timestamp = replay_curve[-1][0]
        self.last_symbols = list(primary_data.keys())

        return replay_curve

    def save(self, path: str | Path) -> Path:
        """Lưu toàn bộ trạng thái engine để có thể load và chạy tiếp."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "initial_cash": self.initial_cash,
            "commission_rate": self.commission_rate,
            "settlement_days": self.settlement_days,
            "auto_manage_sl_tp": self.auto_manage_sl_tp,
            "lookback_days": self.lookback_days,
            "portfolio": self.portfolio,
            "events": self.events,
            "equity_curve": self.equity_curve,
            "last_timestamp": self.last_timestamp,
            "last_primary_resolution": self.last_primary_resolution,
            "last_symbols": self.last_symbols,
            "started": self.started,
            "bot_name": self.bot_name,
        }

        with out_path.open("wb") as f:
            pickle.dump(payload, f)

        return out_path

    @classmethod
    def load(cls, path: str | Path) -> InteractiveEngine:
        """Load engine state từ file pickle đã lưu bởi ``save``."""
        in_path = Path(path)
        with in_path.open("rb") as f:
            payload = pickle.load(f)

        engine = cls(
            initial_cash=payload["initial_cash"],
            commission_rate=payload["commission_rate"],
            settlement_days=payload["settlement_days"],
            auto_manage_sl_tp=payload["auto_manage_sl_tp"],
            lookback_days=payload.get("lookback_days", 365),
        )

        engine.portfolio = payload["portfolio"]
        engine.events = payload.get("events", [])
        engine.equity_curve = payload.get("equity_curve", [])
        engine.last_timestamp = payload.get("last_timestamp")
        engine.last_primary_resolution = payload.get("last_primary_resolution")
        engine.last_symbols = payload.get("last_symbols", [])
        engine.started = payload.get("started", False)
        engine.bot_name = payload.get("bot_name")

        return engine


LiveEngine = InteractiveEngine
LiveEngine = InteractiveEngine
