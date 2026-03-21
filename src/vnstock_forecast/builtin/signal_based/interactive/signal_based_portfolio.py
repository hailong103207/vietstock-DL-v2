"""Signal-based portfolio cho interactive workflow."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from vnstock_forecast.builtin.signal_based.signal import Signal, SignalDirection
from vnstock_forecast.engine.backtest.bot_base import Action
from vnstock_forecast.engine.backtest.portfolio import (
    CloseReason,
    _business_days_between,
)


class PositionStatus(Enum):
    """Lifecycle status của position."""

    OPEN = "open"
    CLOSED = "closed"


class PositionActionType(Enum):
    """Loại action trong lifecycle position."""

    BUY = "buy"
    SELL = "sell"
    UPDATE_RISK = "update_risk"


@dataclass(slots=True)
class ExecutionMarker:
    """Marker cho một lần thực thi action trên timeline."""

    timestamp: datetime
    price: float
    quantity: float
    action_id: str
    kind: PositionActionType


@dataclass(slots=True)
class LinkedSignalRef:
    """Signal metadata đã chuẩn hóa, luôn tồn tại cho mỗi action/position."""

    signal_id: str
    symbol: str
    direction: SignalDirection
    technique: str = "manual"
    reason: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionActionRecord:
    """Action record để trace signal ↔ position."""

    action_type: PositionActionType
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    position_id: Optional[str]
    signal_id: str
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(slots=True)
class PositionSnapshotRecord:
    """Snapshot 1-1 với position để visualize toàn lifecycle."""

    position_id: str
    symbol: str
    resolution: str
    entry: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    buy_markers: list[ExecutionMarker] = field(default_factory=list)
    sell_markers: list[ExecutionMarker] = field(default_factory=list)
    signal_id: str = ""
    technique: str = "manual"
    signal_reason: str = ""
    signal_overlays: dict[str, Any] = field(default_factory=dict)
    context_from_ts: Optional[int] = None
    context_to_ts: Optional[int] = None
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def indicators(self) -> list[Any]:
        return list(self.signal_overlays.get("indicators", []))

    @property
    def hlines(self) -> list[Any]:
        return list(self.signal_overlays.get("hlines", []))

    @property
    def vlines(self) -> list[Any]:
        return list(self.signal_overlays.get("vlines", []))

    @property
    def rectangles(self) -> list[Any]:
        return list(self.signal_overlays.get("rectangles", []))

    @property
    def trendlines(self) -> list[Any]:
        return list(self.signal_overlays.get("trendlines", []))

    def append_marker(self, marker: ExecutionMarker) -> None:
        if marker.kind == PositionActionType.BUY:
            self.buy_markers.append(marker)
        elif marker.kind == PositionActionType.SELL:
            self.sell_markers.append(marker)
        self.updated_at = datetime.now()


@dataclass(slots=True)
class SignalBasedPosition:
    """Position record có liên kết signal và snapshot."""

    id: str
    symbol: str
    entry_price: float
    initial_quantity: float
    open_quantity: float
    entry_time: datetime
    signal_id: str
    snapshot_id: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_holding_days: Optional[int] = None
    status: PositionStatus = PositionStatus.OPEN
    close_reason: Optional[CloseReason] = None
    entry_action: Optional[Action] = None
    exit_action: Optional[Action] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    action_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def quantity(self) -> float:
        return self.open_quantity if self.is_open else self.initial_quantity

    @property
    def cost(self) -> float:
        return self.entry_price * self.initial_quantity

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.initial_quantity

    def market_value(self, current_price: float) -> float:
        return current_price * self.open_quantity

    def can_sell(self, current_time: datetime, settlement_days: int) -> bool:
        if settlement_days <= 0:
            return True
        return (
            _business_days_between(self.entry_time.date(), current_time.date())
            >= settlement_days
        )

    def is_time_expired(self, current_time: datetime) -> bool:
        if self.max_holding_days is None:
            return False
        held = _business_days_between(self.entry_time.date(), current_time.date())
        return held >= self.max_holding_days


class SignalBasedPortfolio:
    """Portfolio quản lý position/action/snapshot cho interactive signal flow."""

    def __init__(
        self,
        initial_cash: float = 100_000_000.0,
        commission_rate: float = 0.0015,
        settlement_days: int = 3,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission_rate = float(commission_rate)
        self.settlement_days = int(settlement_days)

        self._open: dict[str, SignalBasedPosition] = {}
        self._closed: list[SignalBasedPosition] = []
        self._actions: dict[str, PositionActionRecord] = {}
        self._snapshots: dict[str, PositionSnapshotRecord] = {}

    @staticmethod
    def _default_stop_loss(entry: float) -> float:
        return entry * 0.95

    @staticmethod
    def _default_take_profit(entry: float) -> float:
        return entry * 1.10

    @staticmethod
    def _signal_overlays(signal: Optional[Signal]) -> dict[str, Any]:
        snapshot = None if signal is None else signal.snapshot
        if snapshot is None:
            return {}
        return {
            "has_snapshot": True,
            "indicator_count": len(snapshot.indicators),
            "hline_count": len(snapshot.hlines),
            "vline_count": len(snapshot.vlines),
            "rectangle_count": len(snapshot.rectangles),
            "trendline_count": len(snapshot.trendlines),
            "indicators": copy.deepcopy(list(snapshot.indicators)),
            "hlines": copy.deepcopy(list(snapshot.hlines)),
            "vlines": copy.deepcopy(list(snapshot.vlines)),
            "rectangles": copy.deepcopy(list(snapshot.rectangles)),
            "trendlines": copy.deepcopy(list(snapshot.trendlines)),
        }

    @staticmethod
    def _build_signal_ref(
        symbol: str,
        direction: SignalDirection,
        signal: Optional[Signal] = None,
        *,
        reason: str = "",
    ) -> LinkedSignalRef:
        if signal is None:
            return LinkedSignalRef(
                signal_id=f"manual-{uuid.uuid4().hex[:12]}",
                symbol=symbol,
                direction=direction,
                technique="manual",
                reason=reason,
                confidence=1.0,
                metadata={"manual": True},
            )
        return LinkedSignalRef(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            technique=signal.technique,
            reason=signal.reason,
            confidence=signal.confidence,
            metadata=dict(signal.metadata),
        )

    @staticmethod
    def _clone_for_closed_leg(
        position: SignalBasedPosition,
        *,
        sold_quantity: float,
        exit_price: float,
        exit_time: datetime,
        close_reason: CloseReason,
        exit_action: Optional[Action],
    ) -> SignalBasedPosition:
        return SignalBasedPosition(
            id=position.id,
            symbol=position.symbol,
            entry_price=position.entry_price,
            initial_quantity=sold_quantity,
            open_quantity=0.0,
            entry_time=position.entry_time,
            signal_id=position.signal_id,
            snapshot_id=position.snapshot_id,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            max_holding_days=position.max_holding_days,
            status=PositionStatus.CLOSED,
            close_reason=close_reason,
            entry_action=position.entry_action,
            exit_action=exit_action,
            exit_price=exit_price,
            exit_time=exit_time,
            action_ids=list(position.action_ids),
            created_at=position.created_at,
            updated_at=datetime.now(),
        )

    def _record_action(
        self,
        *,
        action_type: PositionActionType,
        symbol: str,
        price: float,
        quantity: float,
        timestamp: datetime,
        signal_ref: LinkedSignalRef,
        position_id: Optional[str],
        note: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> PositionActionRecord:
        action = PositionActionRecord(
            action_type=action_type,
            symbol=symbol,
            price=float(price),
            quantity=float(quantity),
            timestamp=timestamp,
            position_id=position_id,
            signal_id=signal_ref.signal_id,
            note=note,
            metadata=dict(metadata or {}),
        )
        self._actions[action.action_id] = action
        return action

    def open_position(
        self,
        action: Action,
        timestamp: datetime,
        signal: Optional[Signal] = None,
        *,
        resolution: str = "D",
        context_from_ts: Optional[int] = None,
        context_to_ts: Optional[int] = None,
    ) -> SignalBasedPosition:
        assert action.price is not None, "Price phải được resolve trước khi mở lệnh"
        if action.quantity <= 0:
            raise ValueError("quantity phải > 0")

        cost = float(action.price) * float(action.quantity)
        commission = cost * self.commission_rate
        total_cost = cost + commission
        if total_cost > self.cash:
            raise ValueError(
                f"Không đủ tiền: cần {total_cost:,.0f}, có {self.cash:,.0f}"
            )

        signal_ref = self._build_signal_ref(
            action.symbol,
            SignalDirection.BUY,
            signal,
            reason=action.reason,
        )

        plan = None if signal is None else signal.trade_plan
        stop_loss = (
            action.stop_loss
            if action.stop_loss is not None
            else (
                plan.stop_loss
                if plan is not None
                else self._default_stop_loss(action.price)
            )
        )
        take_profit = (
            action.take_profit
            if action.take_profit is not None
            else (
                plan.take_profit
                if plan is not None
                else self._default_take_profit(action.price)
            )
        )

        position_id = uuid.uuid4().hex[:8]
        position = SignalBasedPosition(
            id=position_id,
            symbol=action.symbol,
            entry_price=float(action.price),
            initial_quantity=float(action.quantity),
            open_quantity=float(action.quantity),
            entry_time=timestamp,
            signal_id=signal_ref.signal_id,
            snapshot_id=uuid.uuid4().hex[:12],
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            max_holding_days=action.max_holding_days,
            entry_action=action,
        )

        buy_action = self._record_action(
            action_type=PositionActionType.BUY,
            symbol=position.symbol,
            price=position.entry_price,
            quantity=position.initial_quantity,
            timestamp=timestamp,
            signal_ref=signal_ref,
            position_id=position.id,
            note=action.reason,
            metadata={"resolution": resolution},
        )
        position.action_ids.append(buy_action.action_id)

        snapshot = PositionSnapshotRecord(
            snapshot_id=position.snapshot_id,
            position_id=position.id,
            symbol=position.symbol,
            resolution=resolution,
            entry=position.entry_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            entry_time=timestamp,
            signal_id=signal_ref.signal_id,
            technique=signal_ref.technique,
            signal_reason=signal_ref.reason,
            signal_overlays=self._signal_overlays(signal),
            context_from_ts=context_from_ts,
            context_to_ts=context_to_ts,
        )
        snapshot.append_marker(
            ExecutionMarker(
                timestamp=timestamp,
                price=position.entry_price,
                quantity=position.initial_quantity,
                action_id=buy_action.action_id,
                kind=PositionActionType.BUY,
            )
        )

        self._open[position.id] = position
        self._snapshots[position.id] = snapshot
        self.cash -= total_cost
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        reason: CloseReason,
        exit_action: Optional[Action] = None,
        quantity: Optional[float] = None,
        signal: Optional[Signal] = None,
    ) -> SignalBasedPosition:
        if position_id not in self._open:
            raise KeyError(f"Position '{position_id}' không tồn tại hoặc đã đóng")

        pos = self._open[position_id]
        if reason != CloseReason.END_OF_DATA and not pos.can_sell(
            exit_time,
            self.settlement_days,
        ):
            held = _business_days_between(pos.entry_time.date(), exit_time.date())
            raise ValueError(
                f"T+{self.settlement_days}: '{pos.symbol}' mua ngày "
                f"{pos.entry_time.date()}, mới T+{held}, "
                f"chưa đủ {self.settlement_days} ngày giao dịch để bán."
            )

        sell_qty = quantity if (quantity and quantity > 0) else pos.open_quantity
        if sell_qty <= 0:
            raise ValueError("quantity bán phải > 0")
        if sell_qty > pos.open_quantity:
            raise ValueError(
                f"Số lượng bán ({sell_qty:,.0f}) vượt quá vị thế "
                f"({pos.open_quantity:,.0f}) của '{pos.symbol}'."
            )

        signal_ref = self._build_signal_ref(
            pos.symbol,
            SignalDirection.SELL,
            signal,
            reason=(exit_action.reason if exit_action else ""),
        )

        sell_action = self._record_action(
            action_type=PositionActionType.SELL,
            symbol=pos.symbol,
            price=exit_price,
            quantity=sell_qty,
            timestamp=exit_time,
            signal_ref=signal_ref,
            position_id=pos.id,
            note=(exit_action.reason if exit_action else reason.value),
            metadata={"close_reason": reason.value},
        )

        pos.action_ids.append(sell_action.action_id)
        pos.updated_at = datetime.now()

        snapshot = self._snapshots[pos.id]
        snapshot.append_marker(
            ExecutionMarker(
                timestamp=exit_time,
                price=exit_price,
                quantity=sell_qty,
                action_id=sell_action.action_id,
                kind=PositionActionType.SELL,
            )
        )

        proceeds = exit_price * sell_qty
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission

        closed_leg = self._clone_for_closed_leg(
            pos,
            sold_quantity=sell_qty,
            exit_price=exit_price,
            exit_time=exit_time,
            close_reason=reason,
            exit_action=exit_action,
        )

        remaining = pos.open_quantity - sell_qty
        if remaining <= 0:
            self._open.pop(position_id)
            pos.open_quantity = 0.0
            pos.status = PositionStatus.CLOSED
        else:
            pos.open_quantity = remaining
            pos.status = PositionStatus.OPEN

        self._closed.append(closed_leg)
        return closed_leg

    def update_position_risk(
        self,
        position_id: str,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        note: str = "",
        signal: Optional[Signal] = None,
    ) -> SignalBasedPosition:
        if position_id not in self._open:
            raise KeyError(f"Position '{position_id}' không tồn tại hoặc đã đóng")
        if stop_loss is None and take_profit is None:
            raise ValueError("Cần ít nhất một trong stop_loss/take_profit")

        pos = self._open[position_id]
        ts = timestamp or datetime.now()
        if stop_loss is not None:
            pos.stop_loss = float(stop_loss)
        if take_profit is not None:
            pos.take_profit = float(take_profit)
        pos.updated_at = ts

        snapshot = self._snapshots[position_id]
        if pos.stop_loss is not None:
            snapshot.stop_loss = pos.stop_loss
        if pos.take_profit is not None:
            snapshot.take_profit = pos.take_profit
        snapshot.updated_at = ts

        signal_ref = self._build_signal_ref(
            pos.symbol,
            SignalDirection.SELL,
            signal,
            reason=note,
        )
        risk_action = self._record_action(
            action_type=PositionActionType.UPDATE_RISK,
            symbol=pos.symbol,
            price=pos.entry_price,
            quantity=0.0,
            timestamp=ts,
            signal_ref=signal_ref,
            position_id=pos.id,
            note=note,
            metadata={
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            },
        )
        pos.action_ids.append(risk_action.action_id)
        return pos

    def check_sl_tp(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> list[SignalBasedPosition]:
        closed: list[SignalBasedPosition] = []

        for pos in list(self._open.values()):
            if pos.symbol != symbol:
                continue
            if not pos.can_sell(timestamp, self.settlement_days):
                continue

            if pos.stop_loss is not None and low <= pos.stop_loss:
                closed.append(
                    self.close_position(
                        pos.id,
                        float(pos.stop_loss),
                        timestamp,
                        CloseReason.STOP_LOSS,
                    )
                )
            elif pos.take_profit is not None and high >= pos.take_profit:
                closed.append(
                    self.close_position(
                        pos.id,
                        float(pos.take_profit),
                        timestamp,
                        CloseReason.TAKE_PROFIT,
                    )
                )
            elif pos.is_time_expired(timestamp):
                closed.append(
                    self.close_position(
                        pos.id,
                        float(close),
                        timestamp,
                        CloseReason.TIME_LIMIT,
                    )
                )

        return closed

    @property
    def open_positions(self) -> list[SignalBasedPosition]:
        return list(self._open.values())

    @property
    def closed_positions(self) -> list[SignalBasedPosition]:
        return list(self._closed)

    @property
    def actions(self) -> list[PositionActionRecord]:
        return sorted(self._actions.values(), key=lambda x: x.timestamp)

    def list_actions(
        self,
        *,
        position_id: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> list[PositionActionRecord]:
        result = self.actions
        if position_id is not None:
            result = [a for a in result if a.position_id == position_id]
        if signal_id is not None:
            result = [a for a in result if a.signal_id == signal_id]
        return result

    def get_snapshot(self, position_id: str) -> PositionSnapshotRecord:
        if position_id not in self._snapshots:
            raise KeyError(f"Không tìm thấy snapshot cho position '{position_id}'")
        return self._snapshots[position_id]

    def get_open_position(self, position_id: str) -> SignalBasedPosition:
        if position_id not in self._open:
            raise KeyError(f"Position '{position_id}' không tồn tại hoặc đã đóng")
        return self._open[position_id]

    def list_snapshots(self) -> list[PositionSnapshotRecord]:
        return sorted(self._snapshots.values(), key=lambda x: x.entry_time)

    def equity(self, current_prices: dict[str, float]) -> float:
        market_val = sum(
            pos.market_value(current_prices.get(pos.symbol, pos.entry_price))
            for pos in self._open.values()
        )
        return self.cash + market_val

    def positions_for(self, symbol: str) -> list[SignalBasedPosition]:
        return [p for p in self._open.values() if p.symbol == symbol]

    def sellable_positions(
        self,
        symbol: str,
        current_time: datetime,
    ) -> list[SignalBasedPosition]:
        return [
            p
            for p in self._open.values()
            if p.symbol == symbol and p.can_sell(current_time, self.settlement_days)
        ]

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self._open.values())

    def has_sellable_position(self, symbol: str, current_time: datetime) -> bool:
        return any(
            p.symbol == symbol and p.can_sell(current_time, self.settlement_days)
            for p in self._open.values()
        )
