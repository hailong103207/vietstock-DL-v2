"""Signal – cấu trúc tín hiệu thống nhất cho analysis module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from vnstock_forecast.builtin.signal_based.visualization.snapshot import (
        SignalSnapshot,
    )


class SignalDirection(Enum):
    """Hướng tín hiệu phân tích."""

    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class TradePlan:
    """
    Kế hoạch giao dịch đi kèm tín hiệu BUY.

    Attributes:
        entry:       Giá vào lệnh đề xuất.
        stop_loss:   Giá cắt lỗ.
        take_profit: Giá chốt lời.
    """

    entry: float
    stop_loss: float
    take_profit: float
    max_holding_days: Optional[int] = None
    """
    Số ngày giao dịch tối đa giữ lệnh.
    Sau khoảng thời gian này, engine tự động đóng vị thế để tránh
    chi phí cơ hội. ``None`` = không giới hạn.
    """

    def risk_percent(self) -> float:
        """Phần trăm rủi ro (khoảng cách entry → SL)."""
        if self.entry == 0:
            return 0.0
        return abs(self.entry - self.stop_loss) / self.entry * 100

    def reward_percent(self) -> float:
        """Phần trăm lợi nhuận kỳ vọng (khoảng cách entry → TP)."""
        if self.entry == 0:
            return 0.0
        return abs(self.take_profit - self.entry) / self.entry * 100

    def rr_ratio(self) -> float:
        """Tỷ lệ reward / risk."""
        risk = abs(self.entry - self.stop_loss)
        if risk == 0:
            return 0.0
        reward = abs(self.take_profit - self.entry)
        return reward / risk

    def is_expired(self, entry_time: datetime, current_time: datetime) -> bool:
        """
        Kiểm tra vị thế đã vượt ``max_holding_days`` ngày giao dịch chưa.

        Trả về ``False`` nếu ``max_holding_days`` là ``None``.
        """
        if self.max_holding_days is None:
            return False
        from vnstock_forecast.engine.backtest.portfolio import _business_days_between

        held = _business_days_between(entry_time.date(), current_time.date())
        return held >= self.max_holding_days


@dataclass(slots=True)
class Signal:
    """
    Tín hiệu phân tích kỹ thuật.

    Mỗi technique trả về ``list[Signal]``. Bot sử dụng danh sách này
    để quyết định hành động (mua / bán).

    Attributes:
        technique:   Tên kỹ thuật phân tích tạo ra tín hiệu.
        symbol:      Mã cổ phiếu / tài sản.
        direction:   BUY hoặc SELL.
        timestamp:   Thời điểm phát hiện tín hiệu.
        trade_plan:  Kế hoạch giao dịch (bắt buộc với BUY).
        confidence:  Độ tin cậy [0.0, 1.0]. Mặc định 0.5.
        reason:      Mô tả lý do tín hiệu.
        tags:        Nhãn tùy chọn (vd: ``{"divergence", "oversold"}``).
        metadata:    Dữ liệu bổ sung tùy ý.
        signal_id:   ID duy nhất của signal để trace với actions.
        action_ids:  Danh sách action IDs liên quan đến signal.
        snapshot:    Snapshot dữ liệu trực quan (OHLCV + overlays) để plot.
    """

    technique: str
    symbol: str
    direction: SignalDirection
    timestamp: datetime
    trade_plan: Optional[TradePlan] = None
    confidence: float = 0.5
    reason: str = ""
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid4().hex[:12])
    action_ids: list[str] = field(default_factory=list)
    snapshot: Optional[SignalSnapshot] = None

    def __post_init__(self) -> None:
        """Validate business rules."""
        if self.direction == SignalDirection.BUY and self.trade_plan is None:
            raise ValueError(
                "Tín hiệu BUY bắt buộc phải có TradePlan (entry / SL / TP)."
            )
        if not self.signal_id:
            self.signal_id = uuid4().hex[:12]
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.metadata.setdefault("signal_id", self.signal_id)

    def attach_action(self, action_id: str) -> None:
        """Gắn một ``action_id`` vào signal (không duplicate)."""
        if action_id and action_id not in self.action_ids:
            self.action_ids.append(action_id)

    @property
    def is_buy(self) -> bool:
        return self.direction == SignalDirection.BUY

    @property
    def is_sell(self) -> bool:
        return self.direction == SignalDirection.SELL
