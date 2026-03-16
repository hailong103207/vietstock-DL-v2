"""SignalProfile – hồ sơ độ tin cậy tín hiệu sau backtest."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DirectionStats:
    """
    Thống kê cho một hướng tín hiệu (BUY hoặc SELL).

    Attributes:
        total_signals:    Tổng số tín hiệu phát ra.
        win_count:        Số tín hiệu thắng (PnL > 0).
        loss_count:       Số tín hiệu thua (PnL ≤ 0).
        win_rate:         Tỷ lệ thắng (0.0 – 1.0).
        avg_return_pct:   Lợi nhuận trung bình mỗi tín hiệu (%).
        avg_win_pct:      Lợi nhuận trung bình khi thắng (%).
        avg_loss_pct:     Thua lỗ trung bình khi thua (%).
        risk_reward:      Tỷ lệ reward / risk trung bình.
        frequency:        Tần suất tín hiệu (signals / tổng bars).
    """

    total_signals: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    risk_reward: float = 0.0
    frequency: float = 0.0


@dataclass
class SignalProfile:
    """
    Hồ sơ đánh giá kỹ thuật phân tích, tính toán sau khi backtest.

    Được tạo bởi ``Profiler`` và lưu vào local (JSON).

    Attributes:
        technique_name:  Tên technique (trùng với registry name).
        technique_params: Dict tham số kỹ thuật lúc backtest.
        symbols:         Danh sách symbols đã test.
        period:          Khoảng thời gian backtest (start → end).
        total_bars:      Tổng số bars trong backtest.
        buy_stats:       Thống kê tín hiệu BUY.
        sell_stats:      Thống kê tín hiệu SELL.
        backtest_summary: Tóm tắt kết quả backtest (từ BacktestReport).
        created_at:      Thời gian tạo profile.
    """

    technique_name: str
    technique_params: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    period: str = ""
    total_bars: int = 0
    buy_stats: DirectionStats = field(default_factory=DirectionStats)
    sell_stats: DirectionStats = field(default_factory=DirectionStats)
    backtest_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    #  Tổng hợp
    # ------------------------------------------------------------------

    @property
    def overall_win_rate(self) -> float:
        """Tỷ lệ thắng tổng thể (BUY + SELL)."""
        total = self.buy_stats.total_signals + self.sell_stats.total_signals
        if total == 0:
            return 0.0
        wins = self.buy_stats.win_count + self.sell_stats.win_count
        return wins / total

    @property
    def total_signals(self) -> int:
        """Tổng số tín hiệu đã phát."""
        return self.buy_stats.total_signals + self.sell_stats.total_signals

    # ------------------------------------------------------------------
    #  Serialize / Deserialize
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict (JSON-serializable)."""
        d = asdict(self)
        # Chuyển DirectionStats sang dict
        d["buy_stats"] = asdict(self.buy_stats)
        d["sell_stats"] = asdict(self.sell_stats)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalProfile:
        """Tạo SignalProfile từ dict."""
        buy_data = data.pop("buy_stats", {})
        sell_data = data.pop("sell_stats", {})
        profile = cls(**data)
        profile.buy_stats = DirectionStats(**buy_data)
        profile.sell_stats = DirectionStats(**sell_data)
        return profile

    def save(self, path: str | Path) -> None:
        """
        Lưu profile ra file JSON.

        Args:
            path: Đường dẫn file JSON. Tự tạo thư mục nếu chưa có.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.info("Đã lưu profile: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> SignalProfile:
        """
        Đọc profile từ file JSON.

        Args:
            path: Đường dẫn file JSON.

        Returns:
            ``SignalProfile`` instance.

        Raises:
            FileNotFoundError: File không tồn tại.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def load_all(directory: str | Path) -> dict[str, "SignalProfile"]:
        """
        Đọc tất cả profile từ thư mục.

        Args:
            directory: Thư mục chứa các file ``.json``.

        Returns:
            Dict ``{technique_name: SignalProfile}``.
        """
        directory = Path(directory)
        profiles: dict[str, SignalProfile] = {}
        if not directory.exists():
            return profiles

        for fp in sorted(directory.glob("*.json")):
            try:
                profile = SignalProfile.load(fp)
                profiles[profile.technique_name] = profile
            except Exception as e:
                logger.warning("Không thể đọc profile %s: %s", fp, e)

        return profiles

    # ------------------------------------------------------------------
    #  Display
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """In bảng tóm tắt profile ra console."""
        print("=" * 60)
        print(f"  SIGNAL PROFILE: {self.technique_name}")
        print("=" * 60)
        print(f"  Params:         {self.technique_params}")
        print(f"  Symbols:        {', '.join(self.symbols)}")
        print(f"  Period:         {self.period}")
        print(f"  Total Bars:     {self.total_bars}")
        print(f"  Created:        {self.created_at}")
        print("-" * 60)

        for label, stats in [("BUY", self.buy_stats), ("SELL", self.sell_stats)]:
            print(f"  [{label}]")
            print(f"    Signals:      {stats.total_signals:>8d}")
            print(f"    Wins:         {stats.win_count:>8d}")
            print(f"    Losses:       {stats.loss_count:>8d}")
            print(f"    Win Rate:     {stats.win_rate:>7.1%}")
            print(f"    Avg Return:   {stats.avg_return_pct:>7.2f}%")
            print(f"    Avg Win:      {stats.avg_win_pct:>7.2f}%")
            print(f"    Avg Loss:     {stats.avg_loss_pct:>7.2f}%")
            print(f"    R:R Ratio:    {stats.risk_reward:>8.2f}")
            print(f"    Frequency:    {stats.frequency:>7.4f}")

        print("-" * 60)
        print(f"  Overall Win Rate: {self.overall_win_rate:.1%}")
        print(f"  Total Signals:    {self.total_signals}")
        print("=" * 60)
