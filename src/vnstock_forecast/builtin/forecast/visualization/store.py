"""Signal store – lưu trữ & query signals đã accept bằng pickle."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from vnstock_forecast.builtin.forecast.signal import Signal, SignalDirection

logger = logging.getLogger(__name__)


class SignalStore:
    """Persistent store cho các signals đã được accept.

    Mỗi signal được lưu thành 1 file pickle riêng trong *base_dir*.
    Tên file mã hoá ``technique``, ``symbol``, ``timestamp`` và một
    UUID-stub để tiện lọc nhanh qua filename trước khi deserialize.

    Usage::

        store = SignalStore("signals")
        sid = store.save(signal)

        # Truy vấn
        buys = store.query(symbol="VNM", direction=SignalDirection.BUY)

        # Load theo ID
        sig = store.load(sid)

        # Plot trực tiếp
        from vnstock_forecast.builtin.forecast.visualization import plot_signal
        plot_signal(sig)
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Save / Load
    # ------------------------------------------------------------------

    def save(self, signal: Signal) -> str:
        """Lưu *signal* và trả về signal_id (tên file không phần mở rộng).

        Args:
            signal: ``Signal`` cần lưu (nên có ``snapshot`` đính kèm).

        Returns:
            ``signal_id`` – dùng để ``load()`` hoặc nhận diện sau này.
        """
        ts_str = signal.timestamp.strftime("%Y%m%d_%H%M%S")
        uid = uuid4().hex[:8]
        signal_id = f"{signal.technique}__{signal.symbol}__{ts_str}__{uid}"
        path = self.base_dir / f"{signal_id}.pkl"

        with open(path, "wb") as f:
            pickle.dump(signal, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Đã lưu signal %s → %s", signal_id, path)
        return signal_id

    def save_many(self, signals: list[Signal]) -> list[str]:
        """Lưu nhiều signals, trả về danh sách IDs."""
        return [self.save(s) for s in signals]

    def load(self, signal_id: str) -> Signal:
        """Load signal theo *signal_id*.

        Raises:
            FileNotFoundError: Không tìm thấy file.
        """
        path = self.base_dir / f"{signal_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy signal: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    #  Query
    # ------------------------------------------------------------------

    def query(
        self,
        technique: Optional[str] = None,
        symbol: Optional[str] = None,
        direction: Optional[SignalDirection] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Signal]:
        """Truy vấn signals theo bộ lọc.

        Lọc nhanh qua filename trước rồi mới deserialize & lọc chi tiết.

        Args:
            technique: Tên technique (substring match trên filename).
            symbol:    Mã cổ phiếu (substring match trên filename).
            direction: ``SignalDirection.BUY`` hoặc ``SELL``.
            from_time: Bắt đầu khoảng thời gian.
            to_time:   Kết thúc khoảng thời gian.
            limit:     Số lượng tối đa trả về.

        Returns:
            Danh sách ``Signal`` thoả mãn filters, sắp xếp theo timestamp.
        """
        results: list[Signal] = []

        for path in sorted(self.base_dir.glob("*.pkl")):
            name = path.stem

            # --- fast filename filter ---
            if technique and technique not in name:
                continue
            if symbol and symbol not in name:
                continue

            try:
                signal = self._load_path(path)
            except Exception as exc:
                logger.warning("Bỏ qua %s – lỗi load: %s", path.name, exc)
                continue

            # --- detailed filter ---
            if direction is not None and signal.direction != direction:
                continue
            if from_time is not None and signal.timestamp < from_time:
                continue
            if to_time is not None and signal.timestamp > to_time:
                continue

            results.append(signal)
            if limit is not None and len(results) >= limit:
                break

        return results

    def list_ids(self) -> list[str]:
        """Liệt kê tất cả signal IDs đã lưu."""
        return sorted(p.stem for p in self.base_dir.glob("*.pkl"))

    def count(self) -> int:
        """Tổng số signals đã lưu."""
        return sum(1 for _ in self.base_dir.glob("*.pkl"))

    def delete(self, signal_id: str) -> None:
        """Xoá signal theo ID."""
        path = self.base_dir / f"{signal_id}.pkl"
        if path.exists():
            path.unlink()
            logger.info("Đã xoá signal %s", signal_id)

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_path(path: Path) -> Signal:
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
