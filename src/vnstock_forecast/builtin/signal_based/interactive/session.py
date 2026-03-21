"""Interactive session orchestrator for signal -> action -> position workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from vnstock_forecast.builtin.signal_based.interactive.interactive_engine import (
    InteractiveEngine,
    LiveEngine,
)
from vnstock_forecast.builtin.signal_based.interactive.signal_based_portfolio import (
    PositionActionRecord,
    PositionSnapshotRecord,
    PositionStatus,
    SignalBasedPosition,
)
from vnstock_forecast.builtin.signal_based.registry import get_technique
from vnstock_forecast.builtin.signal_based.signal import Signal, SignalDirection
from vnstock_forecast.builtin.signal_based.technical.base import BaseTechnique
from vnstock_forecast.builtin.signal_based.visualization.snapshot import SignalSnapshot
from vnstock_forecast.engine.backtest.bot_base import Action, ActionType
from vnstock_forecast.engine.data.query import query_grouped_ohlcv

GroupedOHLCV = dict[str, dict[str, pd.DataFrame]]


class InteractiveSignalSession:
    """Công cụ điều phối interactive analysis/execution.

    Trách nhiệm chính:
    - Nhận signal/lệnh tay từ user.
    - Chuẩn hóa thành action records.
    - Tạo/cập nhật position + snapshot theo action.
    - Lưu trữ truy vết đầy đủ signal ↔ action ↔ position ↔ snapshot.
    """

    def __init__(
        self,
        live_engine: Optional[InteractiveEngine] = None,
        base_dir: str | Path = "outputs/interactive/default-session",
        default_resolution: str = "D",
    ) -> None:
        self.live_engine = live_engine or LiveEngine()
        self.base_dir = Path(base_dir)
        self.default_resolution = default_resolution

    @staticmethod
    def _extract_signal_id(signal: Optional[Signal]) -> Optional[str]:
        return None if signal is None else signal.signal_id

    @staticmethod
    def _to_datetime(value: int | float | datetime | None) -> datetime:
        if value is None:
            return datetime.now()
        if isinstance(value, datetime):
            return value
        return datetime.fromtimestamp(int(value))

    @staticmethod
    def _get_price(
        grouped_data: GroupedOHLCV,
        *,
        resolution: str,
        symbol: str,
    ) -> Optional[float]:
        symbol_df = grouped_data.get(resolution, {}).get(symbol, pd.DataFrame())
        if symbol_df.empty:
            return None
        return float(symbol_df.iloc[-1]["Close"])

    def _build_current_prices(
        self,
        symbols: list[str],
        *,
        resolution: str,
        timestamp: datetime,
    ) -> dict[str, float]:
        grouped_data = query_grouped_ohlcv(
            symbols=symbols,
            resolutions=[resolution],
            to_ts=int(timestamp.timestamp()),
            lookback_bars=1,
        )
        prices: dict[str, float] = {}
        for symbol in symbols:
            price = self._get_price(grouped_data, resolution=resolution, symbol=symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    @staticmethod
    def _serialize_signal_overlays(signal: Optional[Signal]) -> dict:
        if signal is None or signal.snapshot is None:
            return {}
        return {
            "has_snapshot": True,
            "indicator_count": len(signal.snapshot.indicators),
            "hline_count": len(signal.snapshot.hlines),
            "vline_count": len(signal.snapshot.vlines),
            "rectangle_count": len(signal.snapshot.rectangles),
            "trendline_count": len(signal.snapshot.trendlines),
        }

    @staticmethod
    def _ensure_list(value: list[str] | str | None, field_name: str) -> list[str]:
        if value is None:
            raise ValueError(f"{field_name} không được để trống")
        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)

        normalized = [item.strip() for item in items if item and item.strip()]
        if not normalized:
            raise ValueError(f"{field_name} không được rỗng")
        return normalized

    @staticmethod
    def _to_unix_ts(value: int | float | datetime, field_name: str) -> int:
        if isinstance(value, datetime):
            return int(value.timestamp())
        if isinstance(value, (int, float)):
            return int(value)
        raise TypeError(f"{field_name} phải là int/float/datetime")

    @staticmethod
    def _normalize_ts_range(
        from_ts: int | float | datetime | None,
        to_ts: int | float | datetime | None,
    ) -> tuple[int, int]:
        now_ts = int(datetime.now().timestamp())
        normalized_to_ts = (
            now_ts
            if to_ts is None
            else InteractiveSignalSession._to_unix_ts(
                to_ts,
                "to_ts",
            )
        )
        normalized_from_ts = (
            normalized_to_ts
            if from_ts is None
            else InteractiveSignalSession._to_unix_ts(from_ts, "from_ts")
        )

        if normalized_from_ts > normalized_to_ts:
            raise ValueError("from_ts không được lớn hơn to_ts")
        return normalized_from_ts, normalized_to_ts

    @staticmethod
    def _validate_techniques(techniques: list[BaseTechnique]) -> list[BaseTechnique]:
        if not techniques:
            raise ValueError("techniques không được rỗng")

        invalid = [tech for tech in techniques if not isinstance(tech, BaseTechnique)]
        if invalid:
            invalid_types = ", ".join(type(item).__name__ for item in invalid)
            raise TypeError(
                "Mọi phần tử trong techniques phải kế thừa BaseTechnique. "
                f"Nhận được: {invalid_types}"
            )
        return techniques

    @staticmethod
    def _get_required_lookback(techniques: list[BaseTechnique]) -> int:
        """Tính lookback bars lớn nhất cho batch analysis."""
        return max(max(int(tech.required_lookback), 0) for tech in techniques)

    @staticmethod
    def _safe_get_df(
        grouped_data: GroupedOHLCV,
        resolution: str,
        symbol: str,
    ) -> pd.DataFrame:
        return grouped_data.get(resolution, {}).get(symbol, pd.DataFrame())

    @staticmethod
    def _merge_ohlcv_frames(
        lookback_df: pd.DataFrame,
        main_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if lookback_df.empty and main_df.empty:
            return pd.DataFrame()
        if lookback_df.empty:
            return main_df.sort_index()
        if main_df.empty:
            return lookback_df.sort_index()

        merged = pd.concat([lookback_df, main_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged.sort_index()

    @staticmethod
    def _signal_timestamp_to_unix_ts(signal: Signal) -> Optional[int]:
        if signal.timestamp is None:
            return None
        if isinstance(signal.timestamp, datetime):
            return int(signal.timestamp.timestamp())
        try:
            return int(signal.timestamp)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_signal_valid(
        signal: Signal,
        from_ts: int,
        to_ts: int,
        min_confidence: float,
    ) -> bool:
        signal_ts = InteractiveSignalSession._signal_timestamp_to_unix_ts(signal)
        if signal_ts is None or signal_ts < from_ts or signal_ts > to_ts:
            return False
        return signal.confidence >= min_confidence

    def get_signal(
        self,
        techniques: list[BaseTechnique],
        symbols: list[str] | str,
        to_ts: int | float | datetime | None = None,
        from_ts: int | float | datetime | None = None,
        resolutions: list[str] | str | None = None,
        min_confidence: float = 0.0,
        attach_snapshot: bool = True,
        snapshot_lookback: Optional[int] = None,
    ) -> list[Signal]:
        """Khung API cho luồng interactive lấy tín hiệu.

        Giai đoạn hiện tại: dùng trên khung ngày
        """
        validated_techniques = self._validate_techniques(techniques)
        normalized_symbols = self._ensure_list(symbols, "symbols")

        if resolutions is None:
            normalized_resolutions = [self.default_resolution]
        else:
            normalized_resolutions = self._ensure_list(resolutions, "resolutions")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence phải nằm trong khoảng [0.0, 1.0]")

        from_ts, to_ts = self._normalize_ts_range(from_ts=from_ts, to_ts=to_ts)

        max_lookback = self._get_required_lookback(validated_techniques)
        lookback_df = query_grouped_ohlcv(
            symbols=normalized_symbols,
            resolutions=normalized_resolutions,
            to_ts=from_ts - 1,
            lookback_bars=max_lookback,
        )
        df = query_grouped_ohlcv(
            symbols=normalized_symbols,
            resolutions=normalized_resolutions,
            from_ts=from_ts,
            to_ts=to_ts,
        )

        merged_grouped_df: GroupedOHLCV = {}
        for resolution in normalized_resolutions:
            merged_grouped_df.setdefault(resolution, {})
            for symbol in normalized_symbols:
                lookback_data = self._safe_get_df(lookback_df, resolution, symbol)
                main_data = self._safe_get_df(df, resolution, symbol)
                merged_grouped_df[resolution][symbol] = self._merge_ohlcv_frames(
                    lookback_data,
                    main_data,
                )

        signals: list[Signal] = []
        original_snapshot_settings: dict[int, tuple[bool, int]] = {
            id(technique): (technique.attach_snapshot, technique.snapshot_lookback)
            for technique in validated_techniques
        }

        try:
            if attach_snapshot:
                for technique in validated_techniques:
                    technique.attach_snapshot = True
                    if snapshot_lookback is not None:
                        technique.snapshot_lookback = int(snapshot_lookback)

            for symbol in normalized_symbols:
                for resolution in normalized_resolutions:
                    symbol_df = merged_grouped_df[resolution][symbol]
                    if symbol_df.empty:
                        continue

                    for technique in validated_techniques:
                        try:
                            current_signals = technique.analyze_batch(symbol_df, symbol)
                        except NotImplementedError as exc:
                            raise NotImplementedError(
                                f"Technique '{technique.name}' chưa hỗ trợ analyze_batch()."
                            ) from exc

                        if not current_signals:
                            continue

                        for sig in current_signals:
                            if not isinstance(sig, Signal):
                                continue
                            if self._is_signal_valid(
                                sig, from_ts, to_ts, min_confidence
                            ):
                                signals.append(sig)
        finally:
            for technique in validated_techniques:
                original_attach, original_lookback = original_snapshot_settings[
                    id(technique)
                ]
                technique.attach_snapshot = original_attach
                technique.snapshot_lookback = original_lookback

        return signals

    def execute_signal(
        self,
        signal: Signal,
        *,
        quantity: float,
        timestamp: int | float | datetime | None = None,
        current_prices: Optional[dict[str, float]] = None,
        primary_resolution: Optional[str] = None,
        position_id: Optional[str] = None,
    ):
        if signal.direction not in {SignalDirection.BUY, SignalDirection.SELL}:
            raise ValueError("Signal direction không hợp lệ")

        dt = self._to_datetime(timestamp)
        resolution = primary_resolution or self.default_resolution
        prices = (
            current_prices
            if current_prices is not None
            else self._build_current_prices(
                [signal.symbol], resolution=resolution, timestamp=dt
            )
        )
        return self.live_engine.execute_signal(
            signal,
            quantity=quantity,
            current_prices=prices,
            timestamp=dt,
            primary_resolution=resolution,
            position_id=position_id,
        )

    def order_buy(
        self,
        *,
        symbol: str,
        quantity: float,
        signal: Optional[Signal] = None,
        timestamp: int | float | datetime | None = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_holding_days: Optional[int] = None,
        reason: str = "",
        primary_resolution: Optional[str] = None,
    ):
        if quantity <= 0:
            raise ValueError("quantity phải > 0")
        if signal is not None and signal.direction != SignalDirection.BUY:
            raise ValueError("order_buy chỉ nhận tín hiệu BUY")

        dt = self._to_datetime(timestamp)
        resolution = primary_resolution or self.default_resolution
        prices = self._build_current_prices(
            [symbol], resolution=resolution, timestamp=dt
        )

        entry_price = price
        if entry_price is None and signal is not None and signal.trade_plan is not None:
            entry_price = signal.trade_plan.entry

        action = Action(
            type=ActionType.BUY,
            symbol=symbol,
            quantity=quantity,
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_days=max_holding_days,
            reason=reason or (signal.reason if signal is not None else ""),
        )
        if signal is not None:
            return self.live_engine.execute_signal(
                signal,
                quantity=quantity,
                current_prices=prices,
                timestamp=dt,
                primary_resolution=resolution,
            )
        return self.live_engine.execute_actions(
            action,
            timestamp=dt,
            current_prices=prices,
            primary_resolution=resolution,
        )

    def order_sell(
        self,
        *,
        symbol: str,
        position_id: Optional[str] = None,
        quantity: Optional[float] = None,
        signal: Optional[Signal] = None,
        timestamp: int | float | datetime | None = None,
        price: Optional[float] = None,
        reason: str = "",
        primary_resolution: Optional[str] = None,
    ):
        if quantity is not None and quantity < 0:
            raise ValueError("quantity không được âm")
        if signal is not None and signal.direction != SignalDirection.SELL:
            raise ValueError("order_sell chỉ nhận tín hiệu SELL")

        dt = self._to_datetime(timestamp)
        resolution = primary_resolution or self.default_resolution
        prices = self._build_current_prices(
            [symbol], resolution=resolution, timestamp=dt
        )

        action = Action(
            type=ActionType.SELL,
            symbol=symbol,
            quantity=float(quantity or 0.0),
            price=price,
            position_id=position_id,
            reason=reason or (signal.reason if signal is not None else ""),
        )
        if signal is not None:
            return self.live_engine.execute_signal(
                signal,
                quantity=float(quantity or 0.0),
                current_prices=prices,
                timestamp=dt,
                primary_resolution=resolution,
                position_id=position_id,
            )
        return self.live_engine.execute_actions(
            action,
            timestamp=dt,
            current_prices=prices,
            primary_resolution=resolution,
        )

    def update_position_risk(
        self,
        *,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        timestamp: int | float | datetime | None = None,
        note: str = "",
        symbol_hint: Optional[str] = None,
        primary_resolution: Optional[str] = None,
    ):
        dt = self._to_datetime(timestamp)
        try:
            position = self.live_engine.portfolio.get_open_position(position_id)
        except KeyError:
            position = None

        symbol = symbol_hint or (position.symbol if position is not None else None)
        if symbol is None:
            raise KeyError(f"Không tìm thấy position '{position_id}'")

        resolution = primary_resolution or self.default_resolution
        prices = self._build_current_prices(
            [symbol], resolution=resolution, timestamp=dt
        )
        return self.live_engine.update_position_risk(
            position_id=position_id,
            timestamp=dt,
            current_prices=prices,
            stop_loss=stop_loss,
            take_profit=take_profit,
            note=note,
            primary_resolution=resolution,
        )

    def list_positions(
        self,
        *,
        symbol: Optional[str] = None,
        status: Optional[PositionStatus] = None,
    ) -> list[SignalBasedPosition]:
        all_positions = (
            self.live_engine.portfolio.open_positions
            + self.live_engine.portfolio.closed_positions
        )
        if symbol is not None:
            all_positions = [p for p in all_positions if p.symbol == symbol]
        if status is not None:
            all_positions = [p for p in all_positions if p.status == status]
        return sorted(all_positions, key=lambda p: p.entry_time)

    def list_actions(
        self,
        *,
        position_id: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> list[PositionActionRecord]:
        return self.live_engine.portfolio.list_actions(
            position_id=position_id,
            signal_id=signal_id,
        )

    def get_snapshot(self, position_id: str) -> PositionSnapshotRecord:
        return self.live_engine.portfolio.get_snapshot(position_id)

    @staticmethod
    def _infer_snapshot_lookback_bars(
        snapshot: PositionSnapshotRecord,
        default_bars: int,
    ) -> int:
        technique_name = (snapshot.technique or "").strip()
        if not technique_name or technique_name.lower() == "manual":
            return default_bars

        try:
            technique_cls = get_technique(technique_name)
            technique = technique_cls()
            return max(int(technique.required_lookback), 1)
        except Exception:
            return default_bars

    def build_plot_snapshot(
        self,
        position_id: str,
        *,
        lookback_bars: Optional[int] = None,
        default_bars_without_signal: int = 30,
        to_ts: int | float | datetime | None = None,
    ) -> SignalSnapshot:
        """Build ``SignalSnapshot`` từ position snapshot đã lưu trong portfolio."""
        position_snapshot = self.get_snapshot(position_id)

        query_to_dt = self._to_datetime(to_ts)
        query_to_ts = int(query_to_dt.timestamp())

        bars = (
            int(lookback_bars)
            if lookback_bars is not None
            else self._infer_snapshot_lookback_bars(
                position_snapshot,
                default_bars=max(int(default_bars_without_signal), 1),
            )
        )
        bars = max(bars, 1)

        grouped = query_grouped_ohlcv(
            symbols=[position_snapshot.symbol],
            resolutions=[position_snapshot.resolution],
            to_ts=query_to_ts,
            lookback_bars=bars,
        )
        ohlcv = (
            grouped.get(position_snapshot.resolution, {})
            .get(
                position_snapshot.symbol,
                pd.DataFrame(),
            )
            .copy()
        )

        if ohlcv.empty:
            marker_times = [
                m.timestamp
                for m in (
                    position_snapshot.buy_markers + position_snapshot.sell_markers
                )
            ]
            marker_times = sorted(set(marker_times)) or [position_snapshot.entry_time]

            base_price = float(position_snapshot.entry)
            rows = [
                {
                    "Timestamp": int(ts.timestamp()),
                    "Open": base_price,
                    "High": base_price * 1.01,
                    "Low": base_price * 0.99,
                    "Close": base_price,
                    "Volume": 1000,
                }
                for ts in marker_times
            ]
            ohlcv = pd.DataFrame(rows).set_index("Timestamp").sort_index()

        ohlcv.index = pd.to_datetime(ohlcv.index, unit="s")

        exit_price = None
        exit_time = None
        if position_snapshot.sell_markers:
            last_sell = sorted(
                position_snapshot.sell_markers,
                key=lambda marker: marker.timestamp,
            )[-1]
            exit_price = float(last_sell.price)
            exit_time = last_sell.timestamp

        buy_points = sorted(
            [
                (marker.timestamp, float(marker.price))
                for marker in position_snapshot.buy_markers
            ],
            key=lambda point: point[0],
        )
        sell_points = sorted(
            [
                (marker.timestamp, float(marker.price))
                for marker in position_snapshot.sell_markers
            ],
            key=lambda point: point[0],
        )

        return SignalSnapshot(
            ohlcv=ohlcv,
            entry=float(position_snapshot.entry),
            entry_time=position_snapshot.entry_time,
            stop_loss=float(position_snapshot.stop_loss),
            take_profit=float(position_snapshot.take_profit),
            exit_price=exit_price,
            exit_time=exit_time,
            buy_points=buy_points,
            sell_points=sell_points,
            signal_time=position_snapshot.entry_time,
            resolution=position_snapshot.resolution,
            symbol=position_snapshot.symbol,
            indicators=list(position_snapshot.indicators),
            hlines=list(position_snapshot.hlines),
            vlines=list(position_snapshot.vlines),
            rectangles=list(position_snapshot.rectangles),
            trendlines=list(position_snapshot.trendlines),
        )

    def save_engine(self, path: str | Path) -> Path:
        """Lưu trạng thái engine hiện tại ra file pickle."""
        return self.live_engine.save(path)

    def load_engine(self, path: str | Path) -> InteractiveEngine:
        """Load engine từ file pickle và thay thế engine hiện tại của session."""
        self.live_engine = LiveEngine.load(path)
        return self.live_engine
