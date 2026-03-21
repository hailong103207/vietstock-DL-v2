"""SignalBasedBacktestBot – bot tổ hợp nhiều technique, tự động phân tích & ra Action."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from vnstock_forecast.builtin.signal_based.profile import SignalProfile
from vnstock_forecast.builtin.signal_based.signal import Signal, SignalDirection
from vnstock_forecast.engine.backtest.bot_base import Action, ActionType, BotBase
from vnstock_forecast.engine.backtest.context import StepContext
from vnstock_forecast.engine.shared.user_bridge import resolve_profile_dir

from .base import BaseTechnique

logger = logging.getLogger(__name__)


class SignalBasedBacktestBot(BotBase):
    """
    Bot phân tích kỹ thuật – tổ hợp N technique thành 1 bot.

    Mỗi bar (``on_step``), bot sẽ:

    1. Gọi ``analyze_step()`` của từng technique cho từng symbol.
    2. Tổng hợp tất cả Signal thu được.
    3. Lọc Signal qua ``accept_signal()`` (customizable).
    4. Chuyển Signal đã lọc thành ``Action`` (BUY/SELL).

    Có thể tùy chỉnh:

    - ``accept_signal(signal, ctx)`` – override để lọc tín hiệu theo
      confidence, profile, hoặc logic tùy ý. Mặc định chấp nhận tất cả.
    - ``allocation`` – phần trăm vốn dùng cho mỗi lệnh mua (0.0–1.0).
    - ``profiles`` – dict ``{technique_name: SignalProfile}`` nạp từ local.
      Nếu có profile, bot sẽ gắn confidence từ profile vào Signal.

    Ngưỡng confidence được cấu hình trên từng technique (``technique.min_confidence``)
    thay vì trên bot, để mỗi technique trong cùng một bot có thể có ngưỡng riêng.

    Example::

        bot = SignalBasedBacktestBot(
            name="RSI_MACD_Combo",
            techniques=[RSICrossover(period=14), MACDCrossover()],
            allocation=0.3,
        )

        # Tùy chỉnh logic lọc
        class SmartBot(SignalBasedBacktestBot):
            def accept_signal(self, signal, ctx):
                # Chỉ chấp nhận BUY khi confidence > 0.6
                if signal.is_buy and signal.confidence < 0.6:
                    return False
                return True
    """

    def __init__(
        self,
        name: str = "SignalBasedBacktestBot",
        description: str = "Bot tổ hợp techniques",
        techniques: Optional[list[BaseTechnique]] = None,
        allocation: float = 0.1,
        sl_pct: float = 0.07,
        tp_pct: float = 0.10,
        profiles: Optional[dict[str, SignalProfile]] = None,
        emit_sl_tp_signals: bool = False,
    ) -> None:
        """
        Args:
            name:        Tên bot.
            description: Mô tả bot.
            techniques:  Danh sách technique instances.
            allocation:  Phần trăm vốn cho mỗi lệnh mua.
            sl_pct:      Stop loss mặc định (%) nếu Signal không có TradePlan.
            tp_pct:      Take profit mặc định (%) nếu Signal không có TradePlan.
            profiles:    Dict profile đã load. ``None`` = không dùng profile.
            emit_sl_tp_signals:
                         ``True``: bot tự phát SELL signal khi chạm SL/TP
                         (dùng khi engine tắt auto_manage_sl_tp).

        Note:
            Ngưỡng confidence được cấu hình trực tiếp trên từng technique
            qua thuộc tính ``min_confidence`` của ``BaseTechnique``, thay vì
            trên bot. Điều này cho phép mỗi technique có ngưỡng riêng khi
            dùng nhiều technique trong cùng một bot.
        """
        self.name = name
        self.description = description
        self.techniques: list[BaseTechnique] = techniques or []
        self.allocation = allocation
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.profiles = profiles or {}
        self.emit_sl_tp_signals = emit_sl_tp_signals

        # Lịch sử signal (ghi lại để profiler phân tích sau)
        self.signal_history: list[Signal] = []
        self.action_history: list[Action] = []

        # Mapping 2 chiều để trace signal ↔ action
        self.signal_to_action_ids: dict[str, list[str]] = {}
        self.action_to_signal_id: dict[str, str] = {}
        self.signal_index: dict[str, Signal] = {}
        self.action_index: dict[str, Action] = {}

        # Theo dõi signal BUY đang mở theo symbol (để nối với SELL sau này)
        self._active_entry_signal_by_symbol: dict[str, str] = {}

        # Đồng bộ lookback của từng technique để snapshot luôn đủ dữ liệu.
        self.sync_technique_lookbacks()

    # ------------------------------------------------------------------
    #  Technique management
    # ------------------------------------------------------------------

    def add_technique(self, technique: BaseTechnique) -> None:
        """Thêm technique vào bot."""
        self.techniques.append(technique)
        self._sync_snapshot_lookback_for_technique(technique)

    def max_required_lookback(self) -> int:
        """Lấy required_lookback lớn nhất trong tất cả techniques."""
        if not self.techniques:
            return 1
        return max(
            int(getattr(t, "required_lookback", 1) or 1) for t in self.techniques
        )

    def sync_technique_lookbacks(self) -> None:
        """
        Đồng bộ ``snapshot_lookback`` theo ``required_lookback`` cho từng technique.

        Quy tắc:
        - ``snapshot_lookback`` của mỗi technique luôn >= 1.
        - Gán trực tiếp bằng ``required_lookback`` của chính technique đó.
        """
        for technique in self.techniques:
            self._sync_snapshot_lookback_for_technique(technique)

    @staticmethod
    def _sync_snapshot_lookback_for_technique(technique: BaseTechnique) -> None:
        required = int(getattr(technique, "required_lookback", 1) or 1)
        technique.snapshot_lookback = max(1, required)

    def load_profiles(self, directory: str | Path | None = None) -> None:
        """
        Nạp tất cả profile từ thư mục.

        Args:
            directory: Thư mục chứa các file ``*.json`` profile.
                       ``None`` = tự resolve theo user-first precedence.
        """
        resolved_dir = resolve_profile_dir(directory)
        self.profiles = SignalProfile.load_all(resolved_dir)
        logger.info(
            "Đã nạp %d profiles: %s",
            len(self.profiles),
            list(self.profiles.keys()),
        )

    # ------------------------------------------------------------------
    #  Bot lifecycle (BotBase interface)
    # ------------------------------------------------------------------

    def on_step(self, ctx: StepContext) -> list[Action]:
        """
        Gọi mỗi bar. Pipeline: analyze → filter → convert to Action.

        Returns:
            Danh sách Action (BUY/SELL) cho engine thực thi.
        """
        # 1) Thu thập signals từ tất cả techniques
        all_signals = self._collect_signals(ctx)

        # 1.1) Tạo SELL signal từ SL/TP nếu bật chế độ bot tự quản trị thoát lệnh
        if self.emit_sl_tp_signals:
            all_signals.extend(self._collect_sl_tp_exit_signals(ctx))

        # 2) Gắn confidence từ profile (nếu có)
        all_signals = self._enrich_with_profiles(all_signals)

        # 3) Lọc signals
        accepted = [s for s in all_signals if self.accept_signal(s, ctx)]

        # 4) Chuyển thành Actions (signal_history được ghi bên trong)
        return self._signals_to_actions(accepted, ctx)

    def on_end(self, ctx: StepContext) -> None:
        """Đồng bộ dữ liệu thoát lệnh vào signal/snapshot từ portfolio cuối kỳ."""
        portfolio = getattr(ctx, "_portfolio", None)
        if portfolio is None:
            return

        for pos in portfolio.closed_positions:
            self._sync_exit_from_closed_position(pos)

    # ------------------------------------------------------------------
    #  Customizable hooks
    # ------------------------------------------------------------------

    def accept_signal(self, signal: Signal, ctx: StepContext) -> bool:
        """
        Quyết định có chấp nhận signal không.

        Override để thêm logic lọc tùy ý (dựa trên confidence, profile,
        trạng thái portfolio, v.v.).

        Mặc định:
        - Chỉ chấp nhận SELL khi đang có vị thế của symbol đó.

        Note:
            Lọc theo ``min_confidence`` đã được thực hiện per-technique
            trong ``_collect_signals`` – không cần kiểm tra lại ở đây.
        """
        if signal.is_sell and not ctx.has_sellable_position(signal.symbol):
            return False

        return True

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _collect_sl_tp_exit_signals(self, ctx: StepContext) -> list[Signal]:
        """Phát SELL signal cho symbol có vị thế sellable chạm SL/TP ở bar hiện tại."""
        signals: list[Signal] = []

        portfolio = getattr(ctx, "_portfolio", None)
        settlement_days = getattr(portfolio, "settlement_days", 0)

        # Chỉ duyệt các vị thế đang mở thay vì quét toàn bộ symbols mỗi bar.
        # touched_by_symbol: symbol -> {position_id -> {reason, price}}
        touched_by_symbol: dict[str, dict[str, dict[str, Any]]] = {}
        bar_cache: dict[str, tuple[float, float]] = {}

        for pos in ctx.positions:
            if not pos.can_sell(ctx.timestamp, settlement_days):
                continue

            symbol = pos.symbol_
            if symbol not in bar_cache:
                try:
                    bar = ctx.latest(symbol)
                except ValueError:
                    continue
                bar_cache[symbol] = (float(bar["High"]), float(bar["Low"]))

            high, low = bar_cache[symbol]

            if pos.stop_loss is not None and low <= float(pos.stop_loss):
                touched_by_symbol.setdefault(symbol, {})[pos.id] = {
                    "reason": "stop_loss",
                    "price": float(pos.stop_loss),
                }
            elif pos.take_profit is not None and high >= float(pos.take_profit):
                touched_by_symbol.setdefault(symbol, {})[pos.id] = {
                    "reason": "take_profit",
                    "price": float(pos.take_profit),
                }

        for symbol, touched in touched_by_symbol.items():
            has_sl = any(info["reason"] == "stop_loss" for info in touched.values())
            exit_reason = "stop_loss" if has_sl else "take_profit"
            signal = Signal(
                technique="risk_manager",
                symbol=symbol,
                direction=SignalDirection.SELL,
                timestamp=ctx.timestamp,
                confidence=1.0,
                reason=f"Auto {exit_reason} at bar {ctx.timestamp}",
                metadata={
                    "source": "sl_tp_monitor",
                    "exit_reason": exit_reason,
                    "touched_positions": touched,
                },
            )
            signals.append(signal)

        return signals

    def _collect_signals(self, ctx: StepContext) -> list[Signal]:
        """Gọi analyze_step() cho mọi technique × mọi symbol.

        Sau khi thu thập, lọc ngay các signal có confidence thấp hơn
        ngưỡng ``min_confidence`` của technique tương ứng.
        """
        signals: list[Signal] = []

        for technique in self.techniques:
            for symbol in ctx.symbols:
                try:
                    result = technique.analyze_step(ctx, symbol)
                    # Lọc theo ngưỡng của chính technique phát ra signal
                    result = [
                        s for s in result if s.confidence >= technique.min_confidence
                    ]
                    signals.extend(result)
                except Exception as exc:
                    logger.warning(
                        "[%s] %s.analyze_step(%s) lỗi: %s",
                        ctx.timestamp,
                        technique.name,
                        symbol,
                        exc,
                    )

        return signals

    def _enrich_with_profiles(self, signals: list[Signal]) -> list[Signal]:
        """Gắn confidence từ profile nếu có."""
        if not self.profiles:
            return signals

        for signal in signals:
            profile = self.profiles.get(signal.technique)
            if profile is None:
                continue

            # Lấy win_rate của direction tương ứng làm confidence
            if signal.is_buy and profile.buy_stats.total_signals > 0:
                signal.confidence = profile.buy_stats.win_rate
            elif signal.is_sell and profile.sell_stats.total_signals > 0:
                signal.confidence = profile.sell_stats.win_rate

        return signals

    def _signals_to_actions(
        self, signals: list[Signal], ctx: StepContext
    ) -> list[Action]:
        """Chuyển danh sách Signal đã lọc thành Action.

        Mỗi symbol chỉ được xử lý 1 lần BUY và 1 lần SELL trong cùng bar
        để tránh duplicate actions khi nhiều technique cùng phát signal.
        """
        actions: list[Action] = []
        handled_buy: set[str] = set()
        handled_sell: set[str] = set()

        for signal in signals:
            self._register_signal(signal)

            if signal.is_buy:
                if signal.symbol in handled_buy:
                    continue
                action = self._buy_action(signal, ctx)
                if action is not None:
                    actions.append(action)
                    handled_buy.add(signal.symbol)
                    self.action_history.append(action)
                    self.signal_history.append(signal)

            elif signal.is_sell:
                if signal.symbol in handled_sell:
                    continue
                sell_actions = self._sell_actions(signal, ctx)
                if sell_actions:
                    actions.extend(sell_actions)
                    handled_sell.add(signal.symbol)
                    self.action_history.extend(sell_actions)
                    self.signal_history.append(signal)

        return actions

    def _buy_action(self, signal: Signal, ctx: StepContext) -> Optional[Action]:
        """Chuyển BUY signal thành Action."""
        # Không mua nếu đã có vị thế
        if ctx.has_position(signal.symbol):
            return None

        price = ctx.price(signal.symbol)

        # Lấy SL/TP từ TradePlan hoặc mặc định
        if signal.trade_plan:
            sl = signal.trade_plan.stop_loss
            tp = signal.trade_plan.take_profit
        else:
            sl = round(price * (1 - self.sl_pct), 2)
            tp = round(price * (1 + self.tp_pct), 2)

        # Lượng tiền vào lệnh tỷ lệ với confidence:
        # capital_to_use = cash * allocation * confidence
        # confidence cao → vào nhiều hơn, confidence thấp → vào ít hơn.
        effective_allocation = self.allocation * signal.confidence
        qty = int(ctx.cash * effective_allocation // price)
        if qty <= 0:
            return None

        action_id = self._new_action_id()

        signal.metadata["entry_time"] = ctx.timestamp
        signal.metadata["entry_price"] = price
        if signal.snapshot is not None:
            signal.snapshot.entry_time = ctx.timestamp
            if signal.snapshot.entry is None:
                signal.snapshot.entry = (
                    signal.trade_plan.entry if signal.trade_plan is not None else price
                )

        action = Action(
            type=ActionType.BUY,
            symbol=signal.symbol,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            reason=self._compose_reason(signal, action_id),
        )
        self._link_action_to_signal(action, action_id, signal)
        self._active_entry_signal_by_symbol[signal.symbol] = signal.signal_id
        return action

    def _sell_actions(self, signal: Signal, ctx: StepContext) -> list[Action]:
        """Chuyển SELL signal thành danh sách Action (FIFO, chỉ bán lô đã qua T+N)."""
        sellable = ctx.sellable_positions(signal.symbol)
        if not sellable:
            return []

        is_sl_tp_monitor_signal = (
            isinstance(signal.metadata, dict)
            and signal.metadata.get("source") == "sl_tp_monitor"
        )
        touched_positions: dict[str, dict[str, Any]] = {}
        if is_sl_tp_monitor_signal:
            raw_touched = signal.metadata.get("touched_positions")
            if isinstance(raw_touched, dict):
                touched_positions = {
                    str(pid): info
                    for pid, info in raw_touched.items()
                    if isinstance(info, dict)
                }

        signal.metadata["exit_time"] = ctx.timestamp
        signal.metadata["exit_price"] = ctx.price(signal.symbol)
        if signal.snapshot is not None:
            signal.snapshot.exit_time = ctx.timestamp
            signal.snapshot.exit_price = float(ctx.price(signal.symbol))

        entry_signal_id = self._active_entry_signal_by_symbol.get(signal.symbol)
        if entry_signal_id:
            signal.metadata["entry_signal_id"] = entry_signal_id

        # FIFO: sắp xếp theo thời gian mua tăng dần, bán lô cũ nhất trước
        sellable.sort(key=lambda p: p.entry_time)

        sell_actions: list[Action] = []
        for pos in sellable:
            if is_sl_tp_monitor_signal and pos.id not in touched_positions:
                continue

            action_price: Optional[float] = None
            if is_sl_tp_monitor_signal:
                info = touched_positions.get(pos.id, {})
                raw_price = info.get("price") if isinstance(info, dict) else None
                if isinstance(raw_price, (int, float)):
                    action_price = float(raw_price)

            action_id = self._new_action_id()
            action = Action(
                type=ActionType.SELL,
                symbol=signal.symbol,
                quantity=pos.quantity,
                price=action_price,
                position_id=pos.id,
                reason=self._compose_reason(signal, action_id),
            )
            self._link_action_to_signal(
                action,
                action_id,
                signal,
                entry_signal_id=entry_signal_id,
            )
            sell_actions.append(action)

        if entry_signal_id and entry_signal_id in self.signal_index:
            entry_signal = self.signal_index[entry_signal_id]
            entry_signal.metadata["exit_time"] = ctx.timestamp
            entry_signal.metadata["exit_price"] = ctx.price(signal.symbol)
            entry_signal.metadata["exit_signal_id"] = signal.signal_id
            if entry_signal.snapshot is not None:
                entry_signal.snapshot.exit_time = ctx.timestamp
                entry_signal.snapshot.exit_price = float(ctx.price(signal.symbol))
            for action in sell_actions:
                action_id = self._action_id_of(action)
                if action_id:
                    entry_signal.attach_action(action_id)
                    self.signal_to_action_ids.setdefault(entry_signal.signal_id, [])
                    if (
                        action_id
                        not in self.signal_to_action_ids[entry_signal.signal_id]
                    ):
                        self.signal_to_action_ids[entry_signal.signal_id].append(
                            action_id
                        )

        # Đã phát SELL để đóng vị thế symbol này
        self._active_entry_signal_by_symbol.pop(signal.symbol, None)
        return sell_actions

    def _register_signal(self, signal: Signal) -> None:
        self.signal_index[signal.signal_id] = signal
        signal.metadata.setdefault("signal_id", signal.signal_id)
        signal.metadata.setdefault("action_ids", signal.action_ids)
        self.signal_to_action_ids.setdefault(signal.signal_id, list(signal.action_ids))

    def _sync_exit_from_closed_position(self, pos: Any) -> None:
        """Map vị thế đã đóng về signal BUY gốc và cập nhật exit metadata/snapshot."""
        if pos.exit_time is None or pos.exit_price is None:
            return

        entry_action = getattr(pos, "entry_action", None)
        if entry_action is None:
            return

        action_metadata = getattr(entry_action, "metadata", None)
        if not isinstance(action_metadata, dict):
            return

        signal_id = action_metadata.get("signal_id")
        if not isinstance(signal_id, str):
            return

        signal = self.signal_index.get(signal_id)
        if signal is None:
            return

        signal.metadata["exit_time"] = pos.exit_time
        signal.metadata["exit_price"] = float(pos.exit_price)
        if signal.snapshot is not None:
            signal.snapshot.exit_time = pos.exit_time
            signal.snapshot.exit_price = float(pos.exit_price)

    @staticmethod
    def _new_action_id() -> str:
        return uuid4().hex[:12]

    @staticmethod
    def _action_id_of(action: Action) -> Optional[str]:
        metadata = getattr(action, "metadata", None)
        if isinstance(metadata, dict):
            action_id = metadata.get("action_id")
            if isinstance(action_id, str):
                return action_id
        return None

    @staticmethod
    def _compose_reason(signal: Signal, action_id: str) -> str:
        base_reason = f"[{signal.technique}] {signal.reason}".strip()
        return (
            f"{base_reason} | sid={signal.signal_id} aid={action_id}"
            if base_reason
            else f"sid={signal.signal_id} aid={action_id}"
        )

    def _link_action_to_signal(
        self,
        action: Action,
        action_id: str,
        signal: Signal,
        entry_signal_id: Optional[str] = None,
    ) -> None:
        signal.attach_action(action_id)

        self.signal_to_action_ids.setdefault(signal.signal_id, [])
        if action_id not in self.signal_to_action_ids[signal.signal_id]:
            self.signal_to_action_ids[signal.signal_id].append(action_id)
        self.action_to_signal_id[action_id] = signal.signal_id

        action_metadata: dict[str, Any] = {
            "action_id": action_id,
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "technique": signal.technique,
        }
        if entry_signal_id:
            action_metadata["entry_signal_id"] = entry_signal_id

        setattr(action, "metadata", action_metadata)
        self.action_index[action_id] = action

    def signal_for_action(self, action_id: str) -> Optional[Signal]:
        """Signal trực tiếp phát ra ``action_id`` này."""
        signal_id = self.action_to_signal_id.get(action_id)
        if signal_id is None:
            return None
        return self.signal_index.get(signal_id)

    def entry_signal_for_action(self, action_id: str) -> Optional[Signal]:
        """Signal BUY gốc liên quan action SELL (nếu có)."""
        action = self.action_index.get(action_id)
        if action is None:
            return None
        metadata = getattr(action, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        entry_signal_id = metadata.get("entry_signal_id")
        if not isinstance(entry_signal_id, str):
            return None
        return self.signal_index.get(entry_signal_id)

    def actions_for_signal(self, signal_id: str) -> list[Action]:
        """Tất cả actions liên kết với ``signal_id``."""
        action_ids = self.signal_to_action_ids.get(signal_id, [])
        return [
            self.action_index[aid] for aid in action_ids if aid in self.action_index
        ]
