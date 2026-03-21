"""
Backtest Module
===============

Chạy backtest cho mọi bot trên mọi khung thời gian. Chỉ cần kế thừa
``BotBase``, implement ``on_step()`` → truyền vào ``BacktestEngine``.

Quick start::

    from vnstock_forecast.backtest import (
        BacktestEngine, BotBase, Action, ActionType,
    )

    class MyBot(BotBase):
        name = "MyBot"

        def on_step(self, ctx):
            # ctx.cash       → tiền mặt
            # ctx.positions  → sổ lệnh (vị thế đang mở)
            # ctx.equity     → tổng tài sản
            # ctx.price(sym) → giá Close hiện tại
            # ctx.history(sym, lookback=N) → N bars gần nhất
            return []  # list[Action]

    engine = BacktestEngine(initial_cash=100_000_000)
    report = engine.run(
        bot=MyBot(),
        data={"VNM": df_ohlcv},        # {symbol: DataFrame}
        start="2023-01-01",
        end="2024-12-31",
    )
    report.print_summary()

Tương thích live trading: interface ``BotBase.on_step(ctx)`` hoàn toàn
giống nhau – chỉ cần thay nguồn dữ liệu real-time phía sau
``StepContext``.
"""

from .backtest import BacktestEngine
from .bot_base import Action, ActionType, BotBase
from .context import StepContext
from .core import EngineCore
from .manual_bot import ManualBot
from .portfolio import CloseReason, Portfolio, Position, TradeEvent
from .report import BacktestReport

__all__ = [
    "BacktestEngine",
    "EngineCore",
    "BotBase",
    "Action",
    "ActionType",
    "StepContext",
    "Portfolio",
    "Position",
    "CloseReason",
    "TradeEvent",
    "BacktestReport",
    "ManualBot",
]
