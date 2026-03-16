"""
Profiler – chạy backtest hàng loạt cho mọi technique đã đăng ký,
tính toán SignalProfile, lưu ra local.

Usage::

    from vnstock_forecast.builtin.forecast.profiler import Profiler

    profiler = Profiler()
    profiles = profiler.run(
        data={"VNM": df_vnm, "VHM": df_vhm},
        start="2023-01-01",
        end="2024-12-31",
    )

    # In kết quả
    for name, profile in profiles.items():
        profile.print_summary()

    # Lưu xuống local
    profiler.save()

    # Hoặc profile technique cụ thể
    from vnstock_forecast.builtin.forecast.strategies import RSICrossover
    profiler.run_single(
        technique=RSICrossover(period=14),
        data={"VNM": df_vnm},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd

from vnstock_forecast.builtin.forecast.technical.bot import AnalysisBot  # noqa: F401
from vnstock_forecast.engine.backtest.bot_base import Action  # noqa: F401
from vnstock_forecast.engine.backtest.bot_base import ActionType  # noqa: F401
from vnstock_forecast.engine.backtest.context import StepContext  # noqa: F401
from vnstock_forecast.engine.backtest.engine import BacktestEngine
from vnstock_forecast.engine.backtest.portfolio import CloseReason
from vnstock_forecast.engine.backtest.report import BacktestReport
from vnstock_forecast.engine.shared.user_bridge import resolve_profile_dir

from .profile import DirectionStats, SignalProfile
from .registry import get_all_techniques
from .signal import Signal, SignalDirection
from .visualization.pdf_report import PDFProfileReport

logger = logging.getLogger(__name__)

# Thư mục lưu profile mặc định
DEFAULT_PROFILE_DIR = resolve_profile_dir(create_if_missing=True)


class TechniqueLike(Protocol):
    """Minimal protocol for registered techniques used by profiler."""

    name: str
    attach_snapshot: bool

    def analyze_step(self, ctx: StepContext, symbol: str) -> list[Signal]: ...


class _ProfilerTechniqueBot(AnalysisBot):
    """Internal bot wrapper used by profiler for one technique."""

    def __init__(
        self,
        technique: TechniqueLike,
        name: str,
        allocation: float = 0.1,
        sl_pct: float = 0.07,
        tp_pct: float = 0.10,
    ) -> None:
        super().__init__(
            name=name,
            description="Profiler internal single-technique bot",
            techniques=[technique],
            allocation=allocation,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            profiles=None,
        )
        self.technique = technique

    def on_step(self, ctx: StepContext) -> list[Action]:
        return super().on_step(ctx)


@dataclass
class ProfileResult:
    """Kết quả profiling cho 1 technique, bao gồm tất cả dữ liệu cần cho PDF."""

    profile: SignalProfile
    report: BacktestReport
    signals: list[Signal]
    technique: Any
    bot: AnalysisBot


class Profiler:
    """
    Chạy backtest cho từng technique đã đăng ký, tính toán SignalProfile.

    Quy trình cho mỗi technique:

    1. Tạo ``AnalysisBot`` wrapper chỉ chứa 1 technique.
    2. Chạy ``BacktestEngine.run()`` để backtest.
    3. Phân tích signal_history của bot + kết quả backtest.
    4. Tính ``DirectionStats`` cho BUY và SELL.
    5. Tạo ``SignalProfile`` và lưu vào dict.

    Attributes:
        profiles:     Dict kết quả: ``{technique_name: SignalProfile}``.
        results:      Dict kết quả chi tiết: ``{technique_name: ProfileResult}``.
        profile_dir:  Thư mục lưu JSON profile.
    """

    def __init__(
        self,
        profile_dir: str | Path = DEFAULT_PROFILE_DIR,
        initial_cash: float = 100_000_000.0,
        commission_rate: float = 0.0015,
        settlement_days: int = 3,
    ) -> None:
        self.profile_dir = Path(profile_dir)
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.settlement_days = settlement_days
        self.profiles: dict[str, SignalProfile] = {}
        self.results: dict[str, ProfileResult] = {}

    # ==================================================================
    #  Public API
    # ==================================================================

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start: Optional[str] = None,
        end: Optional[str] = None,
        techniques: Optional[list[Any]] = None,
    ) -> dict[str, SignalProfile]:
        """
        Chạy profiling cho tất cả technique (hoặc danh sách chỉ định).

        Args:
            data:       ``{symbol: DataFrame}`` OHLCV data.
            start:      Ngày bắt đầu backtest.
            end:        Ngày kết thúc backtest.
            techniques: Danh sách technique instances cần profile.
                        ``None`` = lấy tất cả từ registry.

        Returns:
            Dict ``{technique_name: SignalProfile}``.
        """
        if techniques is None:
            registry = get_all_techniques()
            if not registry:
                logger.warning("Registry trống – không có technique nào để profile.")
                return {}
            techniques = [cls() for cls in registry.values()]

        logger.info(
            "Bắt đầu profiling %d technique(s): %s",
            len(techniques),
            [t.name for t in techniques],
        )

        for technique in techniques:
            try:
                result = self._run_single_full(technique, data, start, end)
                self.profiles[technique.name] = result.profile
                self.results[technique.name] = result
                logger.info(
                    "✓ %s – win_rate=%.1f%%",
                    technique.name,
                    result.profile.overall_win_rate * 100,
                )
            except Exception as exc:
                logger.error("✗ %s – lỗi: %s", technique.name, exc)

        return self.profiles

    def run_single(
        self,
        technique: Any,
        data: dict[str, pd.DataFrame],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> SignalProfile:
        """
        Chạy profiling cho 1 technique.

        Args:
            technique: Instance của BaseTechnique.
            data:      OHLCV data.
            start:     Ngày bắt đầu.
            end:       Ngày kết thúc.

        Returns:
            ``SignalProfile`` đã tính toán xong.
        """
        result = self._run_single_full(technique, data, start, end)
        return result.profile

    def _run_single_full(
        self,
        technique: Any,
        data: dict[str, pd.DataFrame],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> ProfileResult:
        """
        Chạy profiling đầy đủ cho 1 technique, trả về toàn bộ kết quả.

        Returns:
            ``ProfileResult`` chứa profile, report, signals, technique, bot.
        """
        # Bật attach_snapshot để mọi signal đều có dữ liệu plot
        technique.attach_snapshot = True

        # Tạo bot wrapper chỉ chứa 1 technique, allocation cao để
        # đảm bảo đủ tiền thực thi nhiều signals
        bot = _ProfilerTechniqueBot(
            technique=technique,
            name=f"Profiler_{technique.name}",
            allocation=0.3,
        )

        # Chạy backtest
        engine = BacktestEngine(
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
            settlement_days=self.settlement_days,
        )
        report = engine.run(bot=bot, data=data, start=start, end=end)

        # Tính profile từ signal_history và backtest report
        profile = self._compute_profile(
            technique=technique,
            bot=bot,
            report=report,
            data=data,
        )

        return ProfileResult(
            profile=profile,
            report=report,
            signals=list(bot.signal_history),
            technique=technique,
            bot=bot,
        )

    def save(
        self,
        directory: Optional[str | Path] = None,
        pdf: bool = True,
        benchmark_data: Optional[pd.DataFrame] = None,
        max_signal_charts: Optional[int] = 25,
    ) -> None:
        """
        Lưu tất cả profile ra JSON files + PDF reports.

        Args:
            directory:         Thư mục đích. ``None`` = dùng ``self.profile_dir``.
            pdf:               Có tạo PDF report không. Mặc định ``True``.
            benchmark_data:    DataFrame OHLCV benchmark (VNINDEX) để tính
                               alpha/beta trong PDF. ``None`` = bỏ qua.
            max_signal_charts: Số chart signal tối đa trong mỗi PDF.
                               ``None`` = không giới hạn. Mặc định ``50``.
        """
        save_dir = Path(directory) if directory else self.profile_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, profile in self.profiles.items():
            # JSON
            filepath = save_dir / f"{name}.json"
            profile.save(filepath)

            # PDF
            if pdf and name in self.results:
                try:
                    self.save_pdf(
                        name,
                        directory=save_dir,
                        benchmark_data=benchmark_data,
                        max_signal_charts=max_signal_charts,
                    )
                except Exception as exc:
                    logger.error("Không thể tạo PDF cho %s: %s", name, exc)

        logger.info("Đã lưu %d profile(s) vào %s", len(self.profiles), save_dir)

    def save_pdf(
        self,
        technique_name: str,
        directory: Optional[str | Path] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        max_signal_charts: Optional[int] = 50,
    ) -> Path:
        """
        Tạo PDF report cho 1 technique.

        Args:
            technique_name:    Tên technique (key trong ``self.results``).
            directory:         Thư mục đích. ``None`` = dùng ``self.profile_dir``.
            benchmark_data:    DataFrame OHLCV benchmark (VNINDEX).
            max_signal_charts: Số chart signal tối đa. ``None`` = không giới hạn.
                               Mặc định ``50``.

        Returns:
            ``Path`` tới file PDF đã tạo.

        Raises:
            KeyError: ``technique_name`` không có trong results.
        """
        if technique_name not in self.results:
            raise KeyError(
                f"Không tìm thấy kết quả cho '{technique_name}'. "
                f"Hãy chạy profiling trước."
            )

        result = self.results[technique_name]
        save_dir = Path(directory) if directory else self.profile_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = save_dir / f"{technique_name}.pdf"

        pdf_report = PDFProfileReport(
            technique_name=result.technique.name,
            technique_params=result.technique.params,
            description=result.bot.description,
            backtest_report=result.report,
            signal_profile=result.profile,
            signals=result.signals,
            benchmark_data=benchmark_data,
            max_signal_charts=max_signal_charts,
        )

        return pdf_report.generate(pdf_path)

    def load(self, directory: Optional[str | Path] = None) -> dict[str, SignalProfile]:
        """
        Đọc tất cả profile từ thư mục.

        Args:
            directory: Thư mục chứa JSON. ``None`` = dùng ``self.profile_dir``.

        Returns:
            Dict ``{technique_name: SignalProfile}``.
        """
        load_dir = Path(directory) if directory else self.profile_dir
        self.profiles = SignalProfile.load_all(load_dir)
        return self.profiles

    # ==================================================================
    #  Internal
    # ==================================================================

    def _compute_profile(
        self,
        technique: Any,
        bot: AnalysisBot,
        report: BacktestReport,
        data: dict[str, pd.DataFrame],
    ) -> SignalProfile:
        """Tính toán SignalProfile từ kết quả backtest."""
        signals = bot.signal_history
        summary = report.summary()
        closed = report.portfolio.closed_positions

        # Tính tổng bars
        total_bars = len(report.equity_curve) if report.equity_curve else 0

        # Phân loại signals theo direction
        buy_signals = [s for s in signals if s.is_buy]
        sell_signals = [s for s in signals if s.is_sell]

        # Phân loại closed positions theo entry action reason
        # (mỗi position tương ứng với 1 BUY signal)
        buy_stats = self._compute_direction_stats(
            buy_signals, closed, total_bars, SignalDirection.BUY
        )
        sell_stats = self._compute_direction_stats(
            sell_signals, closed, total_bars, SignalDirection.SELL
        )

        return SignalProfile(
            technique_name=technique.name,
            technique_params=technique.params,
            symbols=report.symbols,
            period=f"{report.start.date()} → {report.end.date()}",
            total_bars=total_bars,
            buy_stats=buy_stats,
            sell_stats=sell_stats,
            backtest_summary=summary,
            created_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _compute_direction_stats(
        signals: list[Signal],
        closed_positions: list,
        total_bars: int,
        direction: SignalDirection,
    ) -> DirectionStats:
        """Tính DirectionStats cho 1 hướng (BUY hoặc SELL)."""
        total = len(signals)
        if total == 0:
            return DirectionStats()

        # Lọc closed positions thực tế (không tính end_of_data)
        real_trades = [
            p for p in closed_positions if p.close_reason != CloseReason.END_OF_DATA
        ]

        if direction == SignalDirection.BUY:
            # BUY signals → tra cứu kết quả qua closed positions
            # Mỗi BUY tạo 1 position → kết quả = position.pnl
            wins = [p for p in real_trades if (p.pnl or 0) > 0]
            losses = [p for p in real_trades if (p.pnl or 0) <= 0]

            win_count = len(wins)
            loss_count = len(losses)
            trade_count = win_count + loss_count

            win_rate = win_count / trade_count if trade_count > 0 else 0.0

            avg_return = (
                sum(p.pnl_percent or 0 for p in real_trades) / trade_count
                if trade_count > 0
                else 0.0
            )
            avg_win = sum(p.pnl_percent or 0 for p in wins) / len(wins) if wins else 0.0
            avg_loss = (
                sum(p.pnl_percent or 0 for p in losses) / len(losses) if losses else 0.0
            )
            rr = abs(avg_win / avg_loss) if avg_loss else 0.0

        else:
            # SELL signals – đánh giá bằng giá sau khi bán có giảm không
            # (Đơn giản: tính từ closed positions do MANUAL close)
            manual_closes = [
                p for p in closed_positions if p.close_reason == CloseReason.MANUAL
            ]
            # SELL thành công nếu giá đóng < giá vào (tức đã bán đúng lúc)
            wins = [p for p in manual_closes if (p.pnl or 0) > 0]
            losses = [p for p in manual_closes if (p.pnl or 0) <= 0]

            win_count = len(wins)
            loss_count = len(losses)
            trade_count = win_count + loss_count

            win_rate = win_count / trade_count if trade_count > 0 else 0.0
            avg_return = avg_win = avg_loss = rr = 0.0

            if wins:
                avg_win = sum(p.pnl_percent or 0 for p in wins) / len(wins)
            if losses:
                avg_loss = sum(p.pnl_percent or 0 for p in losses) / len(losses)
            if trade_count > 0:
                avg_return = (
                    sum(p.pnl_percent or 0 for p in manual_closes) / trade_count
                )
            rr = abs(avg_win / avg_loss) if avg_loss else 0.0

        frequency = total / total_bars if total_bars > 0 else 0.0

        return DirectionStats(
            total_signals=total,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=round(win_rate, 4),
            avg_return_pct=round(avg_return, 4),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            risk_reward=round(rr, 4),
            frequency=round(frequency, 6),
        )
