"""PDF Profile Report – xuất báo cáo backtest dưới dạng PDF đầy đủ.

Sử dụng matplotlib PdfPages backend để tạo PDF nhiều trang bao gồm:

1. **Tổng quan** – tên, mô tả, params, backtest summary, equity curve.
2. **Metrics** – thống kê BUY/SELL, alpha/beta vs benchmark, ratios.
3. **Backtest Details** – trade history, event log, signal list.
4. **Signal Charts** – biểu đồ entry/exit cho từng signal có snapshot.

Usage::

    from vnstock_forecast.builtin.signal_based.visualization.pdf_report import PDFProfileReport

    report = PDFProfileReport(
        technique_name="SMA_Crossover",
        technique_params={"fast": 10, "slow": 30},
        description="SMA crossover strategy",
        backtest_report=report,        # BacktestReport
        signal_profile=profile,        # SignalProfile
        signals=bot.signal_history,    # list[Signal]
    )
    report.generate("output/sma_crossover_profile.pdf")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from vnstock_forecast.builtin.signal_based.profile import SignalProfile
    from vnstock_forecast.builtin.signal_based.signal import Signal
    from vnstock_forecast.engine.backtest.report import BacktestReport

logger = logging.getLogger(__name__)

# ======================================================================
#  Constants
# ======================================================================

_PAGE_W, _PAGE_H = 11.69, 8.27  # A4 landscape (inches)
_MARGIN = 0.06  # margin ratio
_TITLE_FONTSIZE = 18
_SUBTITLE_FONTSIZE = 13
_BODY_FONTSIZE = 9
_TABLE_FONTSIZE = 8
_HEADER_COLOR = "#1a237e"
_ACCENT_COLOR = "#0d47a1"
_WIN_COLOR = "#2e7d32"
_LOSS_COLOR = "#c62828"
_TABLE_HEADER_BG = "#1a237e"
_TABLE_HEADER_FG = "white"
_TABLE_EVEN_ROW = "#e8eaf6"
_TABLE_ODD_ROW = "white"

# Maximum rows per table page
_MAX_TABLE_ROWS = 25


# ======================================================================
#  Helpers
# ======================================================================


def _new_page(figsize: tuple[float, float] = (_PAGE_W, _PAGE_H)) -> tuple:
    """Create a new blank page figure."""
    fig = plt.figure(figsize=figsize)
    return fig


def _add_page_header(
    fig: plt.Figure,
    title: str,
    subtitle: str = "",
    y: float = 0.95,
) -> float:
    """Add header text to page, return y position after header."""
    fig.text(
        0.05,
        y,
        title,
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        color=_HEADER_COLOR,
        va="top",
    )
    if subtitle:
        y -= 0.04
        fig.text(
            0.05,
            y,
            subtitle,
            fontsize=_SUBTITLE_FONTSIZE,
            color="gray",
            va="top",
        )
    # Separator line
    line_y = y - 0.025
    fig.add_artist(
        plt.Line2D(
            [0.05, 0.95],
            [line_y, line_y],
            transform=fig.transFigure,
            color=_ACCENT_COLOR,
            linewidth=1.5,
        )
    )
    return line_y - 0.02


def _draw_table(
    fig: plt.Figure,
    data: list[list[str]],
    col_labels: list[str],
    bbox: list[float],
    col_widths: Optional[list[float]] = None,
    title: str = "",
    title_y: Optional[float] = None,
) -> None:
    """Draw a styled table on figure."""
    ax = fig.add_axes(bbox)
    ax.axis("off")

    if not data:
        ax.text(
            0.5,
            0.5,
            "(Không có dữ liệu)",
            ha="center",
            va="center",
            fontsize=_BODY_FONTSIZE,
            color="gray",
        )
        return

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(_TABLE_FONTSIZE)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(_TABLE_HEADER_BG)
        cell.set_text_props(color=_TABLE_HEADER_FG, fontweight="bold")

    # Style data rows
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(_TABLE_EVEN_ROW if i % 2 == 0 else _TABLE_ODD_ROW)

    if title and title_y is not None:
        fig.text(
            bbox[0],
            title_y,
            title,
            fontsize=_SUBTITLE_FONTSIZE,
            fontweight="bold",
            color=_HEADER_COLOR,
            va="bottom",
        )


def _draw_kv_table(
    fig: plt.Figure,
    items: list[tuple[str, str]],
    bbox: list[float],
    title: str = "",
    title_y: Optional[float] = None,
) -> None:
    """Draw a key-value table (2 columns)."""
    data = [[k, v] for k, v in items]
    col_labels = ["Metric", "Value"]
    _draw_table(
        fig,
        data,
        col_labels,
        bbox,
        col_widths=[0.45, 0.55],
        title=title,
        title_y=title_y,
    )


def _format_number(v: Any, decimals: int = 2) -> str:
    """Format number for display."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) >= 1_000_000:
            return f"{v:,.0f}"
        return f"{v:,.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _format_pct(v: float, decimals: int = 2) -> str:
    """Format percentage."""
    return f"{v:.{decimals}f}%"


def _trunc(s: str, max_len: int = 60) -> str:
    """Truncate string to max_len characters with ellipsis."""
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# ======================================================================
#  Main class
# ======================================================================


class PDFProfileReport:
    """
    Tạo báo cáo PDF đầy đủ cho một technique/strategy sau backtest.

    Bao gồm:
    - Trang 1: Tổng quan (overview, params, summary, equity curve)
    - Trang 2: Metrics (BUY/SELL stats, alpha/beta, ratios)
    - Trang 3+: Chi tiết backtest (trade history, event log, signal list)
    - Trang N+: Signal charts (biểu đồ entry/exit từng signal)

    Args:
        technique_name:    Tên technique / strategy.
        technique_params:  Dict tham số kỹ thuật.
        description:       Mô tả technique / bot.
        backtest_report:   ``BacktestReport`` từ engine.
        signal_profile:    ``SignalProfile`` đã tính toán.
        signals:           List ``Signal`` đã kích hoạt trong backtest.
        benchmark_data:    DataFrame OHLCV benchmark (VNINDEX) để tính alpha/beta.
                           ``None`` = bỏ qua alpha/beta.
        max_signal_charts: Số chart signal tối đa được vẽ vào PDF.
                           ``None`` = không giới hạn (có thể rất chậm).
                           Mặc định ``50``.
        max_trade_history_rows: Số dòng tối đa hiển thị trong Trade History.
                           ``None`` = hiển thị tất cả (phân trang).
        max_event_log_rows:    Số dòng tối đa hiển thị trong Event Log.
        max_signal_list_rows:  Số dòng tối đa hiển thị trong Signal List.
    """

    def __init__(
        self,
        technique_name: str,
        technique_params: dict[str, Any],
        description: str = "",
        backtest_report: Optional["BacktestReport"] = None,
        signal_profile: Optional["SignalProfile"] = None,
        signals: Optional[list["Signal"]] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        max_signal_charts: Optional[int] = 50,
        max_trade_history_rows: Optional[int] = 50,
        max_event_log_rows: Optional[int] = 50,
        max_signal_list_rows: Optional[int] = 50,
    ) -> None:
        self.technique_name = technique_name
        self.technique_params = technique_params
        self.description = description
        self.report = backtest_report
        self.profile = signal_profile
        self.signals = signals or []
        self.benchmark_data = benchmark_data
        self.max_signal_charts = max_signal_charts
        self.max_trade_history_rows = max_trade_history_rows
        self.max_event_log_rows = max_event_log_rows
        self.max_signal_list_rows = max_signal_list_rows

    # ==================================================================
    #  Public API
    # ==================================================================

    def generate(self, filepath: str | Path) -> Path:
        """
        Tạo và lưu PDF report.

        Args:
            filepath: Đường dẫn file PDF đầu ra.

        Returns:
            ``Path`` tới file PDF đã tạo.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(str(filepath)) as pdf:
            print("Bắt đầu tạo PDF report cho technique '%s'...", self.technique_name)
            # Page 1: Overview
            print("Tạo trang tổng quan (overview)...")
            self._page_overview(pdf)

            # Page 2: Metrics
            print("Tạo trang metrics...")
            self._page_metrics(pdf)

            # Page 6+: Signal charts
            print("Tạo trang biểu đồ tín hiệu (signal charts)...")
            self._page_signal_charts(pdf)

            # Page 3+: Trade history
            print("Tạo trang lịch sử giao dịch (trade history)...")
            self._page_trade_history(pdf)

            # Page 4+: Event log
            print("Tạo trang nhật ký sự kiện (event log)...")
            self._page_event_log(pdf)

            # Page 5+: Signal list
            print("Tạo trang danh sách tín hiệu (signal list)...")
            self._page_signal_list(pdf)

            # Metadata
            d = pdf.infodict()
            d["Title"] = f"Profile Report: {self.technique_name}"
            d["Author"] = "vnstock-forecast"
            d["CreationDate"] = datetime.now()

        logger.info("Đã tạo PDF report: %s", filepath)
        return filepath

    # ==================================================================
    #  Page 1: Overview
    # ==================================================================

    def _page_overview(self, pdf: PdfPages) -> None:
        """Trang tổng quan: info + summary + equity curve."""
        fig = _new_page()

        # Header
        y = _add_page_header(
            fig,
            f"SIGNAL PROFILE: {self.technique_name}",
            self.description or "Technique Profile Report",
        )

        # ---- Left column: Info + Params ----
        info_items: list[tuple[str, str]] = []

        if self.report:
            info_items.append(("Bot", self.report.bot_name))
            # Truncate long symbol lists to avoid cell overflow
            symbols_str = ", ".join(self.report.symbols)
            info_items.append(("Symbols", _trunc(symbols_str, 55)))
            info_items.append(
                ("Period", f"{self.report.start.date()} → {self.report.end.date()}")
            )
            info_items.append(
                ("Commission", _format_pct(self.report.commission_rate * 100))
            )

        if self.technique_params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.technique_params.items())
            info_items.append(("Parameters", _trunc(params_str, 55)))

        if self.profile:
            info_items.append(("Total Bars", _format_number(self.profile.total_bars)))
            info_items.append(("Created", self.profile.created_at[:19]))

        _draw_kv_table(
            fig,
            info_items,
            bbox=[0.05, y - 0.30, 0.42, 0.30],
            title="Thông tin chung",
            title_y=y,
        )

        # ---- Right column: Backtest Summary ----
        summary_items: list[tuple[str, str]] = []
        if self.report:
            s = self.report.summary()
            summary_items.extend(
                [
                    ("Initial Cash", _format_number(s.get("initial_cash", 0))),
                    ("Final Equity", _format_number(s.get("final_equity", 0))),
                    ("Total PnL", _format_number(s.get("total_pnl", 0))),
                    ("Total Return", _format_pct(s.get("total_return_pct", 0))),
                    ("Total Trades", _format_number(s.get("total_trades", 0))),
                    ("Wins / Losses", f"{s.get('wins', 0)} / {s.get('losses', 0)}"),
                    ("Win Rate", _format_pct(s.get("winrate_pct", 0), 1)),
                    ("Avg Win", _format_pct(s.get("avg_win_pct", 0))),
                    ("Avg Loss", _format_pct(s.get("avg_loss_pct", 0))),
                    ("R:R Ratio", _format_number(s.get("reward_risk_ratio", 0))),
                    ("Max Drawdown", _format_pct(s.get("max_drawdown_pct", 0))),
                ]
            )
            reasons = s.get("close_reasons", {})
            if reasons:
                reasons_str = ", ".join(f"{k}: {v}" for k, v in reasons.items())
                summary_items.append(("Close Reasons", _trunc(reasons_str, 55)))

        _draw_kv_table(
            fig,
            summary_items,
            bbox=[0.52, y - 0.30, 0.44, 0.30],
            title="Kết quả Backtest",
            title_y=y,
        )

        # ---- Bottom: Equity Curve ----
        if self.report and self.report.equity_curve:
            eq_y = y - 0.34
            fig.text(
                0.05,
                eq_y,
                "Equity Curve",
                fontsize=_SUBTITLE_FONTSIZE,
                fontweight="bold",
                color=_HEADER_COLOR,
            )

            ax = fig.add_axes([0.07, 0.06, 0.88, eq_y - 0.10])
            eq_df = self.report.equity_df()
            ax.plot(
                eq_df.index,
                eq_df["equity"],
                color=_ACCENT_COLOR,
                linewidth=1.2,
                label="Equity",
            )
            ax.fill_between(
                eq_df.index,
                eq_df["equity"].iloc[0],
                eq_df["equity"],
                alpha=0.08,
                color=_ACCENT_COLOR,
            )
            ax.axhline(
                self.report.initial_cash,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=f"Initial: {self.report.initial_cash:,.0f}",
            )
            ax.set_ylabel("Equity (VND)", fontsize=_BODY_FONTSIZE)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # Footer
        self._add_footer(fig)

        pdf.savefig(fig)
        plt.close(fig)

    # ==================================================================
    #  Page 2: Metrics
    # ==================================================================

    def _page_metrics(self, pdf: PdfPages) -> None:
        """Trang metrics: BUY/SELL stats, alpha/beta, ratios."""
        fig = _new_page()
        y = _add_page_header(fig, "SIGNAL METRICS", "Đánh giá hiệu quả tín hiệu")

        # ---- BUY Stats ----
        if self.profile:
            bs = self.profile.buy_stats
            buy_data = [
                ("Total Signals", _format_number(bs.total_signals)),
                ("Wins", _format_number(bs.win_count)),
                ("Losses", _format_number(bs.loss_count)),
                ("Win Rate", _format_pct(bs.win_rate * 100, 1)),
                ("Avg Return", _format_pct(bs.avg_return_pct, 2)),
                ("Avg Win", _format_pct(bs.avg_win_pct, 2)),
                ("Avg Loss", _format_pct(bs.avg_loss_pct, 2)),
                ("R:R Ratio", _format_number(bs.risk_reward)),
                ("Frequency", f"{bs.frequency:.4f}"),
            ]

            _draw_kv_table(
                fig,
                buy_data,
                bbox=[0.05, y - 0.32, 0.42, 0.32],
                title="BUY Signal Stats",
                title_y=y,
            )

            ss = self.profile.sell_stats
            sell_data = [
                ("Total Signals", _format_number(ss.total_signals)),
                ("Wins", _format_number(ss.win_count)),
                ("Losses", _format_number(ss.loss_count)),
                ("Win Rate", _format_pct(ss.win_rate * 100, 1)),
                ("Avg Return", _format_pct(ss.avg_return_pct, 2)),
                ("Avg Win", _format_pct(ss.avg_win_pct, 2)),
                ("Avg Loss", _format_pct(ss.avg_loss_pct, 2)),
                ("R:R Ratio", _format_number(ss.risk_reward)),
                ("Frequency", f"{ss.frequency:.4f}"),
            ]

            _draw_kv_table(
                fig,
                sell_data,
                bbox=[0.52, y - 0.32, 0.44, 0.32],
                title="SELL Signal Stats",
                title_y=y,
            )

            # Overall
            overall_y = y - 0.35
            fig.text(
                0.05,
                overall_y,
                f"Overall Win Rate: {self.profile.overall_win_rate:.1%}    |    "
                f"Total Signals: {self.profile.total_signals}",
                fontsize=_SUBTITLE_FONTSIZE,
                fontweight="bold",
                color=_ACCENT_COLOR,
            )
        else:
            fig.text(
                0.5,
                y - 0.15,
                "(Không có Signal Profile data)",
                ha="center",
                fontsize=_BODY_FONTSIZE,
                color="gray",
            )

        # ---- Alpha / Beta / Ratios ----
        ratios_y = y - 0.40
        fig.text(
            0.05,
            ratios_y,
            "Risk-Adjusted Metrics",
            fontsize=_SUBTITLE_FONTSIZE,
            fontweight="bold",
            color=_HEADER_COLOR,
        )

        metrics = self._compute_risk_metrics()
        metrics_items = [
            ("Sharpe Ratio", _format_number(metrics.get("sharpe_ratio", 0), 3)),
            ("Sortino Ratio", _format_number(metrics.get("sortino_ratio", 0), 3)),
            ("Calmar Ratio", _format_number(metrics.get("calmar_ratio", 0), 3)),
            ("Profit Factor", _format_number(metrics.get("profit_factor", 0), 3)),
            ("Expectancy", _format_number(metrics.get("expectancy", 0), 4)),
        ]

        if "alpha" in metrics:
            metrics_items.append(
                ("Alpha (vs VNINDEX)", _format_number(metrics["alpha"], 4))
            )
        if "beta" in metrics:
            metrics_items.append(
                ("Beta (vs VNINDEX)", _format_number(metrics["beta"], 4))
            )

        _draw_kv_table(
            fig,
            metrics_items,
            bbox=[0.05, ratios_y - 0.04 - 0.26, 0.42, 0.26],
        )

        # ---- Monthly Returns Heatmap (if enough data) ----
        if (
            self.report
            and self.report.equity_curve
            and len(self.report.equity_curve) > 30
        ):
            ax = fig.add_axes([0.52, 0.06, 0.44, ratios_y - 0.12])
            self._draw_drawdown_chart(ax)

        self._add_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)

    # ==================================================================
    #  Page 3+: Trade History
    # ==================================================================

    def _page_trade_history(self, pdf: PdfPages) -> None:
        """Trang trade history (phân trang nếu nhiều)."""
        if not self.report:
            return

        df = self.report.trade_history()
        if df.empty:
            return

        col_labels = [
            "ID",
            "Symbol",
            "Entry Time",
            "Exit Time",
            "Entry Price",
            "Exit Price",
            "Qty",
            "SL",
            "TP",
            "PnL",
            "PnL%",
            "Reason",
        ]

        rows = []
        for _, r in df.iterrows():
            rows.append(
                [
                    str(r.get("id", "")),
                    str(r.get("symbol", "")),
                    str(r.get("entry_time", ""))[:16],
                    str(r.get("exit_time", ""))[:16],
                    _format_number(r.get("entry_price", 0), 0),
                    _format_number(r.get("exit_price", 0), 0),
                    _format_number(r.get("quantity", 0), 0),
                    _format_number(r.get("stop_loss", 0), 0),
                    _format_number(r.get("take_profit", 0), 0),
                    _format_number(r.get("pnl", 0), 0),
                    _format_pct(r.get("pnl_pct", 0)),
                    str(r.get("close_reason", "")),
                ]
            )

        # Áp dụng giới hạn tổng số dòng
        total_rows = len(rows)
        cap = self.max_trade_history_rows
        if cap is not None and total_rows > cap:
            rows = rows[:cap]

        # Paginate
        for page_idx in range(0, len(rows), _MAX_TABLE_ROWS):
            chunk = rows[page_idx : page_idx + _MAX_TABLE_ROWS]
            page_num = page_idx // _MAX_TABLE_ROWS + 1
            total_pages = (len(rows) + _MAX_TABLE_ROWS - 1) // _MAX_TABLE_ROWS

            cap_note = (
                f" (giới hạn {cap}/{total_rows})"
                if cap is not None and total_rows > cap
                else ""
            )
            fig = _new_page()
            y = _add_page_header(
                fig,
                "TRADE HISTORY",
                f"Trang {page_num}/{total_pages} – "
                f"Hiển thị {len(rows)}/{total_rows} giao dịch{cap_note}",
            )

            _draw_table(
                fig,
                chunk,
                col_labels,
                bbox=[0.02, 0.04, 0.96, y - 0.06],
                col_widths=[
                    0.06,
                    0.06,
                    0.11,
                    0.11,
                    0.08,
                    0.08,
                    0.06,
                    0.07,
                    0.07,
                    0.09,
                    0.07,
                    0.10,
                ],
            )

            self._add_footer(fig)
            pdf.savefig(fig)
            plt.close(fig)

    # ==================================================================
    #  Page 4+: Event Log
    # ==================================================================

    def _page_event_log(self, pdf: PdfPages) -> None:
        """Trang event log (phân trang nếu nhiều)."""
        if not self.report:
            return

        df = self.report.event_log()
        if df.empty:
            return

        col_labels = [
            "Timestamp",
            "Action",
            "Symbol",
            "Price",
            "Qty",
            "Position ID",
            "Equity",
            "Reason",
        ]

        rows = []
        for _, r in df.iterrows():
            rows.append(
                [
                    str(r.get("timestamp", ""))[:16],
                    str(r.get("action", "")),
                    str(r.get("symbol", "")),
                    _format_number(r.get("price", 0), 0),
                    _format_number(r.get("quantity", 0), 0),
                    str(r.get("position_id", ""))[:8],
                    _format_number(r.get("equity", 0), 0),
                    str(r.get("reason", ""))[:40],
                ]
            )

        # Áp dụng giới hạn tổng số dòng
        total_rows = len(rows)
        cap = self.max_event_log_rows
        if cap is not None and total_rows > cap:
            rows = rows[:cap]

        for page_idx in range(0, len(rows), _MAX_TABLE_ROWS):
            chunk = rows[page_idx : page_idx + _MAX_TABLE_ROWS]
            page_num = page_idx // _MAX_TABLE_ROWS + 1
            total_pages = (len(rows) + _MAX_TABLE_ROWS - 1) // _MAX_TABLE_ROWS

            cap_note = (
                f" (giới hạn {cap}/{total_rows})"
                if cap is not None and total_rows > cap
                else ""
            )
            fig = _new_page()
            y = _add_page_header(
                fig,
                "EVENT LOG",
                f"Trang {page_num}/{total_pages} – "
                f"Hiển thị {len(rows)}/{total_rows} sự kiện{cap_note}",
            )

            _draw_table(
                fig,
                chunk,
                col_labels,
                bbox=[0.02, 0.04, 0.96, y - 0.06],
                col_widths=[0.13, 0.08, 0.07, 0.10, 0.08, 0.10, 0.13, 0.27],
            )

            self._add_footer(fig)
            pdf.savefig(fig)
            plt.close(fig)

    # ==================================================================
    #  Page 5+: Signal List
    # ==================================================================

    def _page_signal_list(self, pdf: PdfPages) -> None:
        """Trang danh sách signal đã kích hoạt."""
        if not self.signals:
            return

        col_labels = [
            "Timestamp",
            "Symbol",
            "Direction",
            "Technique",
            "Confidence",
            "Entry",
            "SL",
            "TP",
            "Reason",
        ]

        rows = []
        for sig in self.signals:
            tp = sig.trade_plan
            rows.append(
                [
                    str(sig.timestamp)[:16],
                    sig.symbol,
                    sig.direction.value.upper(),
                    sig.technique,
                    f"{sig.confidence:.2f}",
                    _format_number(tp.entry, 0) if tp else "–",
                    _format_number(tp.stop_loss, 0) if tp else "–",
                    _format_number(tp.take_profit, 0) if tp else "–",
                    (sig.reason[:35] + "…") if len(sig.reason) > 35 else sig.reason,
                ]
            )

        # Áp dụng giới hạn tổng số dòng
        total_rows = len(rows)
        cap = self.max_signal_list_rows
        if cap is not None and total_rows > cap:
            rows = rows[:cap]

        for page_idx in range(0, len(rows), _MAX_TABLE_ROWS):
            chunk = rows[page_idx : page_idx + _MAX_TABLE_ROWS]
            page_num = page_idx // _MAX_TABLE_ROWS + 1
            total_pages = (len(rows) + _MAX_TABLE_ROWS - 1) // _MAX_TABLE_ROWS

            cap_note = (
                f" (giới hạn {cap}/{total_rows})"
                if cap is not None and total_rows > cap
                else ""
            )
            fig = _new_page()
            y = _add_page_header(
                fig,
                "SIGNAL LIST",
                f"Trang {page_num}/{total_pages} – "
                f"Hiển thị {len(rows)}/{total_rows} tín hiệu đã kích hoạt{cap_note}",
            )

            _draw_table(
                fig,
                chunk,
                col_labels,
                bbox=[0.02, 0.04, 0.96, y - 0.06],
                col_widths=[0.12, 0.07, 0.07, 0.11, 0.08, 0.09, 0.09, 0.09, 0.24],
            )

            self._add_footer(fig)
            pdf.savefig(fig)
            plt.close(fig)

    # ==================================================================
    #  Page 6+: Signal Charts
    # ==================================================================

    def _page_signal_charts(self, pdf: PdfPages) -> None:
        """Trang biểu đồ entry/exit cho MỌI signal (plot_signal).

        Mỗi signal có ``snapshot`` sẽ được vẽ bằng ``plot_signal()``.
        Signal không có snapshot sẽ được bỏ qua kèm warning.

        Số chart bị giới hạn bởi ``self.max_signal_charts`` để tránh PDF
        quá lớn và thời gian render quá lâu.
        """
        if not self.signals:
            return

        from vnstock_forecast.builtin.signal_based.visualization.plotter import (
            plot_signal,
        )

        # Chỉ lấy signals có snapshot
        signals_with_snap = [s for s in self.signals if s.snapshot is not None]
        n_without = len(self.signals) - len(signals_with_snap)

        # Áp dụng giới hạn (lấy ngẫu nhiên thay vì lấy đầu)
        import random as _random

        cap = self.max_signal_charts
        capped = cap is not None and len(signals_with_snap) > cap
        signals_to_plot = (
            _random.sample(signals_with_snap, cap) if capped else signals_with_snap
        )
        n_with = len(signals_with_snap)

        # Thêm trang tiêu đề section
        fig = _new_page()
        _add_page_header(
            fig,
            "SIGNAL CHARTS",
            f"Biểu đồ entry/exit – hiển thị {len(signals_to_plot)}/{len(self.signals)} tín hiệu",  # noqa: E501
        )

        # Thống kê nhanh
        info_y = 0.75
        fig.text(
            0.1,
            info_y,
            f"Tổng signals: {len(self.signals)}    |    "
            f"Có snapshot: {n_with}    |    "
            f"Không có snapshot: {n_without}    |    "
            f"Được vẽ: {len(signals_to_plot)}",
            fontsize=_BODY_FONTSIZE + 1,
            color="gray",
        )
        if n_without > 0:
            fig.text(
                0.1,
                info_y - 0.05,
                "⚠ Signals không có snapshot sẽ không được vẽ. "
                "Hãy bật attach_snapshot=True trên technique trước khi chạy.",
                fontsize=_BODY_FONTSIZE,
                color=_LOSS_COLOR,
            )
        if capped:
            fig.text(
                0.1,
                info_y - 0.10,
                f"⚠ Lấy ngẫu nhiên {cap} chart (max_signal_charts={cap}). "
                f"Truyền max_signal_charts=None để bỏ giới hạn.",
                fontsize=_BODY_FONTSIZE,
                color="darkorange",
            )

        self._add_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)

        # Vẽ từng signal (không extend_bars để tránh query DB cho từng signal)
        plotted = 0
        for i, signal in enumerate(signals_to_plot):
            try:
                chart_fig = plot_signal(
                    signal,
                    figsize=(_PAGE_W, _PAGE_H - 1),
                    title=(
                        f"[{plotted + 1}/{len(signals_to_plot)}] "
                        f"{signal.direction.value.upper()} {signal.symbol} "
                        f"@ {str(signal.timestamp)[:16]} "
                        f"({signal.technique}, conf={signal.confidence:.2f})"
                    ),
                    show=False,
                )
                pdf.savefig(chart_fig)
                plt.close(chart_fig)
                plotted += 1
            except Exception as exc:
                logger.warning(
                    "Không thể vẽ signal chart #%d (%s %s): %s",
                    i + 1,
                    signal.symbol,
                    signal.timestamp,
                    exc,
                )

        logger.info("Đã vẽ %d/%d signal charts vào PDF.", plotted, len(self.signals))

    # ==================================================================
    #  Risk-adjusted metrics
    # ==================================================================

    def _compute_risk_metrics(self) -> dict[str, float]:
        """Tính Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Alpha, Beta."""
        metrics: dict[str, float] = {}

        if not self.report or not self.report.equity_curve:
            return metrics

        # Daily returns from equity curve
        eq_df = self.report.equity_df()
        if len(eq_df) < 2:
            return metrics

        returns = eq_df["equity"].pct_change().dropna()

        if returns.empty:
            return metrics

        # Annualization factor (rough estimate from data frequency)
        n_periods = len(returns)
        duration_days = (eq_df.index[-1] - eq_df.index[0]).days
        if duration_days > 0:
            periods_per_year = n_periods / duration_days * 252
        else:
            periods_per_year = 252

        mean_ret = returns.mean()
        std_ret = returns.std()

        # Sharpe Ratio (risk-free rate ~ 0 for simplicity)
        if std_ret > 0:
            metrics["sharpe_ratio"] = (mean_ret / std_ret) * np.sqrt(periods_per_year)
        else:
            metrics["sharpe_ratio"] = 0.0

        # Sortino Ratio
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        if downside_std > 0:
            metrics["sortino_ratio"] = (mean_ret / downside_std) * np.sqrt(
                periods_per_year
            )
        else:
            metrics["sortino_ratio"] = 0.0

        # Calmar Ratio
        total_return = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) - 1
        max_dd = self._compute_max_drawdown_from_equity(eq_df)
        if max_dd > 0:
            years = max(duration_days / 365.25, 1 / 365.25)
            annual_return = (1 + total_return) ** (1 / years) - 1
            metrics["calmar_ratio"] = annual_return / max_dd
        else:
            metrics["calmar_ratio"] = 0.0

        # Profit Factor & Expectancy from trades
        if self.report:
            from vnstock_forecast.engine.backtest.portfolio import CloseReason

            closed = self.report.portfolio.closed_positions
            real = [p for p in closed if p.close_reason != CloseReason.END_OF_DATA]
            wins_pnl = sum(p.pnl for p in real if (p.pnl or 0) > 0)
            losses_pnl = abs(sum(p.pnl for p in real if (p.pnl or 0) <= 0))

            if losses_pnl > 0:
                metrics["profit_factor"] = wins_pnl / losses_pnl
            else:
                metrics["profit_factor"] = float("inf") if wins_pnl > 0 else 0.0

            # Expectancy = avg_win * win_rate - avg_loss * loss_rate
            n_trades = len(real)
            if n_trades > 0:
                n_wins = sum(1 for p in real if (p.pnl or 0) > 0)
                n_losses = n_trades - n_wins
                wr = n_wins / n_trades
                lr = n_losses / n_trades
                avg_w = (wins_pnl / n_wins) if n_wins > 0 else 0
                avg_l = (losses_pnl / n_losses) if n_losses > 0 else 0
                metrics["expectancy"] = wr * avg_w - lr * avg_l
            else:
                metrics["expectancy"] = 0.0

        # Alpha & Beta vs benchmark
        if self.benchmark_data is not None and not self.benchmark_data.empty:
            alpha, beta = self._compute_alpha_beta(returns)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

        return metrics

    def _compute_alpha_beta(self, portfolio_returns: pd.Series) -> tuple[float, float]:
        """Tính Alpha và Beta vs benchmark (VNINDEX).

        Alpha = portfolio_return - beta * benchmark_return
        Beta  = cov(Rp, Rb) / var(Rb)
        """
        try:
            bm = self.benchmark_data.copy()
            if "Close" not in bm.columns:
                return 0.0, 0.0

            # Normalise benchmark index → DatetimeIndex (date level)
            if not isinstance(bm.index, pd.DatetimeIndex):
                if "Timestamp" in bm.columns:
                    # Timestamp column still present
                    bm.index = pd.to_datetime(bm["Timestamp"], unit="s")
                else:
                    # Index was already set from Timestamp integers (unit=seconds)
                    bm.index = pd.to_datetime(bm.index, unit="s")
            bm.index = bm.index.normalize()  # strip time component → date-level

            bm_returns = bm["Close"].pct_change().dropna()

            # Normalise portfolio index → DatetimeIndex (date level)
            port_idx = portfolio_returns.index
            if not isinstance(port_idx, pd.DatetimeIndex):
                # Integers are Unix timestamps in seconds
                port_idx = pd.to_datetime(port_idx, unit="s")
            port_idx = port_idx.normalize()
            portfolio_returns = portfolio_returns.copy()
            portfolio_returns.index = port_idx

            # Align dates
            combined = pd.DataFrame(
                {
                    "portfolio": portfolio_returns,
                    "benchmark": bm_returns,
                }
            ).dropna()

            if len(combined) < 5:
                return 0.0, 0.0

            cov_matrix = combined.cov()
            var_bm = combined["benchmark"].var()

            if var_bm > 0:
                beta = cov_matrix.loc["portfolio", "benchmark"] / var_bm
            else:
                beta = 0.0

            # Annualized
            mean_p = combined["portfolio"].mean() * 252
            mean_b = combined["benchmark"].mean() * 252
            alpha = mean_p - beta * mean_b

            return alpha, beta

        except Exception as exc:
            logger.warning("Không thể tính alpha/beta: %s", exc)
            return 0.0, 0.0

    @staticmethod
    def _compute_max_drawdown_from_equity(eq_df: pd.DataFrame) -> float:
        """Max drawdown (fraction, not pct) from equity DataFrame."""
        equity = eq_df["equity"]
        peak = equity.cummax()
        dd = (peak - equity) / peak
        return dd.max() if len(dd) > 0 else 0.0

    # ==================================================================
    #  Drawdown chart
    # ==================================================================

    def _draw_drawdown_chart(self, ax: plt.Axes) -> None:
        """Vẽ biểu đồ drawdown trên axes."""
        eq_df = self.report.equity_df()
        equity = eq_df["equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100  # percentage

        ax.fill_between(
            drawdown.index,
            drawdown,
            0,
            alpha=0.3,
            color=_LOSS_COLOR,
            label="Drawdown",
        )
        ax.plot(
            drawdown.index,
            drawdown,
            color=_LOSS_COLOR,
            linewidth=0.8,
        )
        ax.set_ylabel("Drawdown (%)", fontsize=_BODY_FONTSIZE - 1)
        ax.set_title("Underwater Equity (Drawdown)", fontsize=_BODY_FONTSIZE, pad=5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    # ==================================================================
    #  Footer
    # ==================================================================

    @staticmethod
    def _add_footer(fig: plt.Figure) -> None:
        """Add footer with generation time."""
        fig.text(
            0.5,
            0.01,
            f"Generated by vnstock-forecast  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",  # noqa: E501
            ha="center",
            fontsize=7,
            color="gray",
        )
