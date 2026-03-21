"""Signal plotter – vẽ biểu đồ nến từ ``SignalSnapshot`` bằng mplfinance."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from .snapshot import SignalSnapshot

if TYPE_CHECKING:
    from vnstock_forecast.builtin.signal_based.signal import Signal

logger = logging.getLogger(__name__)


# ======================================================================
#  Helpers
# ======================================================================


def _is_non_interactive_backend() -> bool:
    """Kiểm tra backend matplotlib hiện tại có non-interactive không."""
    backend = (plt.get_backend() or "").lower()
    non_interactive_tokens = ("agg", "pdf", "ps", "svg", "cairo", "template")
    return any(token in backend for token in non_interactive_tokens)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo DataFrame có ``DatetimeIndex`` hợp lệ cho mplfinance."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, unit="s")
        except Exception:
            df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _ensure_datetime_series_index(series: pd.Series) -> pd.Series:
    """Đảm bảo Series có ``DatetimeIndex`` để align với OHLCV khi plot."""
    out = series.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, unit="s")
        except Exception:
            out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out


def _ts_to_bar_idx(ohlcv: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Chuyển timestamp thành chỉ số bar gần nhất trong *ohlcv*."""
    diffs = abs(ohlcv.index - ts)
    return int(diffs.argmin())


def _infer_bar_seconds(ohlcv: pd.DataFrame, resolution: str) -> int:
    """Ước lượng độ dài 1 bar (giây)."""
    if len(ohlcv.index) >= 2:
        diffs = pd.Series(ohlcv.index).diff().dropna().dt.total_seconds()
        if not diffs.empty:
            return max(int(diffs.median()), 1)

    res = (resolution or "").upper()
    if res == "D":
        return 24 * 60 * 60
    if res == "W":
        return 7 * 24 * 60 * 60
    if res == "M":
        return 30 * 24 * 60 * 60
    try:
        return max(int(res) * 60, 1)
    except Exception:
        return 24 * 60 * 60


def _extend_ohlcv(
    snapshot: SignalSnapshot,
    *,
    until_ts: Optional[pd.Timestamp] = None,
    extend_bars: int = 0,
) -> pd.DataFrame:
    """Query thêm OHLCV kể từ bar cuối cùng, có thể ép mở rộng tới mốc thời gian.

    Args:
        snapshot:     ``SignalSnapshot`` nguồn.
        until_ts:     Mốc cần phủ tới (ví dụ thời điểm thoát lệnh).
        extend_bars:  Số bars lấy thêm sau khi đã phủ ``until_ts``.

    Returns:
        DataFrame OHLCV đã nối thêm dữ liệu (hoặc nguyên bản nếu thất bại).
    """
    from vnstock_forecast.engine.data.query import query_ohlcv

    ohlcv = _ensure_datetime_index(snapshot.ohlcv)

    if ohlcv.empty:
        return ohlcv

    bar_seconds = _infer_bar_seconds(ohlcv, snapshot.resolution)
    last_ts = int(ohlcv.index[-1].timestamp())

    target_ts = last_ts
    if until_ts is not None:
        target_ts = max(target_ts, int(until_ts.timestamp()))
    if extend_bars > 0:
        target_ts += bar_seconds * extend_bars

    if target_ts <= last_ts:
        return ohlcv

    try:
        extra = query_ohlcv(
            symbols=snapshot.symbol,
            resolutions=snapshot.resolution,
            from_ts=last_ts,
            to_ts=target_ts,
        )
    except Exception as exc:
        logger.warning("Không thể query thêm dữ liệu: %s", exc)
        return ohlcv
    if extra.empty:
        return ohlcv

    ohlcv_cols = [
        c for c in ("Open", "High", "Low", "Close", "Volume") if c in extra.columns
    ]
    extra_df = extra.set_index("Timestamp")[ohlcv_cols].sort_index()
    extra_df = _ensure_datetime_index(extra_df)

    combined = pd.concat([ohlcv[ohlcv_cols], extra_df])
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


# ======================================================================
#  Public API
# ======================================================================


def plot_signal(
    signal_or_snapshot: "Signal | SignalSnapshot",
    *,
    extend_bars: Optional[int] = 15,
    figsize: tuple[int, int] = (16, 10),
    style: str = "charles",
    title: str | None = None,
    savefig: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Vẽ biểu đồ nến với mọi overlay từ ``SignalSnapshot``.

    Hàm này đọc snapshot đính kèm signal (hoặc nhận trực tiếp một
    ``SignalSnapshot``) rồi dùng **mplfinance** render biểu đồ nến kèm:

    * Đường Entry / SL / TP tại vị trí chính xác.
    * Vùng tô risk (đỏ nhạt) & reward (xanh nhạt).
    * Indicator lines trên main chart hoặc subplots riêng.
    * HLine / VLine / Rectangle / TrendLine tuỳ ý.
    * Mũi tên đánh dấu thời điểm phát signal.
    * Đường time-limit nếu có.

    Nếu có dữ liệu thoát lệnh (``exit_time``), hàm tự động mở rộng dữ liệu
    đến bar thoát lệnh trước; sau đó nếu ``extend_bars`` > 0 sẽ query thêm
    tối đa *N* bars tiếp theo qua ``engine.data.query``.

    Args:
        signal_or_snapshot: Đối tượng ``Signal`` (cần có ``snapshot``) hoặc
                            ``SignalSnapshot`` trực tiếp.
        extend_bars:        Số bars muốn mở rộng thêm sau khi đã phủ
                    tới điểm thoát (nếu có). ``None`` → không thêm.
        figsize:            Kích thước figure ``(width, height)``.
        style:              mplfinance style (charles, yahoo, nightclouds…).
        title:              Tiêu đề. ``None`` → tự sinh từ symbol.
        savefig:            Đường dẫn lưu ảnh. ``None`` → không lưu.
        show:               Gọi ``plt.show()``.

    Returns:
        ``matplotlib.figure.Figure``

    Raises:
        ValueError: Nếu *signal_or_snapshot* là ``Signal`` mà không có snapshot.
    """
    # --- Resolve snapshot ---
    if isinstance(signal_or_snapshot, SignalSnapshot):
        snapshot = signal_or_snapshot
    else:
        snapshot = getattr(signal_or_snapshot, "snapshot", None)
        if snapshot is None:
            raise ValueError(
                "Signal không có snapshot. Hãy đảm bảo attach_snapshot=True "
                "trên technique trước khi chạy."
            )

    entry_time = snapshot.entry_time or snapshot.signal_time
    entry_price = snapshot.entry
    exit_time = snapshot.exit_time
    exit_price = snapshot.exit_price

    if not isinstance(signal_or_snapshot, SignalSnapshot):
        metadata = getattr(signal_or_snapshot, "metadata", {})
        if isinstance(metadata, dict):
            if entry_time is None and metadata.get("entry_time") is not None:
                entry_time = pd.Timestamp(metadata["entry_time"]).to_pydatetime()
            if entry_price is None and metadata.get("entry_price") is not None:
                entry_price = float(metadata["entry_price"])
            if exit_time is None and metadata.get("exit_time") is not None:
                exit_time = pd.Timestamp(metadata["exit_time"]).to_pydatetime()
            if exit_price is None and metadata.get("exit_price") is not None:
                exit_price = float(metadata["exit_price"])

    # --- 1) OHLCV ---
    ohlcv = _ensure_datetime_index(snapshot.ohlcv)

    if exit_time is not None or (extend_bars is not None and extend_bars > 0):
        ohlcv = _extend_ohlcv(
            snapshot,
            until_ts=(pd.Timestamp(exit_time) if exit_time is not None else None),
            extend_bars=(extend_bars or 0),
        )

    ohlcv_cols = [
        c for c in ("Open", "High", "Low", "Close", "Volume") if c in ohlcv.columns
    ]
    ohlcv = ohlcv[ohlcv_cols]

    # --- 2) Build addplots cho indicators ---
    addplots: list[dict] = []
    for ind in snapshot.indicators:
        ind_data = _ensure_datetime_series_index(ind.data)
        aligned = ind_data.reindex(ohlcv.index)
        kwargs: dict = dict(
            panel=ind.panel,
            color=ind.color,
            secondary_y=ind.secondary_y,
            ylabel=ind.ylabel or ind.name,
        )
        if ind.type == "bar":
            kwargs.update(type="bar", width=0.7, alpha=ind.alpha)
        else:
            kwargs.update(linestyle=ind.linestyle, width=ind.linewidth)
        addplots.append(mpf.make_addplot(aligned, **kwargs))

    # HLines trên indicator panels → constant-series addplots
    for hline in snapshot.hlines:
        if hline.panel > 0:
            const = pd.Series(hline.value, index=ohlcv.index, dtype=float)
            addplots.append(
                mpf.make_addplot(
                    const,
                    panel=hline.panel,
                    color=hline.color,
                    linestyle=hline.linestyle,
                    width=hline.linewidth,
                    secondary_y=False,
                )
            )

    # --- 3) mplfinance plot ---
    if title is None:
        title = snapshot.symbol or "Signal Chart"

    plot_kwargs: dict = dict(
        type="candle",
        style=style,
        volume="Volume" in ohlcv.columns,
        figsize=figsize,
        title=title,
        returnfig=True,
        warn_too_much_data=10_000,
    )
    if addplots:
        plot_kwargs["addplot"] = addplots

    fig, axes = mpf.plot(ohlcv, **plot_kwargs)
    ax_main = axes[0]

    # --- Compute signal bar index (used in sections 4 & 5) ---
    signal_idx: int | None = None
    if snapshot.signal_time is not None:
        signal_idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(snapshot.signal_time))

    entry_idx: int | None = None
    if entry_time is not None:
        entry_idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(entry_time))

    exit_idx: int | None = None
    if exit_time is not None:
        exit_idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(exit_time))

    # --- 4) Entry / SL / TP (start from entry/signal bar) ---
    sig_x: int = entry_idx if entry_idx is not None else (signal_idx or 0)
    n_bar: int = len(ohlcv) - 1
    line_end: int = exit_idx if exit_idx is not None else n_bar
    line_end = max(sig_x, min(line_end, n_bar))

    if entry_price is not None:
        ax_main.hlines(
            entry_price,
            sig_x,
            line_end,
            colors="#2196F3",
            linestyles="-",
            linewidth=1.6,
            label=f"Entry {entry_price:,.0f}",
            alpha=0.9,
        )
    if snapshot.stop_loss is not None:
        ax_main.hlines(
            snapshot.stop_loss,
            sig_x,
            line_end,
            colors="#F44336",
            linestyles="--",
            linewidth=1.4,
            label=f"SL {snapshot.stop_loss:,.0f}",
            alpha=0.9,
        )
    if snapshot.take_profit is not None:
        ax_main.hlines(
            snapshot.take_profit,
            sig_x,
            line_end,
            colors="#4CAF50",
            linestyles="--",
            linewidth=1.4,
            label=f"TP {snapshot.take_profit:,.0f}",
            alpha=0.9,
        )

    # Tô vùng risk / reward (chỉ từ thời điểm signal)
    x_fill = list(range(sig_x, line_end + 1))
    if entry_price is not None and snapshot.stop_loss is not None:
        sl_lo = min(entry_price, snapshot.stop_loss)
        sl_hi = max(entry_price, snapshot.stop_loss)
        ax_main.fill_between(x_fill, sl_lo, sl_hi, alpha=0.06, color="red")
    if entry_price is not None and snapshot.take_profit is not None:
        tp_lo = min(entry_price, snapshot.take_profit)
        tp_hi = max(entry_price, snapshot.take_profit)
        ax_main.fill_between(x_fill, tp_lo, tp_hi, alpha=0.06, color="green")

    # --- 5) BUY/SELL markers ---
    buy_points: list[tuple[datetime, float]] = list(snapshot.buy_points)
    if not buy_points and entry_time is not None and entry_price is not None:
        buy_points = [(entry_time, float(entry_price))]

    sell_points: list[tuple[datetime, float]] = list(snapshot.sell_points)
    if not sell_points and exit_time is not None and exit_price is not None:
        sell_points = [(exit_time, float(exit_price))]

    buy_plot_points: list[tuple[int, float]] = []
    for ts, price in buy_points:
        idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(ts))
        buy_plot_points.append((idx, float(price)))

    sell_plot_points: list[tuple[int, float]] = []
    for ts, price in sell_points:
        idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(ts))
        sell_plot_points.append((idx, float(price)))

    if buy_plot_points:
        buy_x = [point[0] for point in buy_plot_points]
        buy_y = [point[1] for point in buy_plot_points]
        ax_main.scatter(
            buy_x,
            buy_y,
            marker="^",
            s=85,
            color="#1E88E5",
            edgecolors="white",
            linewidths=0.7,
            zorder=6,
            label="Buy",
        )
        first_buy_idx, first_buy_price = buy_plot_points[0]
        ax_main.annotate(
            " BUY",
            xy=(first_buy_idx, first_buy_price),
            xytext=(max(0, first_buy_idx - 2), first_buy_price * 1.012),
            fontsize=8,
            fontweight="bold",
            color="#1E88E5",
            arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=1.2),
        )

    if sell_plot_points:
        sell_x = [point[0] for point in sell_plot_points]
        sell_y = [point[1] for point in sell_plot_points]
        ax_main.scatter(
            sell_x,
            sell_y,
            marker="v",
            s=85,
            color="#E53935",
            edgecolors="white",
            linewidths=0.7,
            zorder=6,
            label="Sell",
        )
        last_sell_idx, last_sell_price = sell_plot_points[-1]
        ax_main.annotate(
            " SELL",
            xy=(last_sell_idx, last_sell_price),
            xytext=(max(0, last_sell_idx - 2), last_sell_price * 0.988),
            fontsize=8,
            fontweight="bold",
            color="#E53935",
            arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.2),
        )
    elif (
        snapshot.signal_time is not None
        and signal_idx is not None
        and entry_price is not None
    ):
        ax_main.annotate(
            " SIGNAL",
            xy=(signal_idx, entry_price),
            xytext=(max(0, signal_idx - 3), entry_price * 1.015),
            fontsize=9,
            fontweight="bold",
            color="#2196F3",
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5),
        )

    # --- 6) Time-limit vertical ---
    if snapshot.time_limit is not None:
        limit_ts = pd.Timestamp(snapshot.time_limit)
        if limit_ts in ohlcv.index or (
            ohlcv.index.min() <= limit_ts <= ohlcv.index.max()
        ):
            limit_idx = _ts_to_bar_idx(ohlcv, limit_ts)
            ax_main.axvline(
                limit_idx,
                color="orange",
                linestyle=":",
                linewidth=1.3,
                label="Time Limit",
                alpha=0.7,
            )

    # --- 7) Custom HLines (panel 0 only – panel>0 đã xử lý ở bước 2) ---
    for hline in snapshot.hlines:
        if hline.panel == 0:
            ax_main.axhline(
                hline.value,
                color=hline.color,
                linestyle=hline.linestyle,
                linewidth=hline.linewidth,
                label=hline.label,
                alpha=0.7,
            )

    # --- 8) Custom VLines ---
    for vline in snapshot.vlines:
        idx = _ts_to_bar_idx(ohlcv, pd.Timestamp(vline.timestamp))
        ax_main.axvline(
            idx,
            color=vline.color,
            linestyle=vline.linestyle,
            linewidth=vline.linewidth,
            label=vline.label,
            alpha=0.7,
        )

    # --- 9) Rectangles ---
    for rect in snapshot.rectangles:
        x1 = _ts_to_bar_idx(ohlcv, pd.Timestamp(rect.x_start))
        x2 = _ts_to_bar_idx(ohlcv, pd.Timestamp(rect.x_end))
        width = max(x2 - x1, 1)
        height = rect.y_top - rect.y_bottom
        patch = mpatches.FancyBboxPatch(
            (x1, rect.y_bottom),
            width,
            height,
            boxstyle="round,pad=0",
            facecolor=rect.color,
            alpha=rect.alpha,
            edgecolor=rect.color,
            linewidth=0.5,
        )
        ax_main.add_patch(patch)

    # --- 10) TrendLines ---
    for tl in snapshot.trendlines:
        xs = [_ts_to_bar_idx(ohlcv, pd.Timestamp(t)) for t, _ in tl.points]
        ys = [p for _, p in tl.points]
        ax_main.plot(
            xs,
            ys,
            color=tl.color,
            linestyle=tl.linestyle,
            linewidth=tl.linewidth,
            label=tl.label,
            alpha=0.8,
        )

    # --- 11) Legend ---
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_main.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.85)

    # --- Kết thúc ---
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    if show:
        if _is_non_interactive_backend():
            logger.info(
                "Bỏ qua plt.show() vì matplotlib backend '%s' là non-interactive.",
                plt.get_backend(),
            )
        else:
            plt.show()

    return fig
