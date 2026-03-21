"""Signal snapshot – cấu trúc dữ liệu trực quan hóa tín hiệu trên biểu đồ nến."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

# ======================================================================
#  Primitive overlay shapes
# ======================================================================


@dataclass
class HLine:
    """Đường ngang trên biểu đồ.

    Attributes:
        value:     Giá trị trục Y.
        color:     Màu đường.
        linestyle: Kiểu nét vẽ (``"-"``, ``"--"``, ``":"``, ``"-."``)
        linewidth: Độ dày.
        label:     Nhãn hiển thị trên legend.
        panel:     Panel đặt đường. 0 = giá, 2+ = subplot indicator.
    """

    value: float
    color: str = "gray"
    linestyle: str = "--"
    linewidth: float = 1.0
    label: str = ""
    panel: int = 0


@dataclass
class VLine:
    """Đường dọc trên biểu đồ.

    Attributes:
        timestamp: Thời điểm trên trục X.
        color:     Màu đường.
        linestyle: Kiểu nét vẽ.
        linewidth: Độ dày.
        label:     Nhãn hiển thị trên legend.
    """

    timestamp: datetime
    color: str = "gray"
    linestyle: str = "--"
    linewidth: float = 1.0
    label: str = ""


@dataclass
class Rectangle:
    """Vùng hình chữ nhật trên biểu đồ.

    Attributes:
        x_start:  Thời điểm bắt đầu.
        x_end:    Thời điểm kết thúc.
        y_bottom: Giá trị Y dưới.
        y_top:    Giá trị Y trên.
        color:    Màu nền.
        alpha:    Độ trong suốt.
        label:    Nhãn.
    """

    x_start: datetime
    x_end: datetime
    y_bottom: float
    y_top: float
    color: str = "blue"
    alpha: float = 0.15
    label: str = ""


@dataclass
class TrendLine:
    """Đường xu hướng qua 2+ điểm.

    Attributes:
        points:    Danh sách ``(datetime, price)`` ít nhất 2 điểm.
        color:     Màu đường.
        linestyle: Kiểu nét.
        linewidth: Độ dày.
        label:     Nhãn.
    """

    points: list[tuple[datetime, float]]
    color: str = "blue"
    linestyle: str = "-"
    linewidth: float = 1.5
    label: str = ""


# ======================================================================
#  Indicator line – dữ liệu indicator sẵn sàng plot
# ======================================================================


@dataclass
class IndicatorLine:
    """Dữ liệu indicator dạng đường, sẵn sàng plot lên mplfinance.

    Attributes:
        name:        Tên indicator (dùng làm label / ylabel).
        data:        ``pd.Series`` (index = DatetimeIndex, values = giá trị).
        color:       Màu đường.
        linestyle:   Kiểu nét (``"-"``, ``"--"``, ``":"``, ``"-."``)
        linewidth:   Độ dày.
        panel:       Panel để vẽ. 0 = overlay lên biểu đồ giá, 2+ = subplot riêng.
                     (panel 1 thường dành cho Volume.)
        ylabel:      Nhãn trục Y của subplot (nếu ``panel > 0``).
        secondary_y: Dùng trục Y phụ.
        type:        ``"line"`` hoặc ``"bar"`` (histogram).
        alpha:       Độ trong suốt.
    """

    name: str
    data: pd.Series
    color: str = "blue"
    linestyle: str = "-"
    linewidth: float = 1.0
    panel: int = 0
    ylabel: str = ""
    secondary_y: bool = False
    type: str = "line"
    alpha: float = 1.0


# ======================================================================
#  PlotOverlays – collection overlay từ indicator / strategy
# ======================================================================


@dataclass
class PlotOverlays:
    """Collection overlay trả về bởi indicator / BaseTechnique.

    Dùng để gom nhóm tất cả dữ liệu vẽ từ một indicator, sau đó merge
    vào ``SignalSnapshot``.
    """

    indicators: list[IndicatorLine] = field(default_factory=list)
    hlines: list[HLine] = field(default_factory=list)
    vlines: list[VLine] = field(default_factory=list)
    rectangles: list[Rectangle] = field(default_factory=list)
    trendlines: list[TrendLine] = field(default_factory=list)

    def merge(self, other: PlotOverlays) -> PlotOverlays:
        """Gộp *other* vào, trả về bản mới (không mutate bản gốc)."""
        return PlotOverlays(
            indicators=self.indicators + other.indicators,
            hlines=self.hlines + other.hlines,
            vlines=self.vlines + other.vlines,
            rectangles=self.rectangles + other.rectangles,
            trendlines=self.trendlines + other.trendlines,
        )


# ======================================================================
#  SignalSnapshot – tập hợp mọi dữ liệu cần thiết để plot 1 signal
# ======================================================================


@dataclass
class SignalSnapshot:
    """Snapshot dữ liệu tại thời điểm phát signal, phục vụ plot biểu đồ nến.

    Chứa đủ thông tin để ``plot_signal()`` render toàn bộ biểu đồ mà
    không cần truy vấn thêm dữ liệu – trừ khi bật ``extend_to_limit``
    (lúc đó plotter sẽ query thêm bars đến ``time_limit``).

    Attributes:
        ohlcv:       DataFrame OHLCV (DatetimeIndex, cột OHLCV).
        entry:       Giá vào lệnh.
        entry_time:  Thời điểm vào lệnh.
        stop_loss:   Giá cắt lỗ.
        take_profit: Giá chốt lời.
        exit_price:  Giá thoát lệnh (nếu đã có).
        exit_time:   Thời điểm thoát lệnh (nếu đã có).
        signal_time: Thời điểm phát signal.
        time_limit:  Deadline vị thế (query thêm dữ liệu để xem kết quả).
        resolution:  Resolution OHLCV (``"D"``, ``"60"``, …).
        symbol:      Mã cổ phiếu.
        hlines:      Đường ngang bổ sung.
        vlines:      Đường dọc bổ sung.
        rectangles:  Vùng chữ nhật.
        trendlines:  Đường xu hướng.
        indicators:  Dữ liệu indicator dạng đường / histogram.
    """

    ohlcv: pd.DataFrame
    entry: Optional[float] = None
    entry_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    buy_points: list[tuple[datetime, float]] = field(default_factory=list)
    sell_points: list[tuple[datetime, float]] = field(default_factory=list)
    signal_time: Optional[datetime] = None
    time_limit: Optional[datetime] = None
    resolution: str = "D"
    symbol: str = ""
    hlines: list[HLine] = field(default_factory=list)
    vlines: list[VLine] = field(default_factory=list)
    rectangles: list[Rectangle] = field(default_factory=list)
    trendlines: list[TrendLine] = field(default_factory=list)
    indicators: list[IndicatorLine] = field(default_factory=list)
