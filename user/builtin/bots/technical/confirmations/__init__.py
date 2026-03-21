"""
Confirmation signals – các hàm kiểm tra tín hiệu xác nhận.

Mỗi hàm nhận DataFrame OHLCV và trả về :class:`ConfirmationResult`
chứa::

    confirmed: bool   – có xác nhận không
    boost: float      – lượng confidence cộng thêm (0 → 0.1)
    reason: str       – mô tả

Usage điển hình trong một technique::

    from vnstock_forecast.builtin.signal_based.technical.confirmations import (
        apply_confirmations,
        check_volume_surge,
        check_breakout_resistance,
    )

    base_conf = 0.5  # hoặc tính từ tín hiệu chính
    final_conf, reasons = apply_confirmations(base_conf, [
        check_volume_surge(df),
        check_breakout_resistance(df, current_price=price),
    ])
"""

from .base import MAX_BOOST_PER_CONFIRMATION, ConfirmationResult, apply_confirmations
from .breakout_resistance import check_breakout_resistance
from .no_fvg import check_no_large_fvg, detect_fvg_zones
from .volume_surge import check_volume_surge

__all__ = [
    "MAX_BOOST_PER_CONFIRMATION",
    "ConfirmationResult",
    "apply_confirmations",
    "check_volume_surge",
    "check_breakout_resistance",
    "check_no_large_fvg",
    "detect_fvg_zones",
]
