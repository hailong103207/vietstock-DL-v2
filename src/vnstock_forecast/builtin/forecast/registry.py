"""Core registry for technical techniques.

The registry lives in ``src`` as stable system core.
Technique implementations themselves are user-owned (``user/``).
"""

from __future__ import annotations

from typing import Callable

from vnstock_forecast.engine.shared.user_bridge import ensure_technique_modules_loaded

from .signal import Signal  # noqa: F401


class BaseTechniqueProtocol:
    """Protocol-like minimal base for runtime registration checks."""

    name: str = "BaseTechnique"


_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator đăng ký technique class vào global registry."""

    def decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(
                f"Technique '{name}' đã được đăng ký bởi {_REGISTRY[name].__name__}. "
                f"Không thể đăng ký {cls.__name__} cùng tên."
            )
        _REGISTRY[name] = cls
        if getattr(cls, "name", "BaseTechnique") == "BaseTechnique":
            cls.name = name
        return cls

    return decorator


def get_technique(name: str) -> type:
    """Tìm technique class theo tên đã đăng ký."""
    ensure_technique_modules_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(trống)"
        raise KeyError(f"Technique '{name}' chưa đăng ký. Có sẵn: {available}")
    return _REGISTRY[name]


def get_all_techniques() -> dict[str, type]:
    """Trả về dict tất cả technique đã đăng ký: ``{name: class}``."""
    ensure_technique_modules_loaded()
    return dict(_REGISTRY)


def get_list_techniques() -> list:
    """Trả về list instance của tất cả technique đã đăng ký."""
    ensure_technique_modules_loaded()
    return [cls() for cls in _REGISTRY.values()]


def list_technique_names() -> list[str]:
    """Trả về danh sách tên tất cả technique đã đăng ký."""
    ensure_technique_modules_loaded()
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Xóa toàn bộ registry. Chỉ dùng cho testing."""
    _REGISTRY.clear()
