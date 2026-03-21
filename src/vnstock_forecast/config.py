"""
Project-level config helpers.

This module owns every concern that is specific to this project's layout:
- locating the ``config/data/discovery/symbols`` folder
- scanning symbol YAML files (``discover_symbols``)
- registering the ``${symbols:key}`` OmegaConf resolver
- providing a project-aware ``load_config`` / ``load_app_config``

Generic, project-agnostic Hydra/OmegaConf utilities live in
``vnstock_forecast.utils.config_utils``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

import vnstock_forecast.engine.schemas.config as schemas_cfg  # noqa F401: registers AppConfig
from vnstock_forecast.engine.schemas.config import AppConfig  # noqa F401
from vnstock_forecast.engine.utils.config_utils import get_project_root
from vnstock_forecast.engine.utils.config_utils import (
    load_config as _load_config_generic,
)
from vnstock_forecast.engine.utils.config_utils import print_config

__all__ = [
    "discover_symbols",
    "load_config",
    "print_config",
]

# ---------------------------------------------------------------------------
# Symbol discovery
# ---------------------------------------------------------------------------

_SYMBOLS_DIR = "config/data/discovery/symbols"


def discover_symbols(symbols_dir: Path) -> Dict[str, Any]:
    """
    Recursively scan *symbols_dir* for ``*.yaml`` files and return a dict
    keyed by the file stem (lower-cased).

    Layout example::

        symbols/
            VN30.yaml           → key "vn30"
            STEEL/
                STEEL_L4.yaml   → key "steel_l4"

    Returns:
        dict[str, list[str]]
    """
    result: Dict[str, Any] = {}
    for yaml_file in sorted(symbols_dir.rglob("*.yaml")):
        key = yaml_file.stem.lower()
        result[key] = yaml.safe_load(yaml_file.read_text()) or []
    return result


# ---------------------------------------------------------------------------
# OmegaConf resolver  ${symbols:vn30}
# ---------------------------------------------------------------------------


def _symbols_resolver(key: str) -> List[str]:
    symbols_dir = get_project_root() / _SYMBOLS_DIR
    return discover_symbols(symbols_dir).get(key.lower(), [])


# Registered once on first import of this module — works for both
# load_config() and @hydra.main without any further setup.
OmegaConf.register_new_resolver("symbols", _symbols_resolver, replace=True)


# ---------------------------------------------------------------------------
# Project-aware config loader
# ---------------------------------------------------------------------------


def list_symbols_lists() -> List[str]:
    """List all available symbol list keys from discovery directory."""
    symbols_dir = get_project_root() / _SYMBOLS_DIR
    if not symbols_dir.exists():
        return []
    return sorted({yaml_file.stem.lower() for yaml_file in symbols_dir.rglob("*.yaml")})


def search_symbols_lists(query: str) -> List[str]:
    """Search symbol list keys that contain the query string."""
    all_lists = list_symbols_lists()
    query_lower = query.lower()
    return [key for key in all_lists if query_lower in key]


def query_symbols_list(symbols_list_names: str | List[str]) -> List[str]:
    """Query symbol list from discovery directory."""
    symbols_dir = get_project_root() / _SYMBOLS_DIR
    if isinstance(symbols_list_names, str):
        symbols_list_names = [symbols_list_names]
    result = []
    for name in symbols_list_names:
        result.extend(discover_symbols(symbols_dir).get(name.lower(), []))
    # Remove duplicates while preserving order
    seen = set()
    result = [x for x in result if not (x in seen or seen.add(x))]
    return result


def load_config(
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    resolve: bool = True,
) -> DictConfig:
    """Load Hydra config and inject all discovered symbol sets."""
    cfg = _load_config_generic(
        config_name=config_name, overrides=overrides, resolve=False
    )

    symbols_dir = get_project_root() / _SYMBOLS_DIR
    if symbols_dir.exists():
        with open_dict(cfg):
            cfg.data.discovery.symbols = OmegaConf.create(discover_symbols(symbols_dir))

    if resolve:
        OmegaConf.resolve(cfg)

    return cfg


def load_app_config() -> AppConfig:
    """Load config and convert to AppConfig."""
    cfg = load_config()
    return schemas_cfg.to_app_config(cfg)
