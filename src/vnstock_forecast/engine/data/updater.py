"""
OHLCV data updater — Parquet + DuckDB local store.

Storage layout (Hive-style partitioning):
    data/ohlcv/
        resolution=D/
            VHM.parquet
            SHB.parquet
        resolution=W/
            ...

Each .parquet file stores one (symbol, resolution) pair with columns:
    Timestamp | Symbol | Open | High | Low | Close | Volume

Update strategy:
    1. If parquet already exists → read last timestamp
        → fetch only newer data → append & deduplicate.
    2. If parquet does not exist → fetch from (now − lookback_days) → write new file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

import vnstock_forecast.config  # noqa: F401 — registers ${symbols:...} resolver
from vnstock_forecast.engine.client.vietcap.financial import FinancialReport
from vnstock_forecast.engine.client.vietstock import OHLCV
from vnstock_forecast.engine.schemas import UpdaterConfig
from vnstock_forecast.engine.schemas.data import (
    DataClient,
    FinancialUpdaterConfig,
    OhlcvUpdaterConfig,
)
from vnstock_forecast.engine.shared.path import CONFIG_PATH_STR, DATA_PATH_STR
from vnstock_forecast.engine.utils import time_utils

logger = logging.getLogger(__name__)

OHLCV_BASE_DIR = Path(DATA_PATH_STR) / "ohlcv"
FINANCE_BASE_DIR = Path(DATA_PATH_STR) / "finance"

FINANCIAL_DATASETS: dict[str, str] = {
    "cash_flow": "get_cash_flow",
    "income_statement": "get_income_statement",
    "balance_sheet": "get_balance_sheet",
    "footnote": "get_footnote",
    "statistics": "get_statistics_financial",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parquet_path(resolution: str, symbol: str) -> Path:
    """Return the canonical parquet file path for a (resolution, symbol) pair."""
    return OHLCV_BASE_DIR / f"resolution={resolution}" / f"{symbol}.parquet"


def _read_existing(path: Path) -> pd.DataFrame | None:
    """Read an existing parquet file, or return None if it doesn't exist."""
    if path.exists():
        df = pd.read_parquet(path)
        if not df.empty:
            return df
    return None


def _merge_and_deduplicate(
    existing: pd.DataFrame | None, new: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge new data into existing data, drop duplicates on Timestamp,
    keep the latest version, and sort chronologically.
    """
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new.copy()

    combined = (
        combined.drop_duplicates(subset=["Timestamp", "Symbol"], keep="last")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    return combined


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def _save_financial_parquet(
    symbol: str,
    statement_name: str,
    df: pd.DataFrame,
) -> None:
    """Write a financial statement DataFrame to data/finance/<SYMBOL>/<statement>.parquet."""
    output_path = FINANCE_BASE_DIR / symbol.upper() / f"{statement_name}.parquet"

    if df.index.name:
        to_save = df.reset_index()
    else:
        to_save = df.copy()

    to_save.insert(0, "statement", statement_name)
    to_save.insert(0, "symbol", symbol.upper())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_save.to_parquet(output_path, index=False, engine="pyarrow")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_symbol(
    client: OHLCV,
    symbol: str,
    resolution: str,
    lookback_days: int = 365,
) -> pd.DataFrame | None:
    """
    Fetch and persist the latest OHLCV data for a single (symbol, resolution) pair.

    Supports **bidirectional** updates:
      - Forward: fetch data newer than the local max timestamp.
      - Backfill: if lookback_days extends earlier than the local min timestamp,
        fetch the missing historical gap as well.

    Args:
        client: An initialised OHLCV API client.
        symbol: Stock ticker (e.g. "VHM").
        resolution: Vietstock resolution string (e.g. "D", "5", "W").
        lookback_days: How far back to fetch from today.

    Returns:
        The updated DataFrame, or None if nothing was fetched.
    """
    path = _parquet_path(resolution, symbol)
    existing_df = _read_existing(path)

    now_ts = time_utils.get_current_timestamp()
    lookback_ts = time_utils.add_days_to_timestamp(
        time_utils.get_current_date_timestamp(), -lookback_days
    )

    parts: list[pd.DataFrame] = []

    if existing_df is not None:
        local_min_ts = int(existing_df["Timestamp"].min())
        local_max_ts = int(existing_df["Timestamp"].max())

        # --- Backfill: fetch older data if lookback extends before local min ---
        if lookback_ts < local_min_ts:
            logger.info(
                "[%s/%s] Backfill: local starts at %s, lookback wants %s.",
                resolution,
                symbol,
                time_utils.timestamp_to_str(local_min_ts),
                time_utils.timestamp_to_str(lookback_ts),
            )
            backfill_df = client.fetch(
                lookback_ts, local_min_ts - 1, symbol, resolution
            )
            if backfill_df is not None and not backfill_df.empty:
                parts.append(backfill_df)
                logger.info(
                    "[%s/%s] Backfill fetched %d rows.",
                    resolution,
                    symbol,
                    len(backfill_df),
                )

        # --- Forward: fetch newer data after local max ---
        forward_from = local_max_ts + 1
        if forward_from < now_ts:
            logger.info(
                "[%s/%s] Forward: local ends at %s — fetching to now.",
                resolution,
                symbol,
                time_utils.timestamp_to_str(local_max_ts),
            )
            forward_df = client.fetch(forward_from, now_ts, symbol, resolution)
            if forward_df is not None and not forward_df.empty:
                parts.append(forward_df)
                logger.info(
                    "[%s/%s] Forward fetched %d rows.",
                    resolution,
                    symbol,
                    len(forward_df),
                )
        else:
            logger.info("[%s/%s] Already up-to-date (forward).", resolution, symbol)
    else:
        # --- No local data: full fetch from lookback ---
        logger.info(
            "[%s/%s] No local data — fetching last %d days.",
            resolution,
            symbol,
            lookback_days,
        )
        full_df = client.fetch(lookback_ts, now_ts, symbol, resolution)
        if full_df is not None and not full_df.empty:
            parts.append(full_df)

    if not parts:
        logger.warning("[%s/%s] No new data fetched.", resolution, symbol)
        return existing_df

    merged = _merge_and_deduplicate(existing_df, pd.concat(parts, ignore_index=True))
    _save_parquet(merged, path)
    logger.info(
        "[%s/%s] Saved %d rows → %s",
        resolution,
        symbol,
        len(merged),
        path,
    )
    return merged


def _update_ohlcv(cfg: UpdaterConfig) -> tuple[int, int]:
    """
    Run a full OHLCV update for every (symbol × resolution) in the config.

    Args:
        cfg: UpdaterConfig with symbols, resolutions, and lookback_days.

    Returns:
        True if ALL (symbol, resolution) pairs succeeded, False otherwise.
    """
    ohlcv_cfg = cfg.ohlcv
    if not ohlcv_cfg.update:
        logger.info("OHLCV update is disabled.")
        return (0, 0)

    if ohlcv_cfg.client != DataClient.vietstock:
        logger.error(
            "OHLCV updater only supports 'vietstock' client, got '%s'.",
            ohlcv_cfg.client,
        )
        return (0, 1)

    client = OHLCV()
    symbols: list[str] = list(ohlcv_cfg.symbols)
    resolutions: list[str] = list(ohlcv_cfg.resolutions)
    lookback_days: int = ohlcv_cfg.lookback_days

    total = len(symbols) * len(resolutions)
    logger.info(
        "Starting OHLCV update: %d symbols × %d resolutions = %d jobs",
        len(symbols),
        len(resolutions),
        total,
    )

    success_count = 0
    fail_count = 0

    for resolution in resolutions:
        for symbol in symbols:
            try:
                result = update_symbol(client, symbol, resolution, lookback_days)
                if result is not None:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                logger.exception("[%s/%s] Unexpected error.", resolution, symbol)
                fail_count += 1

    logger.info(
        "OHLCV update complete: %d/%d succeeded, %d failed.",
        success_count,
        total,
        fail_count,
    )
    return success_count, fail_count


def _update_financial(cfg: UpdaterConfig) -> tuple[int, int]:
    """
    Run financial update for configured symbols.

    Stored files:
        data/finance/<SYMBOL>/cash_flow.parquet
        data/finance/<SYMBOL>/income_statement.parquet
        data/finance/<SYMBOL>/balance_sheet.parquet
        data/finance/<SYMBOL>/footnote.parquet
        data/finance/<SYMBOL>/statistics.parquet
    """
    financial_cfg = cfg.financial
    if not financial_cfg.update:
        logger.info("Financial update is disabled.")
        return (0, 0)

    if financial_cfg.client != DataClient.vietcap:
        logger.error(
            "Financial updater only supports 'vietcap' client, got '%s'.",
            financial_cfg.client,
        )
        return (0, 1)

    client = FinancialReport()
    symbols: list[str] = list(financial_cfg.symbols)

    total = len(symbols) * len(FINANCIAL_DATASETS)
    logger.info(
        "Starting financial update: %d symbols × %d datasets = %d jobs",
        len(symbols),
        len(FINANCIAL_DATASETS),
        total,
    )

    success_count = 0
    fail_count = 0

    for symbol in symbols:
        for statement_name, method_name in FINANCIAL_DATASETS.items():
            fetcher = getattr(client, method_name)
            try:
                statement_df = fetcher(symbol=symbol)
                if statement_df is None or statement_df.empty:
                    logger.warning(
                        "[finance/%s/%s] Empty data.", symbol, statement_name
                    )
                    fail_count += 1
                    continue

                _save_financial_parquet(symbol, statement_name, statement_df)
                logger.info(
                    "[finance/%s/%s] Saved %d rows.",
                    symbol,
                    statement_name,
                    len(statement_df),
                )
                success_count += 1
            except Exception:
                logger.exception(
                    "[finance/%s/%s] Unexpected error.", symbol, statement_name
                )
                fail_count += 1

    logger.info(
        "Financial update complete: %d/%d succeeded, %d failed.",
        success_count,
        total,
        fail_count,
    )
    return success_count, fail_count


def update(cfg: UpdaterConfig) -> bool:
    """Run enabled updater jobs (OHLCV and/or financial)."""
    ohlcv_success, ohlcv_fail = _update_ohlcv(cfg)
    financial_success, financial_fail = _update_financial(cfg)

    total_success = ohlcv_success + financial_success
    total_fail = ohlcv_fail + financial_fail

    logger.info(
        "All updates complete: %d succeeded, %d failed.",
        total_success,
        total_fail,
    )
    return total_fail == 0


def update_ohlcv(
    symbols: list[str] | str,
    resolutions: list[str] | str,
    lookback_days: int = 365,
) -> tuple[int, int]:
    """
    Public helper để chỉ cập nhật OHLCV cho danh sách symbol/resolution.

    Returns:
        Tuple ``(success_count, fail_count)``.
    """
    symbols_list = [symbols] if isinstance(symbols, str) else list(symbols)
    resolutions_list = (
        [resolutions] if isinstance(resolutions, str) else list(resolutions)
    )

    cfg = UpdaterConfig(
        ohlcv=OhlcvUpdaterConfig(
            update=True,
            client=DataClient.vietstock,
            symbols=symbols_list,
            resolutions=resolutions_list,
            lookback_days=lookback_days,
        ),
        financial=FinancialUpdaterConfig(
            update=False,
            client=DataClient.vietcap,
            symbols=[],
        ),
    )
    return _update_ohlcv(cfg)


@hydra.main(version_base=None, config_path=CONFIG_PATH_STR, config_name="config")
def main(cfg: DictConfig):
    from vnstock_forecast.engine.schemas.config import to_app_config

    app_cfg = to_app_config(cfg)
    update_cfg = app_cfg.data.updater
    update(update_cfg)


if __name__ == "__main__":
    main()
