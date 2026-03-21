"""DuckDB-backed OHLCV query utilities."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vnstock_forecast.engine.data.financial import _create_finance_views
from vnstock_forecast.engine.shared.path import DATA_PATH_STR

OHLCV_BASE_DIR = Path(DATA_PATH_STR) / "ohlcv"


def _glob_pattern() -> str:
    """Return the glob pattern that covers all partitioned parquet files."""
    return str(OHLCV_BASE_DIR / "**" / "*.parquet")


def _has_parquet_files(base_dir: Path) -> bool:
    """Check if a directory tree contains any parquet files."""
    if not base_dir.exists():
        return False
    return any(base_dir.rglob("*.parquet"))


def query_ohlcv(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    from_ts: int | None = None,
    to_ts: int | None = None,
    lookback_bars: int | None = None,
    columns: list[str] | None = None,
    order_by: str = "Symbol, Timestamp",
    limit: int | None = None,
) -> pd.DataFrame:
    """Query the local OHLCV parquet store using DuckDB."""
    if lookback_bars is not None:
        if lookback_bars <= 0:
            raise ValueError("lookback_bars phải > 0")
        if from_ts is not None:
            raise ValueError("Không dùng đồng thời from_ts và lookback_bars")

    glob = _glob_pattern()

    select_cols = ", ".join(columns) if columns else "*"
    base_sql = (
        f"SELECT {select_cols} FROM read_parquet('{glob}', hive_partitioning=true)"
    )

    conditions: list[str] = []
    params: dict = {}

    if isinstance(symbols, str):
        symbols = [symbols]
    if symbols:
        placeholders = ", ".join(f"${f'sym_{i}'}" for i in range(len(symbols)))
        conditions.append(f"Symbol IN ({placeholders})")
        for i, s in enumerate(symbols):
            params[f"sym_{i}"] = s

    if isinstance(resolutions, str):
        resolutions = [resolutions]
    if resolutions:
        placeholders = ", ".join(f"${f'res_{i}'}" for i in range(len(resolutions)))
        conditions.append(f"resolution IN ({placeholders})")
        for i, r in enumerate(resolutions):
            params[f"res_{i}"] = r

    if from_ts is not None:
        conditions.append("Timestamp >= $from_ts")
        params["from_ts"] = from_ts
    if to_ts is not None:
        conditions.append("Timestamp <= $to_ts")
        params["to_ts"] = to_ts

    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    sql = base_sql

    if lookback_bars is not None:
        params["lookback_bars"] = lookback_bars
        sql = f"""
        SELECT * EXCLUDE (_rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY resolution, Symbol
                       ORDER BY Timestamp DESC
                   ) AS _rn
            FROM ({base_sql}) _base
        ) _ranked
        WHERE _rn <= $lookback_bars
        """

    if order_by:
        sql += f" ORDER BY {order_by}"

    if limit is not None:
        sql += f" LIMIT {limit}"

    conn = duckdb.connect()
    try:
        return conn.execute(sql, params).fetchdf()
    finally:
        conn.close()


def query_latest(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
) -> pd.DataFrame:
    """Return the latest row for each (Symbol, resolution) pair."""
    glob = _glob_pattern()

    sql = f"""
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY resolution, Symbol
                   ORDER BY Timestamp DESC
               ) AS _rn
        FROM read_parquet('{glob}', hive_partitioning=true)
    ) sub
    WHERE _rn = 1
    """

    conditions: list[str] = []
    params: dict = {}

    if isinstance(symbols, str):
        symbols = [symbols]
    if symbols:
        placeholders = ", ".join(f"${f'sym_{i}'}" for i in range(len(symbols)))
        conditions.append(f"Symbol IN ({placeholders})")
        for i, s in enumerate(symbols):
            params[f"sym_{i}"] = s

    if isinstance(resolutions, str):
        resolutions = [resolutions]
    if resolutions:
        placeholders = ", ".join(f"${f'res_{i}'}" for i in range(len(resolutions)))
        conditions.append(f"resolution IN ({placeholders})")
        for i, r in enumerate(resolutions):
            params[f"res_{i}"] = r

    if conditions:
        sql = f"""
        SELECT * FROM ({sql}) _outer
        WHERE {" AND ".join(conditions)}
        """

    sql += " ORDER BY resolution, Symbol"

    conn = duckdb.connect()
    try:
        df = conn.execute(sql, params).fetchdf()
        if "_rn" in df.columns:
            df = df.drop(columns=["_rn"])
        return df
    finally:
        conn.close()


def query_ohlcv_grouped(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    from_ts: int | None = None,
    to_ts: int | None = None,
    lookback_bars: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Query OHLCV and return a nested dict grouped by resolution then symbol."""
    df = query_ohlcv(
        symbols=symbols,
        resolutions=resolutions,
        from_ts=from_ts,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
        order_by="resolution, Symbol, Timestamp",
    )

    if df.empty:
        return {}

    ohlcv_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for (resolution, symbol), group in df.groupby(["resolution", "Symbol"], sort=False):
        sub = group[ohlcv_cols].set_index("Timestamp").sort_index()
        result.setdefault(resolution, {})[symbol] = sub

    return result


def query_grouped_ohlcv(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    from_ts: int | None = None,
    to_ts: int | None = None,
    lookback_bars: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Alias of query_ohlcv_grouped for naming compatibility."""
    return query_ohlcv_grouped(
        symbols=symbols,
        resolutions=resolutions,
        from_ts=from_ts,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
    )


def query_ohlcv_lookback(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    to_ts: int | None = None,
    lookback_bars: int = 200,
    columns: list[str] | None = None,
    order_by: str = "Symbol, Timestamp",
    limit: int | None = None,
) -> pd.DataFrame:
    """Convenience API for to_ts + lookback_bars queries."""
    return query_ohlcv(
        symbols=symbols,
        resolutions=resolutions,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
        columns=columns,
        order_by=order_by,
        limit=limit,
    )


def query_ohlcv_grouped_lookback(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    to_ts: int | None = None,
    lookback_bars: int = 200,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Grouped convenience API for to_ts + lookback_bars queries."""
    return query_ohlcv_grouped(
        symbols=symbols,
        resolutions=resolutions,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
    )


def query_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute arbitrary SQL with ohlcv and finance views available."""
    ohlcv_glob = _glob_pattern()
    conn = duckdb.connect()
    try:
        if _has_parquet_files(OHLCV_BASE_DIR):
            conn.execute(
                f"CREATE VIEW ohlcv AS SELECT * FROM read_parquet('{ohlcv_glob}', hive_partitioning=true)"  # noqa E501
            )
        else:
            conn.execute(
                """
                CREATE VIEW ohlcv AS
                SELECT NULL::BIGINT AS Timestamp,
                       NULL::VARCHAR AS Symbol,
                       NULL::DOUBLE AS Open,
                       NULL::DOUBLE AS High,
                       NULL::DOUBLE AS Low,
                       NULL::DOUBLE AS Close,
                       NULL::DOUBLE AS Volume,
                       NULL::VARCHAR AS resolution
                WHERE FALSE
                """
            )

        _create_finance_views(conn)

        return conn.execute(sql, params or {}).fetchdf()
    finally:
        conn.close()
