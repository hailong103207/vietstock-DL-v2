"""Public query APIs for OHLCV and financial datasets."""

from __future__ import annotations

import pandas as pd

from vnstock_forecast.engine.data.financial import (
    list_financial_metrics as _list_financial_metrics,
)
from vnstock_forecast.engine.data.financial import query_financial as _query_financial
from vnstock_forecast.engine.data.financial import (
    query_financial_by_statement as _query_financial_by_statement,
)
from vnstock_forecast.engine.data.financial import (
    search_financial_metrics as _search_financial_metrics,
)
from vnstock_forecast.engine.data.financial import (
    validate_financial_expression as _validate_financial_expression,
)
from vnstock_forecast.engine.data.ohlcv import (
    query_grouped_ohlcv as _query_grouped_ohlcv,
)
from vnstock_forecast.engine.data.ohlcv import query_latest as _query_latest
from vnstock_forecast.engine.data.ohlcv import query_ohlcv as _query_ohlcv
from vnstock_forecast.engine.data.ohlcv import (
    query_ohlcv_grouped as _query_ohlcv_grouped,
)
from vnstock_forecast.engine.data.ohlcv import (
    query_ohlcv_grouped_lookback as _query_ohlcv_grouped_lookback,
)
from vnstock_forecast.engine.data.ohlcv import (
    query_ohlcv_lookback as _query_ohlcv_lookback,
)
from vnstock_forecast.engine.data.ohlcv import query_sql as _query_sql


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
    return _query_ohlcv(
        symbols=symbols,
        resolutions=resolutions,
        from_ts=from_ts,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
        columns=columns,
        order_by=order_by,
        limit=limit,
    )


def query_latest(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
) -> pd.DataFrame:
    """Return the latest row for each (Symbol, resolution) pair."""
    return _query_latest(symbols=symbols, resolutions=resolutions)


def query_ohlcv_grouped(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    from_ts: int | None = None,
    to_ts: int | None = None,
    lookback_bars: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Query OHLCV and return nested dict grouped by resolution then symbol."""
    return _query_ohlcv_grouped(
        symbols=symbols,
        resolutions=resolutions,
        from_ts=from_ts,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
    )


def query_grouped_ohlcv(
    symbols: list[str] | str | None = None,
    resolutions: list[str] | str | None = None,
    from_ts: int | None = None,
    to_ts: int | None = None,
    lookback_bars: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Alias of query_ohlcv_grouped for naming compatibility."""
    return _query_grouped_ohlcv(
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
    return _query_ohlcv_lookback(
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
    return _query_ohlcv_grouped_lookback(
        symbols=symbols,
        resolutions=resolutions,
        to_ts=to_ts,
        lookback_bars=lookback_bars,
    )


def query_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute an arbitrary SQL query against local parquet stores."""
    return _query_sql(sql=sql, params=params)


def query_financial(
    symbols: list[str] | str | None = None,
    statements: list[str] | str | None = None,
    metrics: list[str] | str | None = None,
    periods: list[str] | str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    order_by: str = "symbol, statement, metric, period",
    limit: int | None = None,
) -> pd.DataFrame:
    """Query financial data in long format for flexible stock screening."""
    return _query_financial(
        symbols=symbols,
        statements=statements,
        metrics=metrics,
        periods=periods,
        min_value=min_value,
        max_value=max_value,
        order_by=order_by,
        limit=limit,
    )


def list_financial_metrics(
    symbols: list[str] | str | None = "VHM",
) -> pd.DataFrame:
    """List all available financial metrics with statement name."""
    return _list_financial_metrics(symbols=symbols)


def search_financial_metrics(
    metric_query: str | None = None,
    statement_query: str | None = None,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Search metric catalog by metric and/or statement."""
    return _search_financial_metrics(
        metric_query=metric_query,
        statement_query=statement_query,
        symbols=symbols,
    )


def validate_financial_expression(
    statement: str,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Validate a financial expression and return resolved metric references."""
    return _validate_financial_expression(statement=statement, symbols=symbols)


def query_financial_by_statement(
    symbols: list[str],
    statement: str,
) -> pd.DataFrame:
    """Filter symbols by a simple financial expression."""
    return _query_financial_by_statement(symbols=symbols, statement=statement)
