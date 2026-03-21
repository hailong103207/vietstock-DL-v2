"""DuckDB-backed financial query utilities."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from vnstock_forecast.engine.shared.path import DATA_PATH_STR

FINANCE_BASE_DIR = Path(DATA_PATH_STR) / "finance"
FINANCE_METRICS_CACHE_PATH = FINANCE_BASE_DIR / "metrics.csv"

FINANCE_METADATA_COLUMNS = {
    "symbol",
    "statement",
    "metric",
    "description",
    "filename",
}

_EXPR_TOKEN_REGEX = re.compile(
    r"\s*(>=|<=|==|!=|>|<|\(|\)|\band\b|\bor\b|\bnot\b|\d+\.\d+|\d+|[A-Za-z_][A-Za-z0-9_./%\-]*)\s*",
    re.IGNORECASE,
)


def _finance_glob_pattern() -> str:
    """Return the glob pattern for all financial parquet files."""
    return str(FINANCE_BASE_DIR / "**" / "*.parquet")


def _has_parquet_files(base_dir: Path) -> bool:
    """Check if a directory tree contains any parquet files."""
    if not base_dir.exists():
        return False
    return any(base_dir.rglob("*.parquet"))


def _to_list(value: list[str] | str | None) -> list[str]:
    """Normalize optional string/list input to list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _escape_sql_identifier(name: str) -> str:
    """Escape a SQL identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _build_finance_long_sql(conn: duckdb.DuckDBPyConnection, finance_glob: str) -> str:
    """Build SQL to transform finance_raw wide table into long format without UNPIVOT."""
    schema_df = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{finance_glob}', union_by_name=true, filename=true)"
    ).fetchdf()
    all_columns = [str(column) for column in schema_df["column_name"].tolist()]

    period_columns = [
        column for column in all_columns if column not in FINANCE_METADATA_COLUMNS
    ]

    if not period_columns:
        return """
        SELECT NULL::VARCHAR AS symbol,
               NULL::VARCHAR AS statement,
               NULL::VARCHAR AS metric,
               NULL::VARCHAR AS description,
               NULL::VARCHAR AS period,
               NULL::DOUBLE AS value
        WHERE FALSE
        """

    symbol_expr = (
        "COALESCE(CAST(symbol AS VARCHAR), "
        "regexp_extract(filename, '.*/finance/([^/]+)/[^/]+\\.parquet', 1))"
    )
    statement_expr = (
        "COALESCE(CAST(statement AS VARCHAR), "
        "regexp_extract(filename, '.*/finance/[^/]+/([^/]+)\\.parquet', 1))"
    )

    selects: list[str] = []
    for period_col in period_columns:
        period_literal = period_col.replace("'", "''")
        period_identifier = _escape_sql_identifier(period_col)
        selects.append(
            f"""
            SELECT
                {symbol_expr} AS symbol,
                {statement_expr} AS statement,
                CAST(metric AS VARCHAR) AS metric,
                CAST(description AS VARCHAR) AS description,
                '{period_literal}' AS period,
                TRY_CAST({period_identifier} AS DOUBLE) AS value
            FROM finance_raw
            """
        )

    union_sql = "\nUNION ALL\n".join(selects)
    return f"SELECT * FROM ({union_sql}) _long WHERE value IS NOT NULL"


def _get_finance_period_columns(
    conn: duckdb.DuckDBPyConnection,
    finance_glob: str,
) -> list[str]:
    """Return ordered period columns from financial parquet schema."""
    schema_df = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{finance_glob}', union_by_name=true, filename=true)"
    ).fetchdf()
    all_columns = [str(column) for column in schema_df["column_name"].tolist()]
    period_columns = [
        column for column in all_columns if column not in FINANCE_METADATA_COLUMNS
    ]
    return sorted(period_columns, reverse=True)


def _build_latest_value_expression(period_columns: list[str]) -> str:
    """Build SQL expression that resolves latest non-null value across periods."""
    if not period_columns:
        return "NULL::DOUBLE"

    casted_period_values = [
        f"TRY_CAST({_escape_sql_identifier(period_col)} AS DOUBLE)"
        for period_col in period_columns
    ]
    return f"COALESCE({', '.join(casted_period_values)})"


def _create_finance_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create finance_raw and finance_long views in the given connection."""
    if _has_parquet_files(FINANCE_BASE_DIR):
        finance_glob = _finance_glob_pattern()
        conn.execute(
            f"CREATE VIEW finance_raw AS SELECT * FROM read_parquet('{finance_glob}', union_by_name=true, filename=true)"  # noqa E501
        )
        finance_long_sql = _build_finance_long_sql(conn, finance_glob)
        conn.execute(f"CREATE VIEW finance_long AS {finance_long_sql}")
        return

    conn.execute(
        """
        CREATE VIEW finance_raw AS
        SELECT NULL::VARCHAR AS symbol,
               NULL::VARCHAR AS statement,
               NULL::VARCHAR AS metric,
               NULL::VARCHAR AS description,
               NULL::VARCHAR AS filename
        WHERE FALSE
        """
    )
    conn.execute(
        """
        CREATE VIEW finance_long AS
        SELECT NULL::VARCHAR AS symbol,
               NULL::VARCHAR AS statement,
               NULL::VARCHAR AS metric,
               NULL::VARCHAR AS description,
               NULL::VARCHAR AS period,
               NULL::DOUBLE AS value
        WHERE FALSE
        """
    )


def _get_financial_metric_catalog(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Return distinct (statement, metric, description) available in finance_long."""
    normalized_symbols = _to_list(symbols)

    sql = """
    SELECT DISTINCT
           CAST(statement AS VARCHAR) AS statement,
           CAST(metric AS VARCHAR) AS metric,
           CAST(description AS VARCHAR) AS description
    FROM finance_long
    """

    params: dict[str, str] = {}
    if normalized_symbols:
        placeholders = ", ".join(
            f"${f'sym_{i}'}" for i in range(len(normalized_symbols))
        )
        sql += f" WHERE symbol IN ({placeholders})"
        for i, symbol in enumerate(normalized_symbols):
            params[f"sym_{i}"] = symbol.upper()

    sql += " ORDER BY statement, metric"
    return conn.execute(sql, params).fetchdf()


def _build_financial_metric_cache(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build metric cache with symbol dimension for fast subsequent filtering."""
    return conn.execute(
        """
        SELECT DISTINCT
               UPPER(CAST(symbol AS VARCHAR)) AS symbol,
               CAST(statement AS VARCHAR) AS statement,
               CAST(metric AS VARCHAR) AS metric,
               CAST(description AS VARCHAR) AS description
        FROM finance_long
        WHERE symbol IS NOT NULL
          AND statement IS NOT NULL
          AND metric IS NOT NULL
        ORDER BY symbol, statement, metric
        """
    ).fetchdf()


def _load_financial_metric_cache() -> pd.DataFrame | None:
    """Load financial metric cache from disk if available and valid."""
    if not FINANCE_METRICS_CACHE_PATH.exists():
        return None

    try:
        cached = pd.read_csv(FINANCE_METRICS_CACHE_PATH)
    except Exception:
        return None

    required_columns = {"symbol", "statement", "metric", "description"}
    if not required_columns.issubset(set(cached.columns)):
        return None

    return cached


def _build_financial_latest_values_cache(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Build symbol/statement/metric latest-value cache from wide financial parquet."""
    finance_glob = _finance_glob_pattern()
    period_columns = _get_finance_period_columns(conn, finance_glob)
    latest_value_expr = _build_latest_value_expression(period_columns)

    symbol_expr = (
        "UPPER(COALESCE(CAST(symbol AS VARCHAR), "
        "regexp_extract(filename, '.*/finance/([^/]+)/[^/]+\\.parquet', 1)))"
    )
    statement_expr = (
        "LOWER(COALESCE(CAST(statement AS VARCHAR), "
        "regexp_extract(filename, '.*/finance/[^/]+/([^/]+)\\.parquet', 1)))"
    )

    sql = f"""
    SELECT {symbol_expr} AS symbol,
           {statement_expr} AS statement,
           LOWER(CAST(metric AS VARCHAR)) AS metric,
           {latest_value_expr} AS value
    FROM read_parquet('{finance_glob}', union_by_name=true, filename=true)
    WHERE metric IS NOT NULL
      AND {latest_value_expr} IS NOT NULL
    ORDER BY symbol, statement, metric
    """

    return conn.execute(sql).fetchdf()


def _materialize_metric_catalog_from_cache(
    cache_df: pd.DataFrame,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Convert cached symbol-level rows to public metric catalog shape."""
    if cache_df.empty:
        return pd.DataFrame(columns=["statement", "metric", "description"])

    filtered = cache_df
    normalized_symbols = [item.upper() for item in _to_list(symbols)]
    if normalized_symbols:
        filtered = filtered[
            filtered["symbol"].astype(str).str.upper().isin(normalized_symbols)
        ]

    if filtered.empty:
        return pd.DataFrame(columns=["statement", "metric", "description"])

    return (
        filtered[["statement", "metric", "description"]]
        .drop_duplicates()
        .sort_values(["statement", "metric"])
        .reset_index(drop=True)
    )


def _tokenize_financial_expression(expression: str) -> list[str]:
    """Tokenize a simple financial boolean expression."""
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = _EXPR_TOKEN_REGEX.match(expression, position)
        if not match:
            raise ValueError(
                f"Biểu thức không hợp lệ gần vị trí {position}: '{expression[position:position + 20]}'"
            )
        token = match.group(1)
        if token:
            tokens.append(token)
        position = match.end()
    return tokens


def _parse_financial_expression(tokens: list[str]) -> tuple:
    """Parse tokens into a tiny AST supporting and/or/not/comparisons."""
    index = 0

    def _peek() -> str | None:
        if index >= len(tokens):
            return None
        return tokens[index]

    def _next() -> str:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("Biểu thức chưa đầy đủ")
        token = tokens[index]
        index += 1
        return token

    def _parse_expr() -> tuple:
        return _parse_or()

    def _parse_or() -> tuple:
        left = _parse_and()
        while True:
            token = _peek()
            if token is None or token.lower() != "or":
                break
            _next()
            right = _parse_and()
            left = ("or", left, right)
        return left

    def _parse_and() -> tuple:
        left = _parse_not()
        while True:
            token = _peek()
            if token is None or token.lower() != "and":
                break
            _next()
            right = _parse_not()
            left = ("and", left, right)
        return left

    def _parse_not() -> tuple:
        token = _peek()
        if token is not None and token.lower() == "not":
            _next()
            return ("not", _parse_not())
        return _parse_comparison()

    def _parse_comparison() -> tuple:
        left = _parse_atom()
        token = _peek()
        if token in {">", ">=", "<", "<=", "==", "!="}:
            operator = _next()
            right = _parse_atom()
            return ("cmp", operator, left, right)
        return left

    def _parse_atom() -> tuple:
        token = _peek()
        if token is None:
            raise ValueError("Biểu thức chưa đầy đủ")
        if token == "(":
            _next()
            node = _parse_expr()
            if _peek() != ")":
                raise ValueError("Thiếu dấu ')' trong biểu thức")
            _next()
            return node
        token = _next()
        if re.fullmatch(r"\d+\.\d+|\d+", token):
            return ("num", token)
        if token.lower() in {"and", "or", "not"}:
            raise ValueError(f"Token '{token}' không đúng vị trí")
        return ("id", token)

    ast = _parse_expr()
    if index != len(tokens):
        raise ValueError(f"Biểu thức còn token thừa: '{tokens[index]}'")
    return ast


def _normalize_metric_ref(raw: str) -> tuple[str | None, str]:
    """Normalize a metric reference token to (statement, metric)."""
    token = raw.strip().lower()
    if "." not in token:
        return None, token
    statement, metric = token.split(".", 1)
    if not statement or not metric:
        raise ValueError(f"Metric không hợp lệ: '{raw}'")
    return statement, metric


def _resolve_metric_token(
    raw_token: str,
    metric_catalog: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve metric token to an exact (statement, metric)."""
    statement, metric = _normalize_metric_ref(raw_token)

    catalog = metric_catalog.copy()
    catalog["statement_norm"] = catalog["statement"].astype(str).str.strip().str.lower()
    catalog["metric_norm"] = catalog["metric"].astype(str).str.strip().str.lower()

    if statement is None:
        matches = catalog[catalog["metric_norm"] == metric]
        if len(matches) == 0:
            raise ValueError(
                f"Không tìm thấy metric '{raw_token}'. Dùng search_financial_metrics để tìm metric hợp lệ."
            )
        unique_pairs = {
            (row.statement_norm, row.metric_norm)
            for row in matches[["statement_norm", "metric_norm"]].itertuples(
                index=False
            )
        }
        if len(unique_pairs) > 1:
            statements = ", ".join(sorted({pair[0] for pair in unique_pairs}))
            raise ValueError(
                f"Metric '{raw_token}' bị trùng ở nhiều statement ({statements}). Hãy ghi rõ dạng statement.metric"
            )
        resolved_statement, resolved_metric = next(iter(unique_pairs))
        return resolved_statement, resolved_metric

    matches = catalog[
        (catalog["statement_norm"] == statement) & (catalog["metric_norm"] == metric)
    ]
    if len(matches) == 0:
        raise ValueError(
            f"Không tìm thấy metric '{raw_token}'. Dùng search_financial_metrics(statement_query='{statement}', metric_query='{metric}') để kiểm tra."
        )
    return statement, metric


def _extract_metric_tokens_from_ast(node: tuple) -> list[str]:
    """Collect identifier tokens from parsed expression AST."""
    kind = node[0]
    if kind == "id":
        return [str(node[1])]
    if kind == "num":
        return []
    if kind == "not":
        return _extract_metric_tokens_from_ast(node[1])
    if kind in {"and", "or"}:
        return _extract_metric_tokens_from_ast(
            node[1]
        ) + _extract_metric_tokens_from_ast(node[2])
    if kind == "cmp":
        return _extract_metric_tokens_from_ast(
            node[2]
        ) + _extract_metric_tokens_from_ast(node[3])
    raise ValueError("AST không hợp lệ")


def _compile_financial_ast_to_sql(
    node: tuple,
    alias_map: dict[tuple[str, str], str],
    metric_catalog: pd.DataFrame,
) -> str:
    """Compile expression AST into SQL WHERE expression."""
    kind = node[0]

    if kind == "num":
        return str(node[1])

    if kind == "id":
        statement, metric = _resolve_metric_token(str(node[1]), metric_catalog)
        alias = alias_map[(statement, metric)]
        return alias

    if kind == "not":
        return (
            f"(NOT {_compile_financial_ast_to_sql(node[1], alias_map, metric_catalog)})"
        )

    if kind in {"and", "or"}:
        left_sql = _compile_financial_ast_to_sql(node[1], alias_map, metric_catalog)
        right_sql = _compile_financial_ast_to_sql(node[2], alias_map, metric_catalog)
        op = "AND" if kind == "and" else "OR"
        return f"({left_sql} {op} {right_sql})"

    if kind == "cmp":
        operator = "=" if node[1] == "==" else node[1]
        left_sql = _compile_financial_ast_to_sql(node[2], alias_map, metric_catalog)
        right_sql = _compile_financial_ast_to_sql(node[3], alias_map, metric_catalog)
        return f"({left_sql} {operator} {right_sql})"

    raise ValueError("AST không hợp lệ")


def _evaluate_financial_ast_on_frame(
    node: tuple,
    alias_map: dict[tuple[str, str], str],
    metric_catalog: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.Series | float:
    """Evaluate parsed financial expression on a pivoted dataframe."""
    kind = node[0]

    if kind == "num":
        return float(node[1])

    if kind == "id":
        statement, metric = _resolve_metric_token(str(node[1]), metric_catalog)
        alias = alias_map[(statement, metric)]
        return frame[alias]

    if kind == "not":
        value = _evaluate_financial_ast_on_frame(
            node[1], alias_map, metric_catalog, frame
        )
        if isinstance(value, pd.Series):
            return ~value.fillna(False).astype(bool)
        return float(not bool(value))

    if kind in {"and", "or"}:
        left_value = _evaluate_financial_ast_on_frame(
            node[1], alias_map, metric_catalog, frame
        )
        right_value = _evaluate_financial_ast_on_frame(
            node[2], alias_map, metric_catalog, frame
        )

        if not isinstance(left_value, pd.Series):
            left_value = pd.Series(left_value, index=frame.index)
        if not isinstance(right_value, pd.Series):
            right_value = pd.Series(right_value, index=frame.index)

        left_bool = left_value.fillna(False).astype(bool)
        right_bool = right_value.fillna(False).astype(bool)
        if kind == "and":
            return left_bool & right_bool
        return left_bool | right_bool

    if kind == "cmp":
        operator = node[1]
        left_value = _evaluate_financial_ast_on_frame(
            node[2], alias_map, metric_catalog, frame
        )
        right_value = _evaluate_financial_ast_on_frame(
            node[3], alias_map, metric_catalog, frame
        )

        if operator == ">":
            return left_value > right_value
        if operator == ">=":
            return left_value >= right_value
        if operator == "<":
            return left_value < right_value
        if operator == "<=":
            return left_value <= right_value
        if operator == "==":
            return left_value == right_value
        if operator == "!=":
            return left_value != right_value
        raise ValueError(f"Toán tử không hỗ trợ: {operator}")

    raise ValueError("AST không hợp lệ")


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
    if not _has_parquet_files(FINANCE_BASE_DIR):
        return pd.DataFrame(
            columns=["symbol", "statement", "metric", "description", "period", "value"]
        )

    conn = duckdb.connect()
    try:
        _create_finance_views(conn)

        sql = "SELECT * FROM finance_long"

        conditions: list[str] = []
        params: dict = {}

        normalized_symbols = _to_list(symbols)
        if normalized_symbols:
            placeholders = ", ".join(
                f"${f'sym_{i}'}" for i in range(len(normalized_symbols))
            )
            conditions.append(f"symbol IN ({placeholders})")
            for i, item in enumerate(normalized_symbols):
                params[f"sym_{i}"] = item.upper()

        normalized_statements = _to_list(statements)
        if normalized_statements:
            placeholders = ", ".join(
                f"${f'statement_{i}'}" for i in range(len(normalized_statements))
            )
            conditions.append(f"statement IN ({placeholders})")
            for i, item in enumerate(normalized_statements):
                params[f"statement_{i}"] = item

        normalized_metrics = _to_list(metrics)
        if normalized_metrics:
            placeholders = ", ".join(
                f"${f'metric_{i}'}" for i in range(len(normalized_metrics))
            )
            conditions.append(f"metric IN ({placeholders})")
            for i, item in enumerate(normalized_metrics):
                params[f"metric_{i}"] = item.lower()

        normalized_periods = _to_list(periods)
        if normalized_periods:
            placeholders = ", ".join(
                f"${f'period_{i}'}" for i in range(len(normalized_periods))
            )
            conditions.append(f"period IN ({placeholders})")
            for i, item in enumerate(normalized_periods):
                params[f"period_{i}"] = item

        if min_value is not None:
            conditions.append("value >= $min_value")
            params["min_value"] = min_value

        if max_value is not None:
            conditions.append("value <= $max_value")
            params["max_value"] = max_value

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        if order_by:
            sql += f" ORDER BY {order_by}"

        if limit is not None:
            sql += f" LIMIT {limit}"

        return conn.execute(sql, params).fetchdf()
    finally:
        conn.close()


def list_financial_metrics(
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """List all available financial metrics with statement name."""
    cached = _load_financial_metric_cache()
    if cached is not None:
        return _materialize_metric_catalog_from_cache(cached, symbols=symbols)

    if not _has_parquet_files(FINANCE_BASE_DIR):
        return pd.DataFrame(columns=["statement", "metric", "description"])

    conn = duckdb.connect()
    try:
        _create_finance_views(conn)
        cache_df = _build_financial_metric_cache(conn)
        FINANCE_METRICS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(FINANCE_METRICS_CACHE_PATH, index=False)
        return _materialize_metric_catalog_from_cache(cache_df, symbols=symbols)
    finally:
        conn.close()


def search_financial_metrics(
    metric_query: str | None = None,
    statement_query: str | None = None,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Search metric catalog by metric and/or statement."""
    catalog = list_financial_metrics(symbols=symbols)
    if catalog.empty:
        return catalog

    metric_query = (metric_query or "").strip().lower()
    statement_query = (statement_query or "").strip().lower()

    mask = pd.Series(True, index=catalog.index)
    if metric_query:
        mask &= catalog["metric"].astype(str).str.lower().str.contains(metric_query)
    if statement_query:
        mask &= (
            catalog["statement"].astype(str).str.lower().str.contains(statement_query)
        )

    return catalog[mask].sort_values(["statement", "metric"]).reset_index(drop=True)


def validate_financial_expression(
    statement: str,
    symbols: list[str] | str | None = None,
) -> pd.DataFrame:
    """Validate a financial expression and return resolved metric references."""
    if not statement or not statement.strip():
        raise ValueError("statement không được rỗng")

    if not _has_parquet_files(FINANCE_BASE_DIR):
        raise ValueError("Không có dữ liệu financial parquet để kiểm tra biểu thức")

    conn = duckdb.connect()
    try:
        _create_finance_views(conn)
        metric_catalog = _get_financial_metric_catalog(conn, symbols=symbols)
        tokens = _tokenize_financial_expression(statement)
        ast = _parse_financial_expression(tokens)

        resolved_pairs: set[tuple[str, str]] = set()
        for token in _extract_metric_tokens_from_ast(ast):
            resolved_pairs.add(_resolve_metric_token(token, metric_catalog))

        if not resolved_pairs:
            raise ValueError("Biểu thức không chứa metric nào")

        return pd.DataFrame(
            sorted(resolved_pairs),
            columns=["statement", "metric"],
        )
    finally:
        conn.close()


def query_financial_by_statement(
    symbols: list[str],
    statement: str,
) -> pd.DataFrame:
    """Filter symbols by a simple financial expression."""
    normalized_symbols = [item.upper() for item in _to_list(symbols)]
    if not normalized_symbols:
        raise ValueError("symbols phải là list[str] và không được rỗng")
    if not statement or not statement.strip():
        raise ValueError("statement không được rỗng")

    if not _has_parquet_files(FINANCE_BASE_DIR):
        return pd.DataFrame(columns=["symbol"])

    conn = duckdb.connect()
    try:
        latest_values_cache = _build_financial_latest_values_cache(conn)
    finally:
        conn.close()

    if latest_values_cache.empty:
        return pd.DataFrame(columns=["symbol"])

    metric_catalog = list_financial_metrics(symbols=normalized_symbols)

    tokens = _tokenize_financial_expression(statement)
    ast = _parse_financial_expression(tokens)
    metric_tokens = _extract_metric_tokens_from_ast(ast)
    if not metric_tokens:
        raise ValueError("Biểu thức không chứa metric nào")

    resolved_metrics: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in metric_tokens:
        resolved = _resolve_metric_token(token, metric_catalog)
        if resolved not in seen:
            seen.add(resolved)
            resolved_metrics.append(resolved)

    alias_map = {
        metric_pair: f"m_{idx}" for idx, metric_pair in enumerate(resolved_metrics)
    }

    needed_pairs_df = pd.DataFrame(resolved_metrics, columns=["statement", "metric"])
    filtered_values = latest_values_cache[
        latest_values_cache["symbol"].astype(str).str.upper().isin(normalized_symbols)
    ]
    filtered_values = filtered_values.merge(
        needed_pairs_df,
        on=["statement", "metric"],
        how="inner",
    )

    if filtered_values.empty:
        selected_columns = ["symbol"] + [f"{st}.{mt}" for st, mt in resolved_metrics]
        return pd.DataFrame(columns=selected_columns)

    alias_lookup = {f"{st}::{mt}": alias for (st, mt), alias in alias_map.items()}
    filtered_values = filtered_values.copy()
    filtered_values["alias"] = (
        filtered_values["statement"].astype(str)
        + "::"
        + filtered_values["metric"].astype(str)
    ).map(alias_lookup)

    pivoted = (
        filtered_values.pivot_table(
            index="symbol",
            columns="alias",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for _, alias in alias_map.items():
        if alias not in pivoted.columns:
            pivoted[alias] = pd.NA

    mask = _evaluate_financial_ast_on_frame(ast, alias_map, metric_catalog, pivoted)
    if not isinstance(mask, pd.Series):
        mask = pd.Series(bool(mask), index=pivoted.index)
    mask = mask.fillna(False).astype(bool)

    selected_aliases = [alias_map[(st, mt)] for st, mt in resolved_metrics]
    selected_df = pivoted.loc[mask, ["symbol", *selected_aliases]].copy()

    rename_map = {alias_map[(st, mt)]: f"{st}.{mt}" for st, mt in resolved_metrics}
    selected_df = selected_df.rename(columns=rename_map)
    return selected_df.sort_values("symbol").reset_index(drop=True)
