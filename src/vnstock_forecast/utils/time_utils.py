from datetime import datetime

import pandas as pd


def time_to_timestamp(
    time_input: str | datetime | pd.Timestamp, unit: str = "s"
) -> int:
    """
    Convert a time input to a Unix timestamp.
    Args:
        time_input (str | datetime | pd.Timestamp): The input time to convert.
        unit (str): The unit of the timestamp ('s' for seconds, 'ms' for milliseconds).
    Returns:
        int: The Unix timestamp.
    Raises:
        ValueError: If the input type is unsupported or unit is invalid.
    """
    try:
        # Pandas to_datetime can handle str, datetime, and pd.Timestamp
        ts = pd.to_datetime(time_input)
        timestamp_float = ts.timestamp()
        if unit == "ms":
            return int(timestamp_float * 1000)
        return int(timestamp_float)
    except Exception as e:
        raise ValueError(f"Unsupported time input type or invalid unit: {e}")


def timestamp_to_str(
    ts: int | float, fmt: str = "%Y-%m-%d %H:%M:%S", unit: str = "s"
) -> str:
    """
    Convert Unix timestamp to formatted string.
    Args:
        ts (int | float): The Unix timestamp.
        fmt (str): The format string for output (e.g., '%Y-%m-%d %H:%M:%S').
        unit (str): The unit of the timestamp ('s' for seconds, 'ms' for milliseconds).
    Returns:
        str: The formatted time string.
    """
    if unit == "ms":
        ts = ts / 1000
    dt_obj = pd.to_datetime(ts, unit="s")
    return dt_obj.strftime(fmt)


def get_current_timestamp(unit: str = "s") -> int:
    """
    Get the current Unix timestamp.
    Args:
        unit (str): The unit of the timestamp ('s' for seconds, 'ms' for milliseconds).
    Returns:
        int: The current Unix timestamp.
    """
    now = pd.to_datetime("now")
    timestamp_float = now.timestamp()
    if unit == "ms":
        return int(timestamp_float * 1000)
    return int(timestamp_float)


def get_current_date_timestamp(unit: str = "s") -> int:
    """
    Get the current date's Unix timestamp at midnight.
    Args:
        unit (str): The unit of the timestamp ('s' for seconds, 'ms' for milliseconds).
    Returns:
        int: The Unix timestamp for the current date at midnight.
    """
    now = pd.to_datetime("now").normalize()  # Normalize to midnight
    timestamp_float = now.timestamp()
    if unit == "ms":
        return int(timestamp_float * 1000)
    return int(timestamp_float)


def add_days_to_timestamp(ts: int | float, days: int, unit: str = "s") -> int:
    """
    Add a number of days to a given Unix timestamp.
    Args:
        ts (int | float): The original Unix timestamp.
        days (int): The number of days to add (can be negative).
        unit (str): The unit of the timestamp ('s' for seconds, 'ms' for milliseconds).
    Returns:
        int: The new Unix timestamp after adding the days.
    """
    if unit == "ms":
        ts = ts / 1000
    dt_obj = pd.to_datetime(ts, unit="s") + pd.Timedelta(days=days)
    new_timestamp_float = dt_obj.timestamp()
    if unit == "ms":
        return int(new_timestamp_float * 1000)
    return int(new_timestamp_float)


if __name__ == "__main__":
    # Example usages
    print(time_to_timestamp("2024-01-01 12:00:00"))
    print(timestamp_to_str(time_to_timestamp("2024-01-01 12:00:00")))
    print(get_current_timestamp())
    print(get_current_date_timestamp())
    print(timestamp_to_str(get_current_date_timestamp(), fmt="%Y-%m-%d %H:%M:%S"))
    print(add_days_to_timestamp(time_to_timestamp("2024-01-01 12:00:00"), 10))
    print(
        timestamp_to_str(
            add_days_to_timestamp(time_to_timestamp("2024-01-01 12:00:00"), -1)
        )
    )
