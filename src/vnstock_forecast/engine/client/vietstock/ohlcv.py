import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from vnstock_forecast.engine.utils import time_utils


class OHLCV:
    """
    Client for interacting with Vietstock API.
    """

    BASE_URL: str = "https://api.vietstock.vn/tvnew/history"
    # Headers for the HTTP requests
    HEADERS: dict[str, str] = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Origin": "https://stockchart.vietstock.vn",
        "Referer": "https://stockchart.vietstock.vn/",
    }

    def __init__(self, timeout: int = 5, retries: int = 3, backoff_factor: float = 0.3):
        self.timeout = timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch(
        self, from_ts: int, to_ts: int, ticker: str, resolution: str, countback: int = 2
    ) -> pd.DataFrame | None:
        """
        Get OHLCV historical data from Vietstock API.
        Args:
            from_ts (int): Unix timestamp for statting point.
            to_ts (int): Unix timestamp for ending point.
            ticker (str): Stock ticker symbol.
            resolution (str): Data resolution (e.g., "D" for daily).
            Supported resolutions: "1", "5", "15", "30", "45, "60",
                                    "120", "180", "240", "D", "W", "M"
            countback (int): Old param to limit number of data points (no need to use).
        Returns:
            pd.DataFrame | None: DataFrame with OHLCV data None if request fails.
        Raise:
            requests.RequestException: If the API request fails.
        """
        print(
            f"Fetching OHLCV for {ticker} from {from_ts} to {to_ts} at resolution {resolution}"
        )
        params = {
            "symbol": ticker,
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
            "countback": countback,
        }

        try:
            response = self.session.get(
                self.BASE_URL, headers=self.HEADERS, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            print(type(response))
            print(response)
            # Add params to response
            data = response.json()
            data["params"] = params
            return self._to_dataframe(data)
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None

    def fetch_realtime(
        self, ticker: str, from_ts: int, resolution: str, countback: int = 2
    ) -> pd.DataFrame | None:
        """
        Get OHLCV real-time data from Vietstock API.
        Args:
            ticker (str): Stock ticker symbol.
            from_ts (int): Unix timestamp for starting point.
            resolution (str): Data resolution (e.g., "D" for daily).
            countback (int): Old param to limit number of data points (no need to use).
        Returns:
            pd.DataFrame | None: DataFrame with OHLCV data or None if request fails.
        Raise:
            requests.RequestException: If the API request fails.
        """
        to_ts = time_utils.get_current_timestamp()
        return self.fetch(from_ts, to_ts, ticker, resolution, countback)

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        """
        Convert API response to a pandas DataFrame.
        Args:
            data (dict): API response containing OHLCV data.
        Returns:
            pd.DataFrame: DataFrame with columns
            ['Timestamp', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
        """
        if not data or "c" not in data:
            return pd.DataFrame()  # Return empty DataFrame if data is invalid

        df = pd.DataFrame(
            {
                "Timestamp": data["t"],
                "Symbol": data["params"]["symbol"],
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"],
            }
        )
        return df


if __name__ == "__main__":
    client = OHLCV()
    data = client.fetch(
        from_ts=time_utils.add_days_to_timestamp(
            time_utils.get_current_date_timestamp(), -10
        ),  # Example timestamp
        to_ts=time_utils.get_current_date_timestamp(),  # Example timestamp
        ticker="VND",
        resolution="D",
    )
    print(data)
    data2 = client.fetch_realtime(
        ticker="VHM",
        from_ts=time_utils.add_days_to_timestamp(
            ts=time_utils.get_current_date_timestamp(), days=-5
        ),
        resolution="5",
    )
    print(data2)
