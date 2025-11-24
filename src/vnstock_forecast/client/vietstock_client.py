import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class VietstockClient:
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

    def fetch_history(
        self, from_ts: int, to_ts: int, ticker: str, resolution: str, countback: int = 2
    ) -> dict | None:
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
            dict | None: JSON response from the API or None if request fails.
        Raise:
            requests.RequestException: If the API request fails.
        """
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
            return data
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None


if __name__ == "__main__":
    client = VietstockClient()
    data = client.fetch_history(
        from_ts=1622505600,  # Example timestamp
        to_ts=1625097600,  # Example timestamp
        ticker="VND",
        resolution="D",
    )
    print(data)
