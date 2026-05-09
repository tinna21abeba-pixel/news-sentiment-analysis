"""
data_loader.py
==============
Nova Financial Solutions
Download historical price data via yfinance and save to data/raw/.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Default tickers used in the assignment
DEFAULT_TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_stock_data(
    tickers: list = DEFAULT_TICKERS,
    start: str = "2019-01-01",
    end: str = None,
    save: bool = True,
) -> dict:
    """
    Download daily OHLCV data for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : list
    start   : str  (YYYY-MM-DD)
    end     : str  (YYYY-MM-DD), defaults to today
    save    : bool – save each ticker as CSV in data/raw/

    Returns
    -------
    dict of {ticker: pd.DataFrame}
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    for ticker in tickers:
        print(f"Downloading {ticker} …", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                print("No data returned.")
                continue
            df.reset_index(inplace=True)
            if save:
                path = os.path.join(DATA_DIR, f"{ticker}_price_data.csv")
                df.to_csv(path, index=False)
                print(f"saved → {path}  ({len(df):,} rows)")
            else:
                print(f"loaded ({len(df):,} rows)")
            results[ticker] = df
        except Exception as e:
            print(f"ERROR – {e}")

    return results


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load a previously-downloaded CSV from data/raw/."""
    path = os.path.join(DATA_DIR, f"{ticker}_price_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file found for {ticker} at {path}. "
            f"Run download_stock_data() first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_news_data(filepath: str) -> pd.DataFrame:
    """
    Load the FNSPID news CSV.

    Parameters
    ----------
    filepath : str  Path to the raw news CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"News file not found: {filepath}")
    df = pd.read_csv(filepath)
    expected_cols = {"headline", "url", "publisher", "date", "stock"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"News CSV is missing expected columns: {missing}")
    print(f"Loaded news dataset: {len(df):,} rows, {df['stock'].nunique()} unique tickers.")
    return df


if __name__ == "__main__":
    print("=== Downloading stock price data for default tickers ===")
    download_stock_data()
