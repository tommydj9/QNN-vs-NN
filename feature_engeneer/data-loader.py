import yfinance as yf
import pandas as pd
from pathlib import Path


def load_data(ticker, start_date, end_date, interval='1wk', save_dir='data'):
    """
    Load historical stock data for a given ticker symbol from Yahoo Finance.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    interval (str): Bar frequency; '1wk' lets the provider do the weekly aggregation.
    save_dir (str): Folder to write the frozen CSV into.

    Returns:
    pandas.DataFrame: A DataFrame containing the raw OHLCV data.
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,   # stated on purpose — adjusts OHLC for splits/dividends
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex): ## This drops the multindex dataframe to just a one
        data.columns = data.columns.get_level_values(0)

    # Keep the full OHLCV frame — no column selection here.
    # Look before cleaning: print shape, span, and where any NaNs sit.
    print(f"[{ticker}] rows: {len(data)}")
    print(f"[{ticker}] date range: {data.index.min()} -> {data.index.max()}")
    print(f"[{ticker}] NaNs per column:\n{data.isna().sum()}")

    # Freeze to disk — everything downstream reads this file, not the network.
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / f"raw_{ticker}_{interval}.csv"
    data.to_csv(out_path)
    print(f"[{ticker}] saved -> {out_path}")
    print(data.dtypes)
    print(type(data['Close']))

    return data

load_data("AAPL", "2012-01-01", "2026-01-01", interval='1wk', save_dir='data')