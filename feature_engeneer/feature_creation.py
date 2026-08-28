## I need to create all the features for my models
## Let's remember that the more features you add to your model, 
# the more qubits you will need to use in your quantum circuit, adding width to it,
# and the more difficult it will be to train your quantum neural network.
# For everyh feture sit doubles the number of qubits. 2^N where N is the number of features.

import pandas as pd
import numpy as np

def create_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the given stock data.

    Parameters:
    data (pandas.DataFrame): A DataFrame containing the raw OHLCV data.

    Returns:
    pandas.DataFrame: A DataFrame containing the features.
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    df = pd.DataFrame(index=raw.index)

    close = raw['Close']
    volume = raw['Volume']

    # log-n-returns
    df['ret-1w'] = np.log(close / close.shift(1))

    #volatility

    df['vol-1w'] = df['ret-1w'].rolling(window=8).std()  # 8 trading weeks in a week

    # Momentum of last 8 weeks

    df['mom-8w'] = np.log(close / close.shift(8))

    # Vol Z_score

    std_dev = volume.rolling(window=8).std()
    volume_mean = volume.rolling(window=8).mean()
    df['vol-zscore'] = (volume - volume_mean) / std_dev

    return df

def create_target(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable for the given stock data.

    Parameters:
    data (pandas.DataFrame): A DataFrame containing the raw OHLCV data.

    Returns:
    pandas.DataFrame: A DataFrame containing the target variable.
    """
    df = pd.DataFrame(index=raw.index)
    close = raw['Close']

    # Target: 1 if price goes up next week, 0 otherwise
    df['target'] = (close.shift(-1) > close).astype(int)

    return df



if __name__ == "__main__":
    raw = pd.read_csv('data/raw_AAPL_1wk.csv', index_col=0, parse_dates=True)
    feats = create_features(raw)
    target = create_target(raw)

    print(target['target'].value_counts(normalize=True))

