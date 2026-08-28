import pandas as pd 
import numpy as np
from feature_creation import create_features, create_target
from pathlib import Path

def build_dataset(
    raw: pd.DataFrame,
    train_end: str,
    val_end: str,
    out_dir = "data",
    ticker = 'AAPL'
):

    feats = create_features(raw)
    target = create_target(raw)

    data = feats.join(target).dropna() # data now is a db with all the features and the targets

    ## Crnological split

    train_data = data.loc[:train_end]
    val_data = data.loc[train_end:val_end].iloc[1:] ## The iloc means drop the first row the one in common and keep the rest
    test_data = data.loc[val_end:].iloc[1:]

    feats_cols = []

    for c in data.columns:
        if c != "target":
            feats_cols.append(c)

    X_train, Y_train = train_data[feats_cols], train_data["target"]
    X_val, Y_val = val_data[feats_cols], val_data["target"]
    X_test, Y_test = test_data[feats_cols], test_data["target"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # each block keeps its features AND target together — X and y can't drift apart
    train_data.to_csv(out / f"{ticker}_train.csv")
    val_data.to_csv(out / f"{ticker}_val.csv")
    test_data.to_csv(out / f"{ticker}_test.csv")



    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) ## It expects 3 values, but we have 6 so we squeez them in a tuple


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_AAPL_1wk.csv", index_col=0, parse_dates=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_dataset(
        raw,
        train_end="2020-12-31", #8years of training
        val_end="2022-12-31", # 2 years of evaluation, and 4 years of test.
    )


    # shapes — confirm the three blocks are sized as you expect
    for name, X, y in [("train", X_train, y_train),
                       ("val", X_val, y_val),
                       ("test", X_test, y_test)]:
        # class balance PER SPLIT — the check from last message.
        balance = y.value_counts(normalize=True).round(3).to_dict()
        print(f"{name:5s} rows={len(X):4d}  balance={balance}")