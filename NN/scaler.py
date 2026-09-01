import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


## train mean: [-0. -0.  0. -0.] The scaler learns on the train data, quindi si annulla con i dati del training lo scaler
## train std : [1. 1. 1. 1.]
## val mean  : [-0.12   0.168 -0.234 -0.008] ## Then he applys those number learnt from the train to the other set
## test mean : [ 0.007 -0.072 -0.016 -0.011] ## the fact that these are not 0 it means we didnt do any data-leakge


def load_scaler(ticker = "AAPL", data_dir = "data"):

    d = Path(data_dir)
    train_data = pd.read_csv(d / f"{ticker}_train.csv", index_col=0, parse_dates=True)
    val_data   = pd.read_csv(d / f"{ticker}_val.csv",   index_col=0, parse_dates=True)
    test_data  = pd.read_csv(d / f"{ticker}_test.csv",  index_col=0, parse_dates=True)

    print(train_data.isna().sum())
    print(train_data.head(10))

    #Let's separate features from targets

    feats_cols = [ c for c in train_data.columns if c != "target"]


    X_train, Y_train = train_data[feats_cols], train_data["target"]
    X_val, Y_val = val_data[feats_cols], val_data["target"]
    X_test, Y_test = test_data[feats_cols], test_data["target"]

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train) ## The scaler  learns how to standardize on the train only ofcourse
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # freeze the fitted scaler — needed to transform any future data identically
    joblib.dump(scaler, d / f"{ticker}_scaler.pkl")

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, scaler


if __name__ == "__main__":
    X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, scaler = load_scaler()

    # confirm the fit landed: train should be ~mean 0, ~std 1; val/test CLOSE but not exact
    print("train mean:", X_train_s.mean(axis=0).round(3))
    print("train std :", X_train_s.std(axis=0).round(3))
    print("val mean  :", X_val_s.mean(axis=0).round(3))
    print("test mean :", X_test_s.mean(axis=0).round(3))