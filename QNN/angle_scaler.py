from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pathlib import Path
import joblib as jb


def load_angle_scaler(ticker = "AAPL", data_dir ="data"):

    ## Loading the sets

    d = Path(data_dir)
    train_data = pd.read_csv(d / f"{ticker}_train.csv", index_col=0, parse_dates=True)
    val_data   = pd.read_csv(d / f"{ticker}_val.csv",   index_col=0, parse_dates=True)
    test_data  = pd.read_csv(d / f"{ticker}_test.csv",  index_col=0, parse_dates=True)

    feats_cols = [ c for c in train_data.columns if c != "target"]


    scaler = MinMaxScaler(feature_range=(0, np.pi)) ## Creating the angle scaler that maps value from 0 to PI(3.14)

    X_train_s = scaler.fit_transform(train_data[feats_cols]) ## Fitting the transformer based on the training
    X_val_s = scaler.transform(val_data[feats_cols])
    X_test_s = scaler.transform(test_data[feats_cols])

    # Clipping all the values that exceed PI

    X_train_s  = np.clip(X_train_s, 0, np.pi)
    X_val_s = np.clip(X_val_s, 0 , np.pi)
    X_test_s = np.clip(X_test_s, 0 , np.pi)


    # Take the targets

    Y_train = train_data["target"]
    Y_val = val_data["target"]
    Y_test = test_data["target"]


    # Printing to see how many out-of-range cases there are
    
    print("train range:", X_train_s.min(axis=0).round(3), X_train_s.max(axis=0).round(3))
    print("val at boundary:", ((X_val_s == 0) | (X_val_s == np.pi)).sum())
    print("test at boundary:", ((X_test_s == 0) | (X_test_s == np.pi)).sum())


    # Save the scaler usign joblib

    jb.dump(scaler, d / f"{ticker}_angle_scaler.pkl")

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, scaler




if __name__ == "__main__":
    load_angle_scaler()