import numpy as np
import pandas as pd
import pickle
import process_missings
import importlib

process_missings = importlib.reload(process_missings).process_missings


def load_data():
    with open("./data/churn_model_pd2.pcl", "rb") as f:
        bdl_data = pickle.load(f)

    _, train_data, test_data = bdl_data

    _, df_train_fe, df_train_bnum, df_train_dpi = train_data
    _, df_test_fe, df_test_bnum, df_test_dpi = test_data

    df_train_fe_processed = process_missings(df_train_fe)
    df_test_fe_processed = process_missings(df_test_fe)

    X_train = df_train_fe_processed.drop("target", axis=1)
    y_train = df_train_fe_processed.target

    X_test = df_test_fe_processed.drop("target", axis=1)
    y_test = df_test_fe_processed.target

    return X_train, X_test, y_train, y_test
