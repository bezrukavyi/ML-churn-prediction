import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import numpy as np
import scipy


def calculate_slopes(dataframe):
    dataframe = dataframe.copy()
    print("Slope_columns: ", slope_columns)

    for column in slope_columns:
        # Add new slope column
        dataframe[f"{column}_slope"] = _calculate_column_slopes(dataframe, column)

    return dataframe


def _calculate_column_slopes(df, column):
    print(f"Calculating slopes for {column}...")
    t = np.array([1, 2, 3])
    columns = [f"{column}_mnt3", f"{column}_mnt1", f"{column}_wk1"]
    y = df[columns].ffill(axis=1).values.T
    slopes = np.apply_along_axis(lambda x: scipy.stats.linregress(t, x)[0], 0, y)
    return slopes
