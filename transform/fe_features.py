import pandas as pd
import numpy as np
import pickle

from transform.load_data import LoadData


# TRAIN fe
def generate_train_fe_feature(dataframe):
    cache_key = "cache/fe_features(short)_train.pkl"

    fe_new_df = generate_fe_features(LoadData().df_train_fe)

    with open(cache_key, "wb") as f:
        pickle.dump(fe_new_df, f)

    with open(cache_key, "rb") as f:
        fe_new_df = pickle.load(f)

    return dataframe.merge(fe_new_df, on="abon_id", how="left").fillna(0)


# TEST fe
def generate_test_fe_feature(dataframe):
    cache_key = "cache/fe_features(short)_test.pkl"

    fe_new_df = generate_fe_features(LoadData().df_test_fe)

    with open(cache_key, "wb") as f:
        pickle.dump(fe_new_df, f)

    with open(cache_key, "rb") as f:
        fe_new_df = pickle.load(f)

    return dataframe.merge(fe_new_df, on="abon_id", how="left").fillna(0)


def generate_fe_features(dataframe):
    df = dataframe.copy()
    columns_groups = build_columns_groups(df)

    new_features = {}

    for column in columns_groups.keys():
        statistic_columns = columns_groups[column]
        print(f"Processing {column}...")

        new_features[f"{column}_diff"] = calculate_diff(df, statistic_columns)
        new_features[f"{column}_total"] = calculate_total_sum(df, statistic_columns)

    new_features_df = pd.DataFrame()
    new_features_df["abon_id"] = df["abon_id"]

    new_features_df = pd.concat([new_features_df, pd.DataFrame(new_features)], axis=1)

    return new_features_df


def build_columns_groups(dataframe):
    columns_groups = {}

    for column in dataframe.columns[dataframe.columns.str.contains(pat="_mnt|_wk")].to_list():
        column_name = "_".join(column.split("_")[:-1])

        if column_name not in columns_groups:
            columns_groups[column_name] = []

        columns_groups[column_name].append(column)

    return columns_groups


def calculate_diff_for_group(x):
    if x[-1] == 0:
        return 0

    value = round(((x[0] - x[-1]) / x[-1]) * 100, 2)

    if value == 0:
        return 0

    return value


def calculate_diff(dataframe, columns):
    y = dataframe[columns].ffill(axis=1).values.T

    return np.array(np.apply_along_axis(calculate_diff_for_group, 0, y))


def calculate_total_sum(dataframe, columns):
    y = dataframe[columns].ffill(axis=1).values.T

    return np.array(np.apply_along_axis(lambda x: x.sum(), 0, y))
