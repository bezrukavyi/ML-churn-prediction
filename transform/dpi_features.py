import pandas as pd
import numpy as np
import pickle

from transform.load_data import LoadData

task = 0
task_total = 0


# TRAIN fe
def generate_train_dpi_feature(dataframe):
    cache_key = "cache/dpi_features(short)_train_FIXED.pkl"

    # dpi_new_df = generate_dpi_features(LoadData().df_train_fe, LoadData().df_train_dpi)

    # with open(cache_key, "wb") as f:
    #     pickle.dump(dpi_new_df, f)

    with open(cache_key, "rb") as f:
        dpi_new_df = pickle.load(f)

    return dataframe.merge(dpi_new_df, on="abon_id", how="left").fillna(0)


# TEST fe
def generate_test_dpi_feature(dataframe):
    cache_key = "cache/dpi_features(short)_test_FIXED.pkl"

    # dpi_new_df = generate_dpi_features(LoadData().df_test_fe, LoadData().df_test_dpi)

    # with open(cache_key, "wb") as f:
    #     pickle.dump(dpi_new_df, f)

    with open(cache_key, "rb") as f:
        dpi_new_df = pickle.load(f)

    return dataframe.merge(dpi_new_df, on="abon_id", how="left").fillna(0)


def generate_dpi_features(dataframe, dpi_df):
    global task
    global task_total

    # Load data
    df_data_dpi = dpi_df.copy()

    # Initial build data
    df_data_dpi["target"] = 0
    abon_target = dataframe[["target", "abon_id"]].copy()
    df_data_dpi["target"] = df_data_dpi["abon_id"].map(abon_target.set_index("abon_id")["target"])
    df_data_dpi["target"] = df_data_dpi["target"].astype(int)

    task = 0
    task_total = len(df_data_dpi.abon_id.unique())

    # Suspicous app
    df_data_dpi = assign_application_groupped(df_data_dpi)

    # Generate new features
    df_new = df_data_dpi.groupby("abon_id").apply(calculate_user_metrics_group)

    df_new = df_new.reset_index(drop=True)

    # Format abon_id
    df_new["abon_id"] = df_new["abon_id"].astype(int)

    return df_new


def calculate_share(value, total_value):
    return (value / total_value) * 100 if total_value > 0 else 0


def calculate_susp_app_metrics(group, cache_data):
    suspecias_app_agg_sum = cache_data["suspecias_app_agg_sum"]
    high_churn_app_agg_sum = cache_data["high_churn_app_agg_sum"]

    metrics = {}

    metrics["SUM_of_Count_events_susp_app_sum"] = suspecias_app_agg_sum["SUM_of_Count_events"]
    metrics["MAX_of_day_cnt_susp_app_sum"] = suspecias_app_agg_sum["MAX_of_day_cnt"]
    metrics["SUM_of_Count_events_high_churn_app_sum"] = high_churn_app_agg_sum["SUM_of_Count_events"]

    return metrics


def calculate_feature_common_metrics(group, cache_data):
    metrics = {}

    all_app_agg_sum = cache_data["all_app_agg_sum"]

    metrics["SUM_of_Count_events_sum"] = all_app_agg_sum["SUM_of_Count_events"]
    metrics["MAX_of_day_cnt_sum"] = all_app_agg_sum["MAX_of_day_cnt"]

    return metrics


def calculate_user_metrics_group(group):
    global task
    global task_total

    suspecias_app_df = group[group["is_susp_app"] == True]
    high_churn_app_df = group[group["is_high_churn"] == True]

    cache_data = {
        "all_app_agg_sum": group.sum(),
        "suspecias_app_agg_sum": suspecias_app_df.sum(),
        "high_churn_app_agg_sum": high_churn_app_df.sum(),
    }

    metrics = {
        "abon_id": group.iloc[0].abon_id,
        **calculate_susp_app_metrics(group, cache_data),
        # **calculate_feature_common_metrics(group, cache_data),
    }

    task += 1
    print(f"Progress: {task/task_total*100:.2f}%")

    return pd.Series(metrics)


activ_columns = ["SUM_of_Duration_sec", "SUM_of_Volume_kb", "SUM_of_Count_events", "MAX_of_day_cnt"]


def assign_application_groupped(df):
    df.Application = df.Application.astype(str)

    MAX_of_day_cnt = ["258", "677", "598"]
    SUM_of_Count_events = ["690", "850", "677", "258", "589"]
    SUM_of_Duration_sec = ["850", "258", "716", "677"]
    SUM_of_Volume_kb = ["850", "258", "716"]

    SUSP_APPS = list(set(MAX_of_day_cnt + SUM_of_Count_events + SUM_of_Duration_sec + SUM_of_Volume_kb))

    HIGH_CHURN_APPS = ["1414", "1475", "1477", "1887", "2574", "263", "507", "717", "742", "826", "974"]

    df["is_susp_app"] = df.Application.astype(str).isin(SUSP_APPS)
    df["is_high_churn"] = df.Application.astype(str).isin(HIGH_CHURN_APPS)

    return df
