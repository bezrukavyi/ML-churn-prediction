import pandas as pd
import numpy as np
import pickle

from transform.load_data import LoadData

task = 0
task_total = 0


# TRAIN fe
def generate_train_dpi_feature(dataframe):
    cache_key = "cache/dpi_features(short)_train.pkl"

    dpi_new_df = generate_dpi_features(LoadData().df_train_fe, LoadData().df_train_dpi)

    with open(cache_key, "wb") as f:
        pickle.dump(dpi_new_df, f)

    with open(cache_key, "rb") as f:
        dpi_new_df = pickle.load(f)

    return dataframe.merge(dpi_new_df, on="abon_id", how="left").fillna(0)


# TEST fe
def generate_test_dpi_feature(dataframe):
    cache_key = "cache/dpi_features(short)_test.pkl"

    dpi_new_df = generate_dpi_features(LoadData().df_test_fe, LoadData().df_test_dpi)

    with open(cache_key, "wb") as f:
        pickle.dump(dpi_new_df, f)

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
    if cache_data["suspecias_app_df"].empty:
        return {
            "SUM_of_Count_events_susp_app_sum": 0,
            "MAX_of_day_cnt_susp_app_sum": 0,
            "SUM_of_Volume_kb_susp_app_share": 0,
            "MAX_of_day_cnt_susp_app_share": 0,
            "SUM_of_Duration_sec_susp_app_share": 0,
            "SUM_of_Count_events_sum": 0,
            "MAX_of_day_cnt_sum": 0,
        }

    all_app_agg_sum = cache_data["all_app_agg_sum"]
    suspecias_app_agg_sum = cache_data["suspecias_app_agg_sum"]

    metrics = {}

    metrics["SUM_of_Count_events_susp_app_sum"] = suspecias_app_agg_sum["SUM_of_Count_events"]
    metrics["MAX_of_day_cnt_susp_app_sum"] = suspecias_app_agg_sum["MAX_of_day_cnt"]

    metrics["SUM_of_Volume_kb_susp_app_share"] = calculate_share(
        suspecias_app_agg_sum["SUM_of_Volume_kb"], all_app_agg_sum["SUM_of_Volume_kb"]
    )
    metrics["MAX_of_day_cnt_susp_app_share"] = calculate_share(
        suspecias_app_agg_sum["MAX_of_day_cnt"], all_app_agg_sum["MAX_of_day_cnt"]
    )
    metrics["SUM_of_Duration_sec_susp_app_share"] = calculate_share(
        suspecias_app_agg_sum["SUM_of_Duration_sec"], all_app_agg_sum["SUM_of_Duration_sec"]
    )

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

    suspecias_app_df = group[group["Application_groupped"] == "SUSPICIOUS_APP"]

    cache_data = {
        "all_app_agg_sum": group.sum(),
        "suspecias_app_df": suspecias_app_df,
        "suspecias_app_agg_sum": suspecias_app_df.sum(),
    }

    metrics = {
        "abon_id": group.iloc[0].abon_id,
        **calculate_susp_app_metrics(group, cache_data),
        **calculate_feature_common_metrics(group, cache_data),
    }

    task += 1
    print(f"Progress: {task/task_total*100:.2f}%")

    return pd.Series(metrics)


top_apps = [
    897,
    258,
    2563,
    1414,
    1929,
    2574,
    541,
    677,
    1446,
    1193,
    1453,
    815,
    690,
    1479,
    716,
    850,
    598,
    1500,
    877,
    882,
    1527,
    381,
    254,
]

top_apps_by_churn = {
    "SUM_of_Duration_sec": [
        "897",
        "541",
        "1453",
        "258",
        "28",
        "254",
        "1414",
        "775",
        "1821",
        "1446",
        "2574",
        "677",
        "784",
        "1500",
        "850",
        "808",
        "1193",
        "617",
        "1315",
        "1367",
        "992",
        "677",
        "897",
        "1453",
        "814",
        "826",
        "1414",
        "814",
        "1475",
        "716",
        "1477",
        "1479",
        "717",
        "974",
        "2563",
        "850",
        "381",
        "1367",
        "240",
        "1479",
        "992",
        "1505",
        "612",
        "2532",
        "258",
        "1478",
        "1530",
        "1530",
        "381",
        "254",
    ],
    "SUM_of_Volume_kb": [
        "897",
        "541",
        "1453",
        "514",
        "254",
        "1414",
        "775",
        "263",
        "1446",
        "2574",
        "1295",
        "1297",
        "1500",
        "850",
        "1193",
        "617",
        "413",
        "1440",
        "1270",
        "1315",
        "1446",
        "897",
        "1453",
        "2058",
        "315",
        "1414",
        "814",
        "1929",
        "716",
        "1349",
        "1477",
        "716",
        "1169",
        "1349",
        "2563",
        "2574",
        "381",
        "1479",
        "1315",
        "612",
        "2532",
        "740",
        "258",
        "1478",
        "1527",
        "1389",
        "1270",
        "1527",
        "1530",
        "381",
    ],
    "SUM_of_Count_events": [
        "1333",
        "1453",
        "815",
        "258",
        "254",
        "1414",
        "897",
        "263",
        "1821",
        "2574",
        "677",
        "850",
        "808",
        "1004",
        "1367",
        "677",
        "897",
        "546",
        "1453",
        "814",
        "815",
        "690",
        "826",
        "1414",
        "814",
        "1475",
        "716",
        "1477",
        "1479",
        "716",
        "717",
        "974",
        "2563",
        "850",
        "381",
        "598",
        "1367",
        "240",
        "1479",
        "1887",
        "2046",
        "826",
        "742",
        "258",
        "690",
        "844",
        "507",
        "598",
        "381",
        "254",
    ],
    "MAX_of_day_cnt": [
        "541",
        "815",
        "258",
        "254",
        "1373",
        "775",
        "1375",
        "1038",
        "677",
        "850",
        "808",
        "546",
        "1315",
        "1367",
        "677",
        "604",
        "897",
        "546",
        "815",
        "690",
        "2046",
        "716",
        "711",
        "246",
        "716",
        "717",
        "850",
        "381",
        "598",
        "1367",
        "240",
        "604",
        "1373",
        "992",
        "2046",
        "612",
        "1893",
        "711",
        "258",
        "690",
        "1020",
        "877",
        "2032",
        "882",
        "844",
        "877",
        "882",
        "598",
        "381",
        "254",
    ],
}

activ_columns = ["SUM_of_Duration_sec", "SUM_of_Volume_kb", "SUM_of_Count_events", "MAX_of_day_cnt"]


def assign_application_groupped(df):
    df.Application = df.Application.astype(str)

    MAX_of_day_cnt = ["258", "677", "598"]
    SUM_of_Count_events = ["690", "850", "677", "258", "589"]
    SUM_of_Duration_sec = ["850", "258", "716", "677"]
    SUM_of_Volume_kb = ["850", "258", "716"]

    for feature in activ_columns:
        top_apps_by_churn[feature] += ["SUSPICIOUS_APP"]

    all_ids = list(set(MAX_of_day_cnt + SUM_of_Count_events + SUM_of_Duration_sec + SUM_of_Volume_kb))

    df["Application_groupped"] = df.Application.apply(lambda x: "SUSPICIOUS_APP" if str(x) in all_ids else str(x))

    return df
