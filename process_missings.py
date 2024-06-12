import numpy as np
import pandas as pd


def process_missings(dataframe):
    dataframe = dataframe.copy()

    # Remove data with unsupported type (totally blank series)

    UNSUPPORTED_COLUMNS = [
        "device_has_gprs",
        "bs_of_succ_m1",
        "bs_drop_call_rate",
        "bs_succ_rate",
        "bs_drop_rate",
        "bs_of_recall_m1",
        "bs_of_attemps_all_m1",
        "bs_of_unsucc_low_balance_m1",
        "bs_recall_rate",
        "bs_of_unsucc_attemp_equip_m1",
        "bs_of_succ_but_drop_m1",
    ]

    dataframe = dataframe.drop(columns=UNSUPPORTED_COLUMNS)

    # Drop frauds
    dataframe = dataframe[dataframe.MV_FRAUD_BLOCK != 1]
    dataframe = dataframe.drop(columns=["MV_FRAUD_BLOCK"])

    # NEW COLUMN: device_brand
    # MISSING VALUE (MEDIANA): device_price

    DEVICE_COLUMNS = [
        "device_brand_samsung",
        "device_brand_nokia",
        "device_brand_lenovo",
        "device_brand_apple",
        "device_brand_huawei",
        "device_brand_lg",
        "device_brand_xiaomi",
        "device_brand_meizu",
        "device_brand_prestigio",
        "device_brand_sony",
        "device_brand_other",
    ]

    dataframe[DEVICE_COLUMNS] = dataframe[DEVICE_COLUMNS].fillna(0)
    dataframe["device_brand"] = dataframe[DEVICE_COLUMNS].idxmax(axis=1)
    dataframe["device_brand"] = dataframe["device_brand"].str.replace("device_brand_", "")

    # Group by 'device_brand' and calculate the median for the specified columns
    grouped_medians = dataframe.groupby("device_brand").median(numeric_only=True)

    # Fill missing values with the median for each group
    columns_to_fill = ["device_price", "imei_mean_price", "imei_max_price", "device_height_mm"]

    for column in columns_to_fill:
        blank_df = dataframe[dataframe[column].isnull()]
        dataframe.loc[blank_df.index, column] = blank_df["device_brand"].map(grouped_medians[column])

    MISSING_INDEX = 0

    columns_with_int_dtype = dataframe.select_dtypes(include=["int64", "int32", "float64"]).columns
    highly_missing_columns = dataframe.isnull().mean() * 100
    highly_missing_columns = highly_missing_columns[highly_missing_columns > MISSING_INDEX].index

    highly_missing_int_columns = set(highly_missing_columns) & set(columns_with_int_dtype)

    # data = pd.DataFrame()

    # for col in highly_missing_int_columns:
    #     series_to_concat = describe_series(dataframe[col])
    #     data = pd.concat([data, series_to_concat])

    # print(data)

    dataframe[list(highly_missing_int_columns)] = dataframe[list(highly_missing_int_columns)].fillna(0)

    dataframe = dataframe.drop(columns=["device_brand"])

    return dataframe
