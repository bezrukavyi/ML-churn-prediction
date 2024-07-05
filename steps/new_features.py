import pickle
import pdb


def merge_train_dpi_features(dataframe):
    with open("cache/project_step_3_train.pkl", "rb") as f:
        dpi_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(dpi_df, on="abon_id", how="left").fillna(0)


def merge_test_dpi_features(dataframe):
    with open("cache/project_step_3_test.pkl", "rb") as f:
        dpi_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(dpi_df, on="abon_id", how="left").fillna(0)


def merge_train_bnum_features(dataframe):
    with open("cache/project_step_4(2)_train.pkl", "rb") as f:
        bnum_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_test_bnum_features(dataframe):
    with open("cache/project_step_4(2)_test.pkl", "rb") as f:
        bnum_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_train_fe_features(dataframe):
    with open("cache/project_step_5_train.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_test_fe_features(dataframe):
    with open("cache/project_step_5_test.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_train_fe_total_features(dataframe):
    with open("cache/project_step_5_total_sum_train.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_test_fe_total_features(dataframe):
    with open("cache/project_step_5_total_sum_test.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_train_fe_slope_features(dataframe):
    with open("cache/project_step_5_total_slopes_train.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_test_fe_slope_features(dataframe):
    with open("cache/project_step_5_total_slopes_test.pkl", "rb") as f:
        bnum_df = pickle.load(f)

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)
