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
    with open("cache/project_step_4_train.pkl", "rb") as f:
        bnum_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)


def merge_test_bnum_features(dataframe):
    with open("cache/project_step_4_test.pkl", "rb") as f:
        bnum_df = pickle.load(f).drop(columns=["target"])

    return dataframe.merge(bnum_df, on="abon_id", how="left").fillna(0)
