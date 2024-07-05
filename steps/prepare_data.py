import pickle
import sklearn
from utils.pipeline import Pipeline, PipelineStep

from steps.set_missings import set_missings
from utils.helpers import reduce_mem_usage
from steps.load_data import load_train_data, load_test_data
from steps.feature_selection import feature_selection
from steps.drop_high_correlation import drop_high_correlation

from steps.new_features import (
    merge_train_dpi_features,
    merge_test_dpi_features,
    merge_train_bnum_features,
    merge_test_bnum_features,
    merge_train_fe_features,
    merge_test_fe_features,
    merge_train_fe_total_features,
    merge_test_fe_total_features,
    merge_train_fe_slope_features,
    merge_test_fe_slope_features,
)

load_train_data_step = PipelineStep(load_train_data)
load_test_data_step = PipelineStep(load_test_data)
set_missings_step = PipelineStep(set_missings)
reduce_mem_usage_step = PipelineStep(reduce_mem_usage)
feature_selection_step = PipelineStep(feature_selection)
drop_high_correlation_step = PipelineStep(drop_high_correlation)

merge_train_dpi_features_step = PipelineStep(merge_train_dpi_features)
merge_train_bnum_features_step = PipelineStep(merge_train_bnum_features)
merge_test_dpi_features_step = PipelineStep(merge_test_dpi_features)
merge_test_bnum_features_step = PipelineStep(merge_test_bnum_features)
merge_train_fe_features_step = PipelineStep(merge_train_fe_features)
merge_test_fe_features_step = PipelineStep(merge_test_fe_features)
merge_train_fe_total_features_step = PipelineStep(merge_train_fe_total_features)
merge_test_fe_total_features_step = PipelineStep(merge_test_fe_total_features)
merge_train_fe_slope_features_step = PipelineStep(merge_train_fe_slope_features)
merge_test_fe_slope_features_step = PipelineStep(merge_test_fe_slope_features)

remove_abon_id_step = PipelineStep(lambda df: df.drop("abon_id", axis=1))

transform_train_pipeline = Pipeline(
    "TRANSFORM_TRAIN",
    [
        set_missings_step,
        reduce_mem_usage_step,
        # drop_high_correlation_step,
        merge_train_dpi_features_step,
        # drop_high_correlation_step,
        merge_train_fe_features_step,
        # drop_high_correlation_step,
        merge_train_fe_total_features_step,
        # drop_high_correlation_step,
        merge_train_bnum_features_step,
        # drop_high_correlation_step,
        remove_abon_id_step,
        feature_selection_step,
    ],
)

transform_test_pipeline = Pipeline(
    "TRANSFORM_TEST",
    [
        set_missings_step,
        reduce_mem_usage_step,
        merge_test_dpi_features_step,
        merge_test_bnum_features_step,
        merge_test_fe_features_step,
        merge_test_fe_total_features_step,
    ],
)


def process_train_data():
    print("Processing train data...")
    transformed_train_data = transform_train_pipeline.run(load_train_data(None))

    print("Shape:", transformed_train_data.shape)
    print("Columns:", transformed_train_data.columns)

    save_pickle_data("cache/transformed_train_data.pkl", transformed_train_data)

    print("Data saved")


def process_test_data():
    print("Processing test data...")
    transformed_test_data = transform_test_pipeline.run(load_test_data(None))

    print("Shape:", transformed_test_data.shape)
    print("Columns:", transformed_test_data.columns)

    save_pickle_data("cache/transformed_test_data.pkl", transformed_test_data)

    print("Data saved")


def save_pickle_data(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)

    return data


def load_split_processed_data():
    with open("cache/transformed_train_data.pkl", "rb") as f:
        train_data = convert_to_int(pickle.load(f))

    with open("cache/transformed_test_data.pkl", "rb") as f:
        test_data = convert_to_int(pickle.load(f))

    return (train_data, test_data)


def load_processed_data():
    with open("cache/processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    train_data, test_data = data

    train_data = convert_to_int(train_data)
    test_data = convert_to_int(test_data)

    return (train_data, test_data)


def process_data():
    train_data_pipeline = Pipeline(
        "train",
        [
            load_train_data_step,
            set_missings_step,
        ],
    )

    test_data_pipeline = Pipeline(
        "test",
        [
            load_test_data_step,
            set_missings_step,
        ],
    )

    train_data = train_data_pipeline.run()
    test_data = test_data_pipeline.run()

    save_pickle_data("cache/processed_data.pkl", (train_data, test_data))

    return train_data, test_data


def convert_to_int(df):
    for col in df.columns:
        if set(df[col].unique()) == {0.0, 1.0}:
            df[col] = df[col].astype(int)
    return df
