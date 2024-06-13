import pickle
from steps.set_missings import set_missings
from steps.load_data import load_train_data, load_test_data
from utils.pipeline import Pipeline, PipelineStep


# Data Fetching
load_train_data_step = PipelineStep("load_train_data", load_train_data)
load_test_data_step = PipelineStep("load_test_data", load_test_data)

# Data Preprocessing
set_missings_step = PipelineStep("set_missings", set_missings)


def save_pickle_data(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)

    return data


def load_processed_data():
    with open("cache/processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    return data


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
