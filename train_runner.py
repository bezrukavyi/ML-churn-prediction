import optuna
from steps.prepare_data import process_data, load_processed_data
from studies.lightgbm_v1 import LightGbmV1

# process_data()

LightGbmV1().train(data=load_processed_data(), version="refactoring_v2")
