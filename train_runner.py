import optuna
from steps.prepare_data import process_data, load_processed_data
from studies.xgboost_v1 import XgboostV1

# process_data()

XgboostV1().train(data=load_processed_data(), version="v1")
