import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import json
import pickle
import optuna
from utils.helpers import create_or_load_optuna_study
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from utils.model import save_model
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from imblearn.over_sampling import SMOTE
from steps.prepare_data import load_split_processed_data, process_train_data, process_test_data
import sklearn
import random
from sklearn.model_selection import train_test_split, KFold
from collections import Counter
import warnings
import pdb
from imblearn.under_sampling import RandomUnderSampler


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "all_features_v1_with_oversampling_1_priority_v1_auc_score"

# process_train_data()
# process_test_data()


def oversampling(train_data, test_data):
    train_x = train_data.drop(columns="target")
    train_y = train_data.target

    valid_x = test_data.drop("target", axis=1)[train_x.columns]
    valid_y = test_data.target

    not_churn_data_count = train_data[train_data.target == 0].shape[0]

    not_churn_count_strategy = int(not_churn_data_count * 0.6)
    churn_count_strategy = int(not_churn_data_count * 0.7)

    rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: not_churn_count_strategy})
    train_x, train_y = rus.fit_resample(train_x, train_y)

    smote = SMOTE(random_state=SEED, sampling_strategy={0: not_churn_count_strategy, 1: churn_count_strategy})
    resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

    return resampled_x, resampled_y, valid_x, valid_y


train_data, test_data = load_split_processed_data()

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

resampled_x, resampled_y, valid_x, valid_y = oversampling(train_data, test_data)

dtrain = lgb.Dataset(resampled_x, label=resampled_y)
dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)

print("CHURN: ", Counter(resampled_y)[1])
print("NOT CHURN: ", Counter(resampled_y)[0])

static_params = {
    "random_state": SEED,
    "seed": SEED,
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "feature_pre_filter": False,
}


def objective(trial):
    params = {
        **static_params,
        "lambda_l1": trial.suggest_float("lambda_l1", 2, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 4, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 100, 420),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 10, 20),  # MAXIMUM 15, now the best was 10
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 200),
    }

    f1_scores = []
    auc_scores = []

    pruned_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

    validation_gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        callbacks=[pruned_callback],
    )

    y_pred_proba = validation_gbm.predict(valid_x, num_iteration=validation_gbm.best_iteration)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1_score = sklearn.metrics.f1_score(valid_y, y_pred)
    auc_score = sklearn.metrics.roc_auc_score(valid_y, y_pred_proba)

    f1_scores.append(f1_score)
    auc_scores.append(auc_score)

    return 0.6 * np.mean(auc_scores) + 0.4 * np.mean(f1_scores)


study_name = f"LightGbmV2_{version}"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///example.db",
    direction="maximize",
    load_if_exists=True,
)

try:
    study.optimize(objective, n_trials=1000)

finally:
    trial = study.best_trial

    model_params = {**static_params, **trial.params}

    print("Best params:", model_params)

    print("Fitting model...")
    model = lgb.train(
        model_params,
        dtrain,
        valid_sets=[dvalid],
    )

    print("Saving model...")
    save_model(model, study_name, list(train_x.columns))

    # save best params
    with open(f"models/{study_name}.json", "wb") as f:
        f.write(json.dumps(model_params).encode())

    print("Done!")
