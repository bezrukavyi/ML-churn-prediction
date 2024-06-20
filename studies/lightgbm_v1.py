import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import optuna
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
import warnings
import pdb
import random
from steps.prepare_data import load_split_processed_data, process_train_data, process_test_data

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "new_features_v1_tuning_v1"

# process_train_data()
# process_test_data()

train_data, test_data = load_split_processed_data()

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

train_x = train_data.drop("target", axis=1)
train_y = train_data.target

valid_x = test_data.drop("target", axis=1)[train_x.columns]
valid_y = test_data.target

dtrain = lgb.Dataset(train_x, label=train_y)
dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)

static_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "is_unbalance": True,
}


def objective(trial):
    param = {
        **static_params,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 200),
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        callbacks=[pruning_callback],
    )

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)

    f1_score = sklearn.metrics.f1_score(valid_y, pred_labels)

    return f1_score


study_name = f"LightGbmV1_{version}"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///example.db",
    direction="maximize",
    load_if_exists=True,
)

try:
    study.optimize(objective, n_trials=100)

finally:
    trial = study.best_trial

    model_params = {**static_params, **trial.params}

    model_params.pop("early_stopping_rounds")

    model = lgb.LGBMClassifier(**model_params)

    train_x = train_data.drop("target", axis=1)
    train_y = train_data.target

    print("Fitting model...")
    model.fit(train_x, train_y)

    print("Saving model...")
    save_model(model, study_name, list(train_x.columns))

    # save best params
    with open(f"models/{study_name}.json", "wb") as f:
        f.write(json.dumps(model_params).encode())

    print("Done!")
