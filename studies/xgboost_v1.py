import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import optuna
from utils.helpers import create_or_load_optuna_study
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from utils.model import save_model
import xgboost as xgb
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

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "new_features_v3_tuning_v2"

# process_train_data()
# process_test_data()

train_data, test_data = load_split_processed_data()

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

train_x = train_data.drop("target", axis=1)
train_y = train_data.target

valid_x = test_data.drop("target", axis=1)[train_x.columns]
valid_y = test_data.target

smote = SMOTE(sampling_strategy="auto", random_state=SEED)
resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

dtrain = xgb.DMatrix(resampled_x, label=resampled_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)

print("Scale_pos_weight: ", Counter(resampled_y)[0] / Counter(resampled_y)[1])

static_params = {"random_state": SEED, "n_jobs": 4}


def objective(trial):
    params = {
        **static_params,
        "verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 5, 20, step=1),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    f1_scores = []
    auc_scores = []

    for train_index, val_index in kf.split(resampled_x):
        X_train = resampled_x.iloc[train_index]
        y_train = resampled_y.iloc[train_index]

        X_val = resampled_x.iloc[val_index]
        y_val = resampled_y.iloc[val_index]

        kfold_dtrain = xgb.DMatrix(X_train, label=y_train)
        kfold_dvalid = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            kfold_dtrain,
            evals=[(kfold_dvalid, "validation")],
        )

        y_pred_proba = model.predict(kfold_dvalid)

        threshold = 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)

        f1_score = sklearn.metrics.f1_score(y_val, y_pred)
        auc_score = sklearn.metrics.roc_auc_score(y_val, y_pred_proba)

        f1_scores.append(f1_score)
        auc_scores.append(auc_score)

    validation_gbm = xgb.train(
        params,
        dtrain,
        evals=[(dvalid, "validation")],
    )

    y_pred_proba = validation_gbm.predict(dvalid)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1_score = sklearn.metrics.f1_score(valid_y, y_pred)
    auc_score = sklearn.metrics.roc_auc_score(valid_y, y_pred_proba)

    f1_scores.append(f1_score)
    auc_scores.append(auc_score)

    return 0.4 * np.mean(auc_scores) + 0.6 * np.mean(f1_scores)


study_name = f"XgboostV1_{version}"

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

    print("Best params:", model_params)

    print("Fitting model...")
    model = xgb.train(
        model_params,
        dtrain,
        evals=[(dvalid, "validation")],
    )

    print("Saving model...")
    save_model(model, study_name, list(train_x.columns))

    # save best params
    with open(f"models/{study_name}.json", "wb") as f:
        f.write(json.dumps(model_params).encode())

    print("Done!")
