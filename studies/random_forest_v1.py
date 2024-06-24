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
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "new_feature_selection_v1_with_tuning_v2"

# process_train_data()
# process_test_data()

train_data, test_data = load_split_processed_data()

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

train_x = train_data.drop("target", axis=1)
train_y = train_data.target

valid_x = test_data.drop("target", axis=1)[train_x.columns]
valid_y = test_data.target

not_churn_data_count = train_data[train_data.target == 0].shape[0]
not_churn_count_strategy = int(not_churn_data_count * 0.75)
churn_count_strategy = int(not_churn_data_count * 0.75)

# Undersampling -> Oversampling
rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: not_churn_count_strategy})
train_x, train_y = rus.fit_resample(train_x, train_y)

smote = SMOTE(random_state=SEED, sampling_strategy={0: not_churn_count_strategy, 1: churn_count_strategy})
resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

# resampled_x, resampled_y = train_x, train_y

print("CHURN: ", Counter(resampled_y)[1])
print("NOT CHURN: ", Counter(resampled_y)[0])

static_params = {
    "random_state": SEED,
}


def objective(trial):
    params = {
        **static_params,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_features": trial.suggest_categorical(name="max_features", choices=["log2", "sqrt"]),
        "max_depth": trial.suggest_int("max_depth", 5, 20, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    f1_scores = []
    auc_scores = []

    for train_index, val_index in kf.split(resampled_x):
        X_train = resampled_x.iloc[train_index]
        y_train = resampled_y.iloc[train_index]

        X_val = resampled_x.iloc[val_index]
        y_val = resampled_y.iloc[val_index]

        model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)
        threshold = 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)

        f1_score = sklearn.metrics.f1_score(y_val, y_pred)
        auc_score = sklearn.metrics.roc_auc_score(y_val, y_pred_proba)

        f1_scores.append(f1_score)
        auc_scores.append(auc_score)

    model = RandomForestClassifier(**params)

    model.fit(train_x, train_y)

    y_pred_proba = model.predict_proba(valid_x)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1_score = sklearn.metrics.f1_score(valid_y, y_pred)
    auc_score = sklearn.metrics.roc_auc_score(valid_y, y_pred_proba)

    f1_scores.append(f1_score)
    auc_scores.append(auc_score)

    return 0.5 * np.mean(auc_scores) + 0.5 * np.mean(f1_scores)


study_name = f"RandomForestClassifierV1_{version}"

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
    model = RandomForestClassifier(**model_params)

    model.fit(train_x, train_y)

    print("Saving model...")
    save_model(model, study_name, list(train_x.columns))

    # save best params
    with open(f"models/{study_name}.json", "wb") as f:
        f.write(json.dumps(model_params).encode())

    print("Done!")
