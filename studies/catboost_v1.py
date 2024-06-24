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
import catboost as cb


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "new_feature_selection_v1_with_oversampling_v1"

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
not_churn_count_strategy = int(not_churn_data_count * 1)
churn_count_strategy = int(not_churn_data_count * 1)

# Undersampling -> Oversampling
rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: not_churn_count_strategy})
train_x, train_y = rus.fit_resample(train_x, train_y)

smote = SMOTE(random_state=SEED, sampling_strategy={0: not_churn_count_strategy, 1: churn_count_strategy})
resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

# resampled_x, resampled_y = train_x, train_y

print("CHURN: ", Counter(resampled_y)[1])
print("NOT CHURN: ", Counter(resampled_y)[0])

static_params = {
    "random_seed": SEED,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "use_best_model": True,
}


def objective(trial):
    params = {
        **static_params,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 16),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.2, 1.0),
        "border_count": trial.suggest_int("border_count", 15, 512),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 500),
    }

    f1_scores = []
    auc_scores = []

    model = cb.CatBoostClassifier(**params)

    model.fit(
        resampled_x,
        resampled_y,
        eval_set=(valid_x, valid_y),
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose=False,
        use_best_model=True,
    )

    y_pred_proba = model.predict_proba(valid_x)[:, 1]
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1_score = sklearn.metrics.f1_score(valid_y, y_pred)
    auc_score = sklearn.metrics.roc_auc_score(valid_y, y_pred_proba)

    f1_scores.append(f1_score)
    auc_scores.append(auc_score)

    return 0.6 * np.mean(auc_scores) + 0.4 * np.mean(f1_scores)


study_name = f"CatboostV1_{version}"

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///example.db",
    direction="maximize",
    load_if_exists=True,
)

try:
    study.optimize(objective, n_trials=1)

finally:
    trial = study.best_trial

    model_params = {**static_params, **trial.params}

    print("Best params:", model_params)

    print("Fitting model...")
    model.fit(
        resampled_x,
        resampled_y,
        eval_set=(valid_x, valid_y),
        early_stopping_rounds=model_params["early_stopping_rounds"],
        verbose=False,
        use_best_model=True,
    )

    print("Saving model...")
    save_model(model, study_name, list(train_x.columns))

    # save best params
    with open(f"models/{study_name}.json", "wb") as f:
        f.write(json.dumps(model_params).encode())

    print("Done!")
