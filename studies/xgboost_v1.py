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
from datetime import datetime
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import cross_val_score
import sklearn
import xgboost as xgb
import random
from steps.prepare_data import load_split_processed_data, process_train_data, process_test_data

hostname = "127.0.0.1"
port = 3306
username = "user"
password = "password"
database = "example"
mysql_connection_string = f"mysql://{username}:{password}@{hostname}:{port}/{database}"


class XgboostV1:
    def __init__(self):
        pass

    def train(self, *, data, version=1.0):
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)

        train_data, test_data = data

        train_x = train_data.drop("target", axis=1)
        train_y = train_data.target

        valid_x = test_data.drop("target", axis=1)
        valid_y = test_data.target

        # scaler = sklearn.preprocessing.StandardScaler()
        # train_x = scaler.fit_transform(train_x)
        # valid_x = scaler.fit_transform(valid_x)

        smote = SMOTE(sampling_strategy="minority", random_state=SEED)
        resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

        static_params = {"random_state": SEED, "n_jobs": 4}

        def objective(trial):
            param = {
                **static_params,
                "verbosity": 0,
                "objective": "binary:logistic",
                "tree_method": "exact",
                "booster": "gbtree",
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 5, 15, step=1),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            }

            model = xgb.XGBClassifier(**param)
            model.fit(resampled_x, resampled_y)

            pred_proba = model.predict_proba(valid_x)[:, 1]
            # pred = model.predict(valid_x)
            # f1_score = sklearn.metrics.f1_score(valid_y, pred)
            auc_score = sklearn.metrics.roc_auc_score(valid_y, pred_proba)

            return auc_score

        try:
            study_name = f"XgboostV1_{version}"

            study = create_or_load_optuna_study(
                study_name=f"XgboostV1_{version}",
                storage=mysql_connection_string,
                direction="maximize",
            )

            study.optimize(objective, n_trials=100)

        finally:
            trial = study.best_trial

            model_params = {**static_params, **trial.params}

            print("Best params: ", model_params)

            model = xgb.XGBClassifier(**model_params)

            print("Fitting model...")
            model.fit(resampled_x, resampled_y)

            print("Saving model...")
            save_model(model, study_name, list(train_x.columns))

            # save best params
            with open(f"models/{study_name}.json", "wb") as f:
                f.write(json.dumps(model_params).encode())

            print("Done!")


# process_train_data()
# process_test_data()
XgboostV1().train(data=load_split_processed_data(), version="180924_f1_score_v2")
