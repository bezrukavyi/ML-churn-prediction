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
import xgboost as xgb
import random


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

        static_params = {"random_state": SEED, "n_jobs": 4}

        def objective(trial):
            dtrain = xgb.DMatrix(train_x, label=train_y)
            dvalid = xgb.DMatrix(valid_x, label=valid_y)

            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                "tree_method": "exact",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if param["booster"] in ["gbtree", "dart"]:
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

            bst = xgb.train(param, dtrain)

            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            f1_score = sklearn.metrics.f1_score(valid_y, pred_labels)
            return f1_score

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
            model.fit(train_x, train_y)

            print("Saving model...")
            save_model(model, study_name, list(train_x.columns))

            # save best params
            with open(f"models/{study_name}.json", "wb") as f:
                f.write(json.dumps(model_params).encode())

            print("Done!")
