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


hostname = "127.0.0.1"
port = 3306
username = "user"
password = "password"
database = "example"
mysql_connection_string = f"mysql://{username}:{password}@{hostname}:{port}/{database}"


class LightGbmV1:
    def __init__(self):
        pass

    def train(self, *, data, version=1.0):
        static_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "scale_pos_weight": 94,
        }

        def objective(trial):
            train_data, test_data = data

            train_x = train_data.drop("target", axis=1)
            train_y = train_data.target

            valid_x = test_data.drop("target", axis=1)
            valid_y = test_data.target

            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

            param = {
                **static_params,
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                # "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                # "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
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
            accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

            return accuracy

        try:
            study_name = f"LightGbmV1_{version}"

            study = create_or_load_optuna_study(
                study_name=f"LightGbmV1_{version}",
                storage=mysql_connection_string,
                direction="maximize",
            )

            study.optimize(objective, n_trials=20)

        finally:
            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("Value: {}".format(trial.value))

            print("Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            model_params = {**static_params, **trial.params}

            model = lgb.LGBMClassifier(**model_params)

            train_data, _ = data

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
