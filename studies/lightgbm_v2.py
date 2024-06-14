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
from collections import Counter


hostname = "127.0.0.1"
port = 3306
username = "user"
password = "password"
database = "example"
mysql_connection_string = f"mysql://{username}:{password}@{hostname}:{port}/{database}"


class LightGbmV2:
    def __init__(self):
        pass

    def train(self, *, data, version=1.0):
        smote = SMOTE(sampling_strategy="minority", random_state=42)
        train_data, test_data = data

        train_x = train_data.drop("target", axis=1)
        train_y = train_data.target

        valid_x = test_data.drop("target", axis=1)
        valid_y = test_data.target

        smote = SMOTE(sampling_strategy="minority", random_state=42)
        resampled_x, resampled_y = smote.fit_resample(train_x, train_y)

        counter = Counter(resampled_y)
        neg = counter[0]
        pos = counter[1]
        scale_pos_weight = neg / pos

        print(scale_pos_weight)

        static_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "scale_pos_weight": scale_pos_weight,
        }

        def objective(trial):
            dtrain = lgb.Dataset(resampled_x, label=resampled_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

            param = {
                **static_params,
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "max_depth": trial.suggest_int("max_depth", 10, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 150),
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

        try:
            study_name = f"LightGbmV2_{version}"

            study = create_or_load_optuna_study(
                study_name=f"LightGbmV2_{version}",
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

            model_params.pop("early_stopping_rounds")

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
