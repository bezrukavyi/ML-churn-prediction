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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import random


hostname = "127.0.0.1"
port = 3306
username = "user"
password = "password"
database = "example"
mysql_connection_string = f"mysql://{username}:{password}@{hostname}:{port}/{database}"


class RandomForestV1:
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
            n_estimators = trial.suggest_categorical(name="n_estimators", choices=[100, 200, 300, 400, 500])
            max_depth = trial.suggest_categorical(name="max_depth", choices=[10, 30, 50, 70, 90, 110])
            min_samples_split = trial.suggest_categorical(name="min_samples_split", choices=[2, 4, 6, 8, 10])
            min_samples_leaf = trial.suggest_categorical(name="min_samples_leaf", choices=[1, 2, 3, 4])

            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }

            print(f"Trying params: {params}")

            model = RandomForestClassifier(**params)

            print("Fitting model...")
            model.fit(train_x, train_y)

            print("Predicting...")
            pred = model.predict(valid_x)
            score = sklearn.metrics.f1_score(valid_y, pred)

            print(f"Score: {score}")

            return score

        try:
            study_name = f"RandomForestV1_{version}"

            study = create_or_load_optuna_study(
                study_name=f"RandomForestV1_{version}",
                storage=mysql_connection_string,
                direction="maximize",
            )

            study.optimize(objective, n_trials=20)

        finally:
            trial = study.best_trial

            model_params = {**static_params, **trial.params}

            print("Best params: ", model_params)

            model = RandomForestClassifier(**model_params)

            print("Fitting model...")
            model.fit(train_x, train_y)

            print("Saving model...")
            save_model(model, study_name, list(train_x.columns))

            # save best params
            with open(f"models/{study_name}.json", "wb") as f:
                f.write(json.dumps(model_params).encode())

            print("Done!")
