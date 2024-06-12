import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import importlib

import load_data

load_data = importlib.reload(load_data).load_data


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = load_data()
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    # optuna create-study --storage mysql://user:password@127.0.0.1:3306/example --study-name testtest

    hostname = "127.0.0.1"
    port = 3306  # default MySQL port
    username = "user"
    password = "password"
    database = "example"

    # Create the connection string
    connection_string = f"mysql://{username}:{password}@{hostname}:{port}/{database}"

    # Load the study
    study = optuna.load_study(study_name="testtest", storage=connection_string)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Value: {}".format(trial.value))

    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Trial 3 finished with value: 0.9480066666666667 and parameters: {'lambda_l1': 0.0024824191460962666, 'lambda_l2': 0.007096591922852544, 'num_leaves': 236, 'feature_fraction': 0.7525409268789377, 'bagging_fraction': 0.5292340434728193, 'bagging_freq': 7, 'min_child_samples': 42}
