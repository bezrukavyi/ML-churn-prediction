import pickle
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
import ipdb
import sklearn

import lightgbm as lgbm
import xgboost as xgb


def save_model(model=None, name="model", features=[]):
    if model:
        with open(f"models/{name}.pickle", "wb") as file:
            pickle.dump((model, features), file)
            print("Save", name)

    return name


def load_model(file_name):
    with open(f"models/{file_name}", "rb") as file:
        model_tpl = pickle.load(file)

    return model_tpl


def save_final_model(model=None, name="model", features=[]):
    if model:
        with open(f"final_models/{name}.pickle", "wb") as file:
            pickle.dump((model, features), file)
            print("Save", name)

    return name


def load_final_model(file_name):
    with open(f"final_models/{file_name}", "rb") as file:
        model_tpl = pickle.load(file)

    return model_tpl


def predict(model_name, dataframe):
    model, features = load_model(model_name)

    X = dataframe[features]
    y_true = dataframe.target

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    Metrics().call(y_true, y_pred, y_pred_proba)

    return model.predict(X)


def predict_booster(model_name, dataframe, scale=False):
    model, features = load_model(model_name)

    X = dataframe[features]
    y_true = dataframe.target

    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    Metrics().call(y_true, y_pred, y_pred_proba)

    return y_pred, model, features


def predict_booster_model(model, dataframe):
    X = dataframe.drop(columns=["target"])
    y_true = dataframe.target

    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    Metrics().call(y_true, y_pred, y_pred_proba)

    return y_pred


def predict_xgbm(model_name, dataframe):
    model, features = load_model(model_name)

    X = dataframe[features]
    y_true = dataframe.target

    dvalid = xgb.DMatrix(X, label=y_true)

    y_pred_proba = model.predict(dvalid)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    Metrics().call(y_true, y_pred, y_pred_proba)

    return y_pred


def lgb_booster_to_model(model_name, X, y):
    booster, features = load_model(f"{model_name}.pickle")

    model_params = booster.params

    model_params.pop("num_iterations")
    model_params.pop("early_stopping_round")

    model_cls = lgbm.LGBMClassifier(**model_params)

    model_cls = model_cls.fit(X[features], y, init_model=booster)

    save_final_model(model_cls, model_name, features)
    return load_final_model(f"{model_name}.pickle")


class Metrics:
    def __init__(self):
        pass

    def call(self, y_true, y_pred, y_pred_proba):
        print("\nMetrics")

        # AUC
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
        print(f"AUC: {auc:.2f}")

        # classification report
        class_report = sklearn.metrics.classification_report(y_true, y_pred)
        print("Classification Report:")
        print(class_report)

        return auc, class_report

        # confusion matrix
        # conf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        # print("Confusion Matrix:")
        # print(conf_matrix)

    # def specificity_score(self, y_true, y_pred):
    #     tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    #     return tn / (tn + fp)

    # def fpr_score(self, y_true, y_pred):
    #     tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    #     return fp / (tn + fp)
