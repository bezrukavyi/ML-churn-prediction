import pickle
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
import ipdb

import lightgbm as lgbm


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


def predict(model_name, dataframe):
    model, features = load_model(model_name)

    X = dataframe[features]
    y_true = dataframe.target
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    Metrics().call(y_true, y_pred, y_pred_proba)

    return model.predict(X)


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
