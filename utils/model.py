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


class Metrics:
    def __init__(self):
        pass

    def call(self, y_true, y_pred, y_pred_proba):
        print("\nMetrics")

        # AUC
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
        print(f"AUC: {auc:.3f}")

        matthews_corrcoef = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
        print(f"Matthews Correlation Coefficient: {matthews_corrcoef:.3f}")

        # classification report
        class_report = sklearn.metrics.classification_report(y_true, y_pred)
        print("Classification Report:")
        print(class_report)

        return auc, class_report
