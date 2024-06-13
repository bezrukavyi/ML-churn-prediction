import pickle
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

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

    Metrics().call(y_true, y_pred)

    return model.predict(X)


class Metrics:
    def __init__(self):
        pass

    def call(self, y_true, y_pred):
        print("\nMetrics")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("FPR:", self.fpr_score(y_true, y_pred), "\n")

    def specificity_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def fpr_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (tn + fp)
