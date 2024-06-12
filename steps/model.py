import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

import lightgbm as lgbm


def save_model(model=None, version=1.0, features=[]):
    name = str(model.__class__).split(".")[-1][:-2] + "_" + str(version) + ".pickle"
    if model:
        with open(f"models/{name}", "wb") as file:
            pickle.dump((model, features), file)
            print("Save", name)


def predict(dataframe, name):
    model, features = load(name)

    dataframe = dataframe.copy()

    X = dataframe[features]

    return model.predict(X)


def load(file_name):
    with open(f"models/{file_name}", "rb") as file:
        model_tpl = pickle.load(file)

    return model_tpl


class Predictor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.accuracy_score = accuracy_score
        self.precision_score = precision_score

    def call(self, dataframe):
        X = dataframe.drop(columns=["target"], axis=1)
        y_true = dataframe.target

        y_pred = predict(X, self.model_name)

        print("\nMetrics")
        print("Accuracy:", self.accuracy_score(y_true, y_pred))
        print("Precision:", self.precision_score(y_true, y_pred))
        print("FPR:", self.fpr_score(y_true, y_pred), "\n")

    def specificity_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def fpr_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (tn + fp)
