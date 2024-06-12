import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from steps.model import save_model
import lightgbm as lgbm


class LightGbmV1:
    def __init__(self, version=1.0):
        self.version = version

    def train(self, dataframe):
        dataframe = dataframe.copy()

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "scale_pos_weight": 94,
            "lambda_l1": 3.4925825074775134e-08,
            "lambda_l2": 1.541663754329288e-05,
            "num_leaves": 110,
        }

        X = dataframe.drop(columns=["target"], axis=1)
        y = dataframe.target

        lgbm_model = lgbm.LGBMClassifier(**params, verbose=-1)

        lgbm_model.fit(X, y)

        save_model(lgbm_model, self.version, list(X.columns))

        return lgbm_model, dataframe
