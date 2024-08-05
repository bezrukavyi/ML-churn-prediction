import pickle
from utils.pipeline import PipelineStep
from utils.helpers import SingletonMeta


class LoadData(metaclass=SingletonMeta):
    def __init__(self):
        with open("./data/churn_model_pd2.pcl", "rb") as f:
            bdl_data = pickle.load(f)

        info_data, train_data, test_data = bdl_data

        self.info_data = info_data
        self.train_data = train_data
        self.test_data = test_data

        _, df_train_fe, df_train_bnum, df_train_dpi = self.train_data
        _, df_test_fe, df_test_bnum, df_test_dpi = self.test_data

        # Train
        self.df_train_fe = df_train_fe
        self.df_train_bnum = df_train_bnum
        self.df_train_dpi = df_train_dpi
        # Test
        self.df_test_fe = df_test_fe
        self.df_test_bnum = df_test_bnum
        self.df_test_dpi = df_test_dpi


def load_train_data(_):
    return LoadData().df_train_fe


def load_test_data(_):
    return LoadData().df_test_fe
