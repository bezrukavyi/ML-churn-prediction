from datetime import timedelta, datetime
import logging
import optuna
import time
import numpy as np
import pandas as pd


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ElapsedFormatter:

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=int(elapsed_seconds))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"{now} - {record.getMessage()} ({elapsed})"


class Logger(metaclass=SingletonMeta):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(ElapsedFormatter())
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


def create_or_load_optuna_study(*, study_name, storage, direction, **extra_args):
    study_summaries = optuna.get_all_study_summaries(storage)
    for study_summary in study_summaries:
        if study_summary.study_name == study_name:
            print(f"Loading existing study: {study_name}")
            return optuna.load_study(study_name=study_name, storage=storage)

    print(f"Creating new study: {study_name}")
    return optuna.create_study(study_name=study_name, storage=storage, direction=direction)


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_type):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
