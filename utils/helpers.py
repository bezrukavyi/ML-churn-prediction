from datetime import timedelta, datetime
import logging
import optuna
import time


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
