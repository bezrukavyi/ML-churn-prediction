from datetime import timedelta, datetime
import logging
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
