import logging
from functools import reduce
from typing import List, Callable, Any, Optional
from utils.helpers import Logger


logger = Logger().get_logger()


class PipelineStep:
    def __init__(self, name: str, func: Callable[[Any], Any] = None):
        self.name = name
        self.func = func

    def run(self, data: Any) -> Any:
        return self.func(data)


class Pipeline:
    def __init__(self, name, steps: List[PipelineStep] = []):
        self.name = name
        self._steps: List[PipelineStep] = steps

    def run(self, data: Optional[Any] = None) -> Any:
        print("-----------------")
        logger.info(f"Running pipeline: {self.name}")
        result = reduce(self._log_and_run_step, enumerate(self._steps), data)
        logger.info(f"{len(self._steps) + 1}. Done")
        print("-----------------\n")
        return result

    def _log_and_run_step(self, data: Any, step_with_index: tuple[int, PipelineStep]) -> Any:
        index, step = step_with_index
        logger.info(f"{index}. Running step: {step.name}")

        return step.run(data)
