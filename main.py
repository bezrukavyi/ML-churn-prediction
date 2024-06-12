from utils.pipeline import Pipeline, PipelineStep
from steps.set_missings import set_missings
from steps.load_data import load_train_data, load_test_data
from steps.model import Predictor
from steps.lightgbm_v1 import LightGbmV1

set_missings_step = PipelineStep("set_missings", set_missings)

load_train_data_step = PipelineStep("load_train_data", load_train_data)
load_test_data_step = PipelineStep("load_test_data", load_test_data)

train_step = PipelineStep("train", LightGbmV1().train)

result_step = PipelineStep("metrics", Predictor("LGBMClassifier_1.0.pickle").call)

train_pipeline = Pipeline("train", [load_train_data_step, set_missings_step, result_step])
test_pipeline = Pipeline("test", [load_test_data_step, set_missings_step, result_step])

train_pipeline.run()
test_pipeline.run()
