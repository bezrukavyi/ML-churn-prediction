import optuna
from steps.prepare_data import load_processed_data
from utils.model import predict


train_data, test_data = load_processed_data()

model_name = "LightGbmV1_hyperparemeter_optimization_14_06_2024_v1.pickle"

print("Train")
predict(model_name, train_data)

print("Test")
predict(model_name, test_data)
