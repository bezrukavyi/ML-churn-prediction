import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
import json
import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from utils.model import save_model
from steps.feature_selection import FEATURES
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from steps.prepare_data import load_split_processed_data, process_train_data, process_test_data
import sklearn
import random
from sklearn.model_selection import train_test_split, KFold
from collections import Counter
import warnings
import pdb
from utils.helpers import reduce_mem_usage
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import utils.nn_engine as engine
import torch.nn as nn

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    device = "mps"
elif torch.cuda.is_available():
    x = torch.ones(1, device="cuda")
    device = "cuda"
else:
    device = "cpu"

print(f"device: {device}")

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

version = "new_feature_selection_v1_with_oversampling_v1"

# process_train_data()
# process_test_data()

train_data, test_data = load_split_processed_data()

train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)


print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

FEATURES_COUNT = 100
FEATURES_SELECTED = FEATURES[:FEATURES_COUNT]

features = train_data.drop("target", axis=1)[FEATURES_SELECTED].columns.tolist()

train_x = train_data.drop("target", axis=1)[features]
train_y = train_data.target

valid_x = test_data.drop("target", axis=1)[features]
valid_y = test_data.target

churn_data_count = train_data[train_data.target == 1].shape[0]

# Undersampling -> Oversampling
rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: churn_data_count})
train_x, train_y = rus.fit_resample(train_x, train_y)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)

# smote = SMOTE(random_state=SEED, sampling_strategy={0: not_churn_count_strategy, 1: churn_count_strategy})
# train_x, train_y = smote.fit_resample(train_x, train_y)

print("CHURN: ", Counter(train_y)[1])
print("NOT CHURN: ", Counter(train_y)[0])


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer_stack(x)


class LargeDataset(Dataset):
    def __init__(self, x_data, y_data, device):
        self.x_data = x_data
        self.y_data = y_data
        self.device = device

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_data.iloc[idx], dtype=torch.float32).to(self.device)
        return x, y


train_dataloader = DataLoader(
    LargeDataset(train_x, train_y, device=device),
    batch_size=32,
    shuffle=True,
    pin_memory=True,
)

test_dataloader = DataLoader(
    LargeDataset(train_x, train_y, device=device),
    batch_size=32,
    shuffle=True,
    pin_memory=True,
)

EPOCHS = 100

model = NeuralNetwork(input_shape=FEATURES_COUNT, hidden_units=60).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=0)

model_results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    device=device,
)
