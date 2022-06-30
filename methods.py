from torch import distributions
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Even if you don't use them, you still have to import
import random
import numpy as np

# Networks

class Net(nn.Module):
    def __init__(self, device=None, dtype=None, input_size=1):
        self.device = device
        self.dtype = dtype
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50, device=self.device, dtype=self.dtype)
        self.fc2 = nn.Linear(50, 30, device=self.device, dtype=self.dtype)
        self.fc3 = nn.Linear(30, 3, device=self.device, dtype=self.dtype)
        self.fc4 = nn.Linear(3, 1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.relu(x)


class NetClassifier(nn.Module):
    def __init__(self, device=None, dtype=None, input_size=1):
        self.device = device
        self.dtype = dtype
        super(NetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50, device=self.device, dtype=self.dtype)
        self.fc2 = nn.Linear(50, 30, device=self.device, dtype=self.dtype)
        self.fc3 = nn.Linear(30, 3, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LinearNet(nn.Module):
    def __init__(self, device=None, dtype=None, input_size=1):
        self.device = device
        self.dtype = dtype
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = self.fc1(x.float())
        return x  # torch.sigmoid(x) #torch.tanh(x*3-1.5) + 1 #

####################################################################################################################

# Methods


class NNClassifier:
    def __init__(self, n_observations=None, device=None, dtype=None, input_size=1, class_weights=None):
        self.optimizes = True
        self.class_weights = class_weights
        self.type = "classifier"
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = NetClassifier(device=self.device, dtype=self.dtype, input_size=self.input_size)

    def criterion(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def predict(self, x):
        return self.model(x)

    def calculate_regularization_loss(self):
        reg_loss = 0
        for param in self.model.parameters():
            reg_loss += param.norm(2) ** 2
        return reg_loss



class NNPredictor:
    def __init__(self, n_observations=None, device=None, dtype=None, input_size=1, class_weights=None):
        self.optimizes = True
        self.type = "predictor"
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = Net(device=self.device, dtype=self.dtype, input_size=self.input_size)

    def criterion(self):
        return nn.MSELoss()

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def predict(self, x):
        return self.model(x)

    def calculate_regularization_loss(self):
        reg_loss = 0
        for param in self.model.parameters():
            reg_loss += param.norm(2) ** 2
        return reg_loss


class Linear:
    def __init__(self, n_observations=None, device=None, dtype=None, input_size=1, class_weights=None):
        self.optimizes = True
        self.type = "predictor"
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = LinearNet(device=self.device, dtype=self.dtype, input_size=self.input_size)

    def criterion(self, class_weights=None):
        return nn.MSELoss()

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def predict(self, x):
        return self.model(x)

    def calculate_regularization_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)


class SVM:
    def __init__(self, device=None, dtype=None, input_size=1, class_weights=None, kernel='rbf', C=1.0, gamma=0.0):
        self.class_weights = class_weights
        self.optimizes = False
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', max_iter=100000) # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}

    def get_parameters(self):
        return None

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def calculate_regularization_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

class RF:
    def __init__(self, device=None, dtype=None, input_size=1, class_weights=None):
        self.optimizes = False
        self.input_size = input_size
        self.dtype = dtype
        self.model = RandomForestClassifier()

    def get_parameters(self):
        return None

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def calculate_regularization_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)


class AdaBoost:
    def __init__(self, device=None, dtype=None, input_size=1, class_weights=None):
        self.optimizes = False
        self.input_size = input_size
        self.dtype = dtype
        self.model = AdaBoostClassifier()

    def get_parameters(self):
        return None

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def calculate_regularization_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)