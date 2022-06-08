from torch import distributions
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Even if you don't use them, you still have to import
import random
import numpy as np






class Net(nn.Module):
    def __init__(self, device=None, dtype=None, input_size=1):
        self.device = device
        self.dtype = dtype
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 9, device=self.device, dtype=self.dtype)
        self.fc2 = nn.Linear(9, 6, device=self.device, dtype=self.dtype)
        self.fc3 = nn.Linear(6, 3, device=self.device, dtype=self.dtype)
        self.fc4 = nn.Linear(3, 1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.relu(x)


class NN:
    def __init__(self, n_observations=None, device=None, dtype=None, input_size=1):
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = Net(device=self.device, dtype=self.dtype, input_size=self.input_size)

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def predict(self, x):
        return self.model(x)

    def calculate_regularization_loss(self):
        reg_loss = 0
        for param in self.model.parameters():
            reg_loss += param.norm(2) ** 2
        return reg_loss


class LinearNet(nn.Module):
    def __init__(self, device=None, dtype=None, input_size=1):
        self.device = device
        self.dtype = dtype
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = self.fc1(x.float())
        return x  # torch.sigmoid(x) #torch.tanh(x*3-1.5) + 1 #


class Linear:
    def __init__(self, n_observations=None, device=None, dtype=None, input_size=1):
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.n_observations = n_observations
        self.model = LinearNet(device=self.device, dtype=self.dtype, input_size=self.input_size)

    def get_parameters(self):
        return list(set(self.model.parameters()))

    def predict(self, x):
        return self.model(x)

    def calculate_regularization_loss(self):
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

