import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


EPSILON = 1e-6


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def layer_init_zero(layer):
    nn.init.constant_(layer.weight, 0)
    return layer


def layer_init_xavier(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def layer_init_uniform(layer, low=-0.003, high=0.003):
    nn.init.uniform_(layer.weight, low, high)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(nn.Module):
    def __init__(self, input_size=20):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        return x

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_units=64, hidden_layers=1, activation=torch.nn.functional.relu):
        super().__init__()
        _layers = []
        _layers.append(layer_init_xavier(nn.Linear(input_size, hidden_units)))
        for _ in range(hidden_layers):
            _layers.append(layer_init_xavier(nn.Linear(hidden_units, hidden_units)))
        self.layers = nn.ModuleList(_layers)
        self.fc_head = layer_init_xavier(nn.Linear(hidden_units, 1))
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.fc_head(x)

class FCNNOldRepo(nn.Module):
    def __init__(self, input_size, hidden_units=64, hidden_layers=1, activation=torch.nn.functional.relu):
        super().__init__()
        hidden_units = tuple(hidden_units for i in range(hidden_layers))
        dims = (input_size,) + hidden_units
        self.layers = nn.ModuleList([layer_init_xavier(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.activation = activation
        self.feature_dim = dims[-1]
        self.fc_head = layer_init_xavier(nn.Linear(self.feature_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.fc_head(x)
        return x
