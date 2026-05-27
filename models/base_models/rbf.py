import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


EPSILON = 1e-6


class RBF():
    def __init__(self, n_centers=20, action_min=0, action_max=20, var=1):
        self.centers = np.linspace(action_min, action_max, num=n_centers)
        self.vars = np.ones(n_centers) * var

    def get_features(self, x):
        distances = np.power(x - self.centers, 2)
        features = np.exp(-distances / (2 * np.power(self.vars, 2)))
        return features

class RBFMultiActions():
    def __init__(self, n_centers=20, action_min=0, action_max=20, var=1, n_actions=3):
        self.centers = np.linspace([action_min]*n_actions, [action_max]*n_actions, num=n_centers)
        self.vars = np.ones([n_centers, n_actions]) * var

    def get_features(self, x):
        distances = np.power(x - self.centers, 2)
        features = np.exp(-distances / (2 * np.power(self.vars, 2)))
        return features
