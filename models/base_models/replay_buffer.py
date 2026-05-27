import torch
import random
import numpy as np
import torch.nn as nn


class ReplayBuffer():
    def __init__(self):
        self.action_buffer = []
        self.target_buffer = []
        self.features = []

    def append(self, action, target, features=None):
        self.action_buffer.append(action)
        self.target_buffer.append(target)
        self.features.append(features)

    def sample(self, n_samples=1):
        if not len(self.action_buffer) or n_samples > len(self.action_buffer):
            return []
        idxes = random.sample(range(len(self.action_buffer)), k=n_samples)
        return np.asarray(self.action_buffer)[idxes], np.asarray(self.target_buffer)[idxes], np.asarray(self.features)[idxes]

    def get_all_samples(self):
        return self.sample(n_samples = len(self.action_buffer))
    
    def get_last_sample(self, num_samples=1):
        if not len(self.action_buffer):
            return []
        return np.asarray(self.action_buffer)[-num_samples:], np.asarray(self.target_buffer)[-num_samples:], np.asarray(self.features)[-num_samples:]
    
    def __len__(self):
        return len(self.action_buffer)
