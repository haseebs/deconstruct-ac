import torch
import time
import random
import numpy as np
import torch.nn as nn

from .base_models.tilecoding import TileCoder
from .base_models.mlp import MLP


class TC_MLP(nn.Module):
    # TODO maybe pass the cfg.agent as arg instead
    def __init__(self, n_actions, num_tilings, num_tiles, action_min, action_max, n_bins) -> None:
        super().__init__()
        self.action_max = action_max
        self.action_min = action_min

        self.tc = TileCoder(ndims=n_actions,
                            num_tilings=num_tilings,
                            ranges=[(action_min,action_max) for _ in range(n_actions)],
                            num_tiles=[num_tiles] * n_actions)

        self.value_net = MLP(input_size=self.tc.total_tiles)
        self.value_net.fc1.weight.data[0] = 0 # optimistic init

        self.__precomputed_features: torch.Tensor | None = None
        self.__feature_idx_to_action: list[float] | None = None
        self.__generate_all_features(n_bins)

    def __generate_feature(self, action: list[float]) -> np.ndarray:
        """
        action: a single action [action_dim]
        """
        indices = self.tc.transform(action)
        features = np.zeros(self.tc.total_tiles)
        features[indices] = 1
        return features

    def __generate_all_features(self, n_bins: int) -> None:
        all_features = []
        feature_indices = []
        action_list = np.arange(n_bins) * self.action_max / n_bins
        for k_p in action_list:
            for k_i in action_list:
                for k_d in action_list:
                    action = [k_p, k_i, k_d]
                    all_features.append(self.__generate_feature(action))
                    feature_indices.append(action)
        self.__precomputed_features = torch.FloatTensor(np.asarray(all_features))
        self.__feature_idx_to_action = np.asarray(feature_indices)

    def forward(self, actions: list[list[float]]) -> torch.Tensor:
        """
        actions: list of actions [n_actions, action_dim]
        """
        if len(actions) == 1:
            features = self.__generate_feature(actions[0])
        else:
            features = [self.__generate_feature(a) for a in actions]
            features = np.asarray(features)
        return self.value_net(torch.FloatTensor(features))

    def forward_all_features(self) -> torch.Tensor:
        assert self.__precomputed_features is not None, "__precomputed_features doesn't exist"
        return self.value_net(self.__precomputed_features)

    def get_actions_from_idx(self, feature_idx: float | list[float]) -> list[float] | list[list[float]]:
        return self.__feature_idx_to_action[feature_idx]
