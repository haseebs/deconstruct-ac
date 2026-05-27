import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from scipy.special import gamma, digamma
UPPER = 1e64


class BetaNN(nn.Module):
    """
    num_action independent Beta distributions. Sample from distributions and then map them to [action_min, action_max]
    """
    def __init__(self, num_actions, mean, shape,
                 action_min, action_max, clip_stddev=1, log_prob_reduction: str | None = None):

        super(BetaNN, self).__init__()

        self.num_actions = num_actions

        self.mean = nn.Parameter(torch.FloatTensor([mean]*num_actions), requires_grad=True) 
        self.shape = nn.Parameter(torch.FloatTensor([shape]*num_actions), requires_grad=True)

        # Action rescaling
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)
        self.shape_bias = torch.FloatTensor([1.])
        self.log_prob_reduction = log_prob_reduction or "sum"
        if self.log_prob_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid log_prob_reduction={self.log_prob_reduction!r}; expected 'mean' or 'sum'.")

    def _reduce_log_prob(self, log_prob_per_dim: torch.Tensor) -> torch.Tensor:
        if self.log_prob_reduction == "mean":
            return log_prob_per_dim.mean(dim=1)
        return log_prob_per_dim.sum(dim=1)

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.shape.reshape(-1).tolist(), [0]

    def forward(self):
        """
        Performs the forward pass through the network, predicting the mean and shape.

        Parameters
        ----------
        num_samples : int
        The number of actions to sample

        Returns
        -------
        2-tuple of torch.Tensor of float
        The mean and shape of the Beta policy in the
        argument state
        """

        mean = torch.nn.functional.softplus(self.mean) + self.shape_bias
        shape = torch.nn.functional.softplus(self.shape) + self.shape_bias

        return mean, shape

    def rsample(self, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Returns
        -------
        torch.Tensor of float
        A sampled action
        """
        mean, shape = self.forward()
        actions = torch.zeros((num_samples, self.num_actions))
        log_probs = torch.zeros((num_samples, self.num_actions))
        for i in range(self.num_actions):
            beta = Beta(mean[i], shape[i])
            original_action = beta.rsample((num_samples, ))
            # original_action = beta.sample((num_samples, ))
            log_prob = beta.log_prob(original_action)

            action = original_action * (self.action_max - self.action_min) + self.action_min
            action = (torch.clamp(action, self.action_min, self.action_max)).squeeze(-1)

            actions[:, i] = action
            log_probs[:, i] = log_prob

        if num_samples == 1:
            action = action.squeeze(0)

        # if self.num_actions == 1:
        #     log_prob.unsqueeze(-1)

        return actions, self._reduce_log_prob(log_probs), (mean, shape), None

    def sample(self, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        num_samples : int
        The number of actions to sample

        Returns
        -------
        torch.Tensor of float
        A sampled action
        """
        mean, shape = self.forward()
        actions = torch.zeros((num_samples, self.num_actions))
        log_probs = torch.zeros((num_samples, self.num_actions))
        try:
            for i in range(self.num_actions):
                beta = Beta(mean[i], shape[i])
                original_action = beta.sample((num_samples, ))
                log_prob = beta.log_prob(original_action)

                action = original_action * (self.action_max - self.action_min) + self.action_min
                action = (torch.clamp(action, self.action_min, self.action_max)).squeeze(-1)
                actions[:, i] = action
                log_probs[:, i] = log_prob

            if num_samples == 1:
                actions = actions.squeeze(0)
        except:
            from IPython import embed; embed(); exit()

        return actions, self._reduce_log_prob(log_probs), (mean, shape), None

    def log_prob(self, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, shape = self.forward()
        # if there is only one action
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        log_probs = torch.zeros_like(actions)
        for i in range(self.num_actions):
            beta = Beta(mean[i], shape[i])
            original_action = (actions[:, i] - self.action_min) / (self.action_max - self.action_min)
            log_prob = beta.log_prob(original_action)
            log_probs[:, i] = log_prob

        if self.num_actions == 1:
            log_probs.unsqueeze(-1)

        if show:
            print(f"mean {mean}, shape: {shape}")

        return self._reduce_log_prob(log_probs)


    def entropy(self):
        mean, shape = [param.detach().numpy() for param in self.forward()]
        # mean, shape = self.forward()
        logBeta = np.log(np.minimum(gamma(mean), UPPER)) + np.log(np.minimum(gamma(shape), UPPER)) - np.log(np.minimum(gamma(mean+shape), UPPER))
        ent = torch.FloatTensor(logBeta - (mean - 1)*digamma(mean) - (shape-1)*digamma(shape) + (mean+shape-2)*digamma(mean+shape))
        ent.requires_grad = True
        return ent


    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
        The device to move the network to

        Returns
        -------
        nn.Module
        The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(BetaNN, self).to(device)
