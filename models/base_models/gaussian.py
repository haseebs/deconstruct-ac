import torch
import time
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


EPSILON = 1e-6


class Gaussian(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)` using a `tanh` activation.
    """
    def __init__(self, num_actions, mean_init, std_init,
                 action_min, action_max, clip_stddev=10, log_prob_reduction: str | None = None):
        """
        Constructor

        Parameters
        ----------
        num_actions : int
            The dimensionality of the action vector
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(Gaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.mean = nn.Parameter(torch.FloatTensor([mean_init]*num_actions), requires_grad=True)
        self.log_std = nn.Parameter(torch.FloatTensor([std_init]*num_actions), requires_grad=True)

        # Action rescaling
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)
        self.log_prob_reduction = log_prob_reduction or "mean"
        if self.log_prob_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid log_prob_reduction={self.log_prob_reduction!r}; expected 'mean' or 'sum'.")

    def _reduce_log_prob(self, log_prob_per_dim: torch.Tensor) -> torch.Tensor:
        if self.log_prob_reduction == "mean":
            return log_prob_per_dim.mean(dim=-1)
        return log_prob_per_dim.sum(dim=-1)

    def forward(self):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        num_samples : int
            The number of actions to sample

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        mean = torch.tanh(self.mean)
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + \
            self.action_min  # ∈ [action_min, action_max]
        log_std = self.log_std

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.log_std.reshape(-1).tolist(), [0]

    def rsample(self, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward()
        std = log_std.exp()
        normal = Normal(mean, std)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        action = normal.rsample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)
        # if num_samples == 1:
        #     action = action.squeeze(0)

        log_prob = self._reduce_log_prob(normal.log_prob(action))
        # if self.num_actions == 1:
        #     log_prob.unsqueeze(-1)

        return action, log_prob, mean, None

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
        mean, log_std = self.forward()
        std = log_std.exp()
        normal = Normal(mean, std)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        log_prob = self._reduce_log_prob(normal.log_prob(action))

        if num_samples == 1:
            action = action.squeeze(0)

        # if self.num_actions == 1:
        #     log_prob.unsqueeze(-1)

        return action, log_prob, mean, None

    def log_prob(self, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward()
        std = log_std.exp()
        normal = Normal(mean, std)

        log_prob = self._reduce_log_prob(normal.log_prob(actions))
        # if self.num_actions == 1:
        #     log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

    def entropy(self):
        # mean, log_std = [param.detach() for param in self.forward()]
        _, log_std = self.forward()
        return 0.5 + 0.5 * math.log(2 * math.pi) + log_std

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
        return super(Gaussian, self).to(device)





class GaussianFixedStd(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)` using a `tanh` activation.
    """
    def __init__(self, num_actions, mean_init, std_init,
                 action_min, action_max, anneal_coef=1.0, clip_stddev=10):

        super(GaussianFixedStd, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.mean = nn.Parameter(torch.FloatTensor([mean_init]*num_actions), requires_grad=True)
        # the init is log_std, so we take exp. We store std directly for annealing purposes
        self.std = torch.FloatTensor([std_init]*num_actions).exp()
        # Action rescaling
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)

        self.anneal_coef = anneal_coef
        assert self.anneal_coef <= 1 and self.anneal_coef >= 0, "annealing coefficient should be between 0 and 1!"

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.std.reshape(-1).tolist(), [0]

    def forward(self):

        mean = torch.tanh(self.mean)
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + \
            self.action_min  # ∈ [action_min, action_max]

        return mean

    def get_deterministic_action(self):
        """
        rsample
        """
        mean = self.forward()
        action = mean
        action = torch.clamp(action, self.action_min, self.action_max)

        return action

    """every time this function is called anneal noise by reducing the std"""
    def anneal(self):
        self.std = self.std * self.anneal_coef
        self.std.clamp_(0.1, 3.0)


    def sample(self, num_samples=1):

        mean = self.forward()
        std = self.std
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        log_prob = normal.log_prob(action)
        if num_samples == 1:
            action = action.squeeze(0)

        return action, log_prob, mean, None

    def log_prob(self, actions, show=False):

        mean = self.forward()
        std = self.std
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)
        # if self.num_actions == 1:
        #     log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob


    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(GaussianFixedStd, self).to(device)
