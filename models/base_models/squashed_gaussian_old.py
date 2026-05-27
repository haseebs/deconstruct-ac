import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


EPSILON = 1e-6


class SquashedGaussian(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)` using a `tanh` activation.
    """
    def __init__(self, num_actions, mean_init, std_init,
                 action_min, action_max, clip_stddev=1000):
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
        super(SquashedGaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.mean = nn.Parameter(torch.FloatTensor([mean_init]), requires_grad=True)
        self.log_std = nn.Parameter(torch.FloatTensor([std_init]), requires_grad=True)

        # Action rescaling
        #if action_space is None:
        #    self.action_scale = torch.tensor(1.)
        #   self.action_bias = torch.tensor(0.)
        self.action_scale = torch.FloatTensor(
            (action_max - action_min) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_max + action_min) / 2.)

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
        mean = self.mean
        log_std = self.log_std

        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
            return mean, log_std


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

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        x_t = normal.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

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

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        x_t = normal.sample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

    def log_prob(self, x_t_batch, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward()
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        y_t = torch.tanh(x_t_batch)
        log_prob = normal.log_prob(x_t_batch)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

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
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SquashedGaussian, self).to(device)


if __name__ == '__main__':
    from IPython import embed; embed()

