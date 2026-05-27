import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# Global variables
EPSILON = 1e-6


class SquashedGaussian(nn.Module):
    """
    Class SquashedGaussian implements a policy following a squashed
    Gaussian distribution in each state, parameterized by an MLP.
    """
    def __init__(self, num_actions, mean_init, std_init,
                 action_min, action_max, clip_stddev=1000, log_prob_reduction: str | None = None):
        """
        Constructor

        Parameters
        ----------
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        """
        super(SquashedGaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.mean = nn.Parameter(torch.FloatTensor([mean_init]*num_actions), requires_grad=True)
        self.log_std = nn.Parameter(torch.FloatTensor([std_init]*num_actions), requires_grad=True)

        if type(action_max) == list:
            assert np.sum(action_max)/len(action_max) == action_max[0]
            assert np.sum(action_min)/len(action_min) == action_min[0]

        # action rescaling
        self.action_scale = (action_max[0] - action_min[0]) / 2.
        self.action_bias = (action_max[0] + action_min[0]) / 2.
        self.log_prob_reduction = log_prob_reduction or "mean"
        if self.log_prob_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid log_prob_reduction={self.log_prob_reduction!r}; expected 'mean' or 'sum'.")

    def _reduce_joint_log_prob(self, joint_log_prob: torch.Tensor) -> torch.Tensor:
        if self.log_prob_reduction == "mean":
            return joint_log_prob / self.num_actions
        return joint_log_prob

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.log_std.reshape(-1).tolist(), [0]


    def forward(self):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

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

    def sample(self, num_samples=1):
        """
        Samples the policy for an action in the argument


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

        # note i think this wont work with num_actions=1
        log_prob = self._reduce_joint_log_prob(log_prob).view(-1)
        # if self.num_actions > 1:
        #     log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, None

    def rsample(self, num_samples=1):
        """
        Samples the policy for an action in the argument state using
        the reparameterization trick

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
        # if self.num_actions > 1:
        #     log_prob = log_prob.unsqueeze(-1)
        log_prob = self._reduce_joint_log_prob(log_prob).view(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, None

    def log_prob(self, action_batch):
        """
        Calculates the log probability of taking the action generated
        from x_t, where x_t is returned from sample or rsample. The
        log probability is returned for each action dimension separately.
        """
        mean, log_std = self.forward()
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # `action_batch` is in environment units; invert the affine transform back to tanh-space.
        y = (action_batch - self.action_bias) / self.action_scale
        y = torch.clamp(y, -1.0 + EPSILON, 1.0 - EPSILON)
        out = torch.atanh(y)

        log_prob = normal.log_prob(out)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        log_prob = self._reduce_joint_log_prob(log_prob).view(-1)

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
