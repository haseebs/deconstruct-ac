import math
import numpy as np
import torch
import torch.nn as nn
from torch import inf, nan
from torch.distributions import Chi2, constraints, Independent
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, broadcast_all
import torch.nn.functional as F


__all__ = ["StudentT"]


class StudentT(Distribution):
    r"""
    Creates a Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m = self.loc.clone(memory_format=torch.contiguous_format)
        m[self.df <= 1] = nan
        return m

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        m = self.df.clone(memory_format=torch.contiguous_format)
        m[self.df > 2] = (
            self.scale[self.df > 2].pow(2)
            * self.df[self.df > 2]
            / (self.df[self.df > 2] - 2)
        )
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self._chi2 = Chi2(self.df)
        batch_shape = self.df.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(self.df)
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df)
        return self.loc + self.scale * Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z

    def entropy(self):
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )
    
class Student(nn.Module):
    def __init__(self, num_actions, mean_init, shape_init, df_init,
                 action_min, action_max, fixed_df=False, log_prob_reduction: str | None = None):

        super(Student, self).__init__()

        self.num_actions = num_actions

        self.mean = nn.Parameter(torch.FloatTensor([mean_init]*num_actions), requires_grad=True)
        self.log_shape = nn.Parameter(torch.FloatTensor([shape_init]*num_actions), requires_grad=True)

        if fixed_df:
            self.register_buffer('df_param', torch.FloatTensor([df_init]*num_actions))
        else:
            self.df_param = nn.Parameter(torch.FloatTensor([df_init]*num_actions), requires_grad=True)


        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)

        self.log_upper_bound = torch.log(self.action_max - 1e-6)
        self.log_lower_bound = torch.FloatTensor([-14.])
        self.dof_bias = torch.tensor(1.0001)
        self.head_activation_fn = lambda x: torch.nn.functional.softplus(x) + self.dof_bias
        self.log_prob_reduction = log_prob_reduction or "sum"
        if self.log_prob_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid log_prob_reduction={self.log_prob_reduction!r}; expected 'mean' or 'sum'.")

    def _reduce_log_prob(self, log_prob_per_dim: torch.Tensor) -> torch.Tensor:
        if self.log_prob_reduction == "mean":
            return log_prob_per_dim.mean(dim=-1)
        return log_prob_per_dim.sum(dim=-1)

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.log_shape.reshape(-1).tolist(), self.df_param.reshape(-1).tolist()

    def forward(self):
        mean = torch.tanh(self.mean)
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        shape = torch.exp(torch.clamp(self.log_shape, min=self.log_lower_bound, max=self.log_upper_bound))
        dfx = self.head_activation_fn(self.df_param)
        return mean, shape, dfx
    
    def rsample(self, num_samples=1):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)
        actions = student.rsample(sample_shape=(num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = student.log_prob(actions)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        """
        student t is loc-scale as well
        return mean as greedy action
        """
        return actions, self._reduce_log_prob(log_probs), student.mean, None
    
    def sample(self, num_samples=1, deterministic=False):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)
        actions = student.sample(sample_shape=(num_samples,)).detach()
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)

        log_probs = student.log_prob(actions)
        return actions, self._reduce_log_prob(log_probs), student.mean, None
    
    def log_prob(self, actions):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)
        log_probs = student.log_prob(actions)
        return self._reduce_log_prob(log_probs)
    

    def entropy(self):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)
        student = Independent(student, 1)
        return student.entropy()


    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(Student, self).to(device)            
