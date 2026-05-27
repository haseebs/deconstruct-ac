import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_models.student import StudentT


EPSILON = 1e-6


class SquashedStudent(nn.Module):
    def __init__(
        self,
        num_actions,
        mean_init,
        shape_init,
        df_init,
        action_min,
        action_max,
        fixed_df=False,
        clip_stddev=1000,
        log_prob_reduction: str | None = None,
    ):
        super(SquashedStudent, self).__init__()

        self.num_actions = num_actions
        self.mean = nn.Parameter(torch.FloatTensor([mean_init] * num_actions), requires_grad=True)
        self.log_shape = nn.Parameter(torch.FloatTensor([shape_init] * num_actions), requires_grad=True)

        if fixed_df:
            self.register_buffer("df_param", torch.FloatTensor([df_init] * num_actions))
        else:
            self.df_param = nn.Parameter(torch.FloatTensor([df_init] * num_actions), requires_grad=True)

        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        if type(action_max) == list:
            assert np.sum(action_max) / len(action_max) == action_max[0]
            assert np.sum(action_min) / len(action_min) == action_min[0]

        self.action_scale = torch.tensor((action_max[0] - action_min[0]) / 2.0)
        self.action_bias = torch.tensor((action_max[0] + action_min[0]) / 2.0)

        self.dof_bias = torch.tensor(1.0001)
        self.log_prob_reduction = log_prob_reduction or "sum"
        if self.log_prob_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid log_prob_reduction={self.log_prob_reduction!r}; expected 'mean' or 'sum'.")

    def _reduce_joint_log_prob(self, joint_log_prob: torch.Tensor) -> torch.Tensor:
        if self.log_prob_reduction == "mean":
            return joint_log_prob / self.num_actions
        return joint_log_prob

    def get_params(self):
        return self.mean.reshape(-1).tolist(), self.log_shape.reshape(-1).tolist(), self.df_param.reshape(-1).tolist()

    def forward(self):
        mean = self.mean
        log_shape = self.log_shape
        if self.clip_stddev:
            log_shape = torch.clamp(log_shape, min=-self.clip_std_threshold, max=self.clip_std_threshold)
        shape = log_shape.exp()
        dfx = F.softplus(self.df_param) + self.dof_bias
        return mean, shape, dfx

    def rsample(self, num_samples=1):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)

        x_t = student.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        joint_log_prob = student.log_prob(x_t).sum(dim=-1)
        joint_log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=-1)
        log_prob = self._reduce_joint_log_prob(joint_log_prob).view(-1)

        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean, None

    def sample(self, num_samples=1):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)

        x_t = student.sample((num_samples,)).detach()
        if num_samples == 1:
            x_t = x_t.squeeze(0)

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        joint_log_prob = student.log_prob(x_t).sum(dim=-1)
        joint_log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=-1)
        log_prob = self._reduce_joint_log_prob(joint_log_prob).view(-1)

        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean, None

    def log_prob(self, action_batch):
        mean, shape, dfx = self.forward()
        student = StudentT(dfx, mean, shape)

        y = (action_batch - self.action_bias) / self.action_scale
        y = torch.clamp(y, -1.0 + EPSILON, 1.0 - EPSILON)
        x_t = torch.atanh(y)

        joint_log_prob = student.log_prob(x_t).sum(dim=-1)
        joint_log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPSILON).sum(dim=-1)
        return self._reduce_joint_log_prob(joint_log_prob).view(-1)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.dof_bias = self.dof_bias.to(device)
        return super(SquashedStudent, self).to(device)
