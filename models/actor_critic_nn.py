import time
import random
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

from .base_models.mlp import FCNNOldRepo
from .base_models.softmax import SoftmaxPolicyWrapper

class Critic:
    def __init__(self, n_actions: int, hidden_units: int, hidden_layers: int, lr: float, lr_replay: float) -> None:
        self.value_net = FCNNOldRepo(input_size=n_actions, hidden_units=hidden_units)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = Adam(self.value_net.parameters(), lr=lr)
        self.optimizer_replay = Adam(self.value_net.parameters(), lr=lr_replay)

    def get_qvalues(self, actions: list[list[float]]) -> torch.Tensor:
        """
        actions: list of actions [n_actions, action_dim]
        """
        if torch.is_tensor(actions):
            actions_tensor = actions.to(dtype=torch.float32)
        else:
            actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
        return self.value_net(actions_tensor)


class BaseActor:
    def __init__(self, policy: nn.Module, lr: float) -> None:
        """
        I added policy_old and its optimizer here for greedyac
        policy/optimizer: actor
        """
        self.policy = policy
        self.optimizer = Adam(self.policy.parameters(), lr=lr)


class Actor(BaseActor):
    """
    TODO from haseeb: why does the base class exist?
    TODO: this actor wrapper may need to be extended more to cover policy usages
    I built this actor especially considering the special KL policy form: 

    \pi_t(a|s) \propto \pi_{t-1}(a|s) * \exp(Q_{t-1}(s,a))

    where we must simultaneously have access to 
        1) sampling from the policy, with the above form, that is having access to both \pi_{t-1}(a|s) and \exp(Q_{t-1}(s,a))
        2) log prob, this again requires access to both
        3) update only the current policy 
    """
    def __init__(self, policy: nn.Module, lr: float) -> None:
        super(Actor, self).__init__(policy, lr)
        if isinstance(policy, SoftmaxPolicyWrapper):
            self.policy = policy
            self.optimizer = Adam(self.policy.policy.parameters(), lr=lr)


if __name__ == "__main__":
    from base_models.gaussian import Gaussian
    
    # g = Gaussian(3,0,1,[0,0,0],[20,20,20])
    # a = Actor(g, 0.1)

    # for _ in range(10):
    #     print("policy: \t ", list(a.policy.parameters()))
    #     print("polic_old: \t", list(a.policy_old.parameters()))
    #     e = a.policy.log_prob(torch.FloatTensor([1,1,1]))
    #     e.backward()
    #     a.optimizer.step()
    #     print()

    """test the softmax policy wrapper here"""
    g = SoftmaxPolicyWrapper(num_actions=3, kl=True)
    a = Actor(policy=g, lr=0.1)
    for _ in range(10):
        print("policy: \t ", list(a.policy.policy.parameters()))
        print("polic_old: \t", list(a.policy.policy_old.parameters()))
        e = a.policy.log_prob(torch.FloatTensor([[1,1,1], [2,2,2]])).mean()
        a.policy_backup_hook()
        a.optimizer.zero_grad()
        e.backward()
        a.optimizer.step()
        a.policy.sample()
        print()
