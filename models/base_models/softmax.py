import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy


EPSILON = 1e-6


class Softmax(nn.Module):
    """
    We discretize the action space into bins along each action dimension.
    Softmax outputs an action as a tuple of integer like [8, 5, 2], to be fed into NN. 
    Previously, before feeding into the NN we have RBF features, but this is removed here.

    Note: we can assign different number of bins to each action dimension, by using padding
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    """
    def __init__(self, num_actions, action_min, action_max, num_bins=10):
        super(Softmax, self).__init__()

        self.num_actions = num_actions
        """our interface is uniform, so we discretize the action space here"""
        self.num_bins = num_bins
        self.action_max, self.action_min = np.array(action_max), np.array(action_min)
        self.action_unit_interval = (self.action_max - self.action_min) / num_bins
        self.logits = nn.Parameter(torch.FloatTensor(np.random.rand(num_actions, num_bins)*0.1), requires_grad=True)

    def get_params(self):
        return self.logits.reshape(-1).tolist(), [0], [0]

    def forward(self):
        return self.logits

    def sample(self, num_samples=1):

        logits = self.forward()
        
        """
        logits is of size (action_dim, num_bins)
        probs is of size (batch_size, action_dims, num_bins), 
        actions_bins is of size (batch_size, action_dims)
        see https://github.com/pytorch/pytorch/issues/43250
        """
        probs = F.softmax(logits, dim=-1)
        rep_probs = probs.repeat(num_samples, 1, 1)
        policy = torch.distributions.Categorical(probs=rep_probs)
        original_action_bins = policy.sample()
        action_bins = original_action_bins * self.action_unit_interval
        if num_samples == 1:
            action_bins = action_bins.squeeze(0)
        log_prob = F.log_softmax(logits, dim=-1)
        
        """
        computing log policy:
        each action consists of 3 action dimensions, 
        so for an input action batch (num_sample, num_actions), 
        the log-policy return should be (num_samples, )
        
        for softmax, we sample actions by specifying bins along each action dim independently,
        so log-policy should be log-policy-dim1 + log-policy-dim2 + ...
        """
        log_prob_out = torch.zeros((num_samples, ))
        # log_prob_out = torch.zeros((num_samples, self.num_actions))
        for sample_dim in range(num_samples):
            for action_dim in range(self.num_actions):
                log_prob_out[sample_dim] += log_prob[action_dim, original_action_bins[sample_dim, action_dim].long()]

        return action_bins.float(), log_prob_out, logits.argmax(dim=-1), None
    
    def rsample(self, num_samples=1):
        return self.sample(num_samples=num_samples)

    def all_log_prob(self):
        logits = self.forward()
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def log_prob(self, actions):
        """
        Returns the log probability of taking actions in states.
        """
        logits = self.forward()
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_out = torch.zeros((actions.shape[0], ))
        original_actions = actions / self.action_unit_interval
        for sample_dim in range(actions.shape[0]):
            for action_dim in range(self.num_actions):
                log_probs_out[sample_dim] += log_probs[action_dim, original_actions[sample_dim, action_dim].long()]

        return log_probs_out

    def entropy(self):
        logits = self.forward()
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1))).sum()
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
        return super(Softmax, self).to(device)



class SoftmaxPolicyWrapper(Softmax):
    """
    a KL-regularized softmax policy, \pi_t \propto \pi_{t-1}\exp(Q_{t-1}). 
    A even more general version with both KL and can be seen at Eq.(5) of https://arxiv.org/pdf/2003.14089.pdf
    """
    def __init__(self, num_actions, action_min, action_max, kl_coef=1.0, num_bins=10, kl=False):
        super(SoftmaxPolicyWrapper, self).__init__(num_actions, action_min, action_max, num_bins)
        self.kl_coef = kl_coef
        self.kl = kl
        self.policy = Softmax(num_actions=num_actions, action_min=action_min, action_max=action_max)
        self.policy_old = copy.deepcopy(self.policy)
        for param in self.policy_old.parameters():
            param.requires_grad = False
        del self.logits

    def rsample(self, num_samples=1):
        return self.sample(num_samples=num_samples)
    
    def forward(self):
        return self.policy.logits

    def sample(self, num_samples=1):

        if not self.kl:
            return self.policy.sample(num_samples)
        else:
            logits = self.forward()
            
            prev_policy = F.softmax(self.policy_old.logits, dim=-1)
            unnormalized_policy_now = torch.exp(self.kl_coef * logits)
            probs = prev_policy * unnormalized_policy_now
            probs /= probs.sum(dim=1).unsqueeze(-1)

            rep_probs = probs.repeat(num_samples, 1, 1)
            policy = torch.distributions.Categorical(probs=rep_probs)
            action_bins = policy.sample()
            log_prob = torch.log(probs)
            log_prob_out = torch.zeros((num_samples, self.num_actions))
            for sample_dim in range(num_samples):
                log_prob_out[sample_dim, :] = torch.gather(log_prob, dim=-1, index=action_bins[sample_dim, :].unsqueeze(-1).long()).squeeze()

            if num_samples == 1:
                action_bins = action_bins.squeeze(0)
                log_prob_out = log_prob_out.squeeze(0)

            return action_bins.float(), log_prob_out, logits.argmax(dim=-1), None
    
    def log_prob(self, actions):
        if not self.kl:
            return self.policy.log_prob(actions)
        else:
            logits = self.forward()
            prev_policy = F.softmax(self.policy_old.logits, dim=-1)
            unnormalized_policy_now = torch.exp(self.kl_coef * logits)
            probs = prev_policy * unnormalized_policy_now
            probs /= probs.sum(dim=1).unsqueeze(-1)
            log_probs = torch.log(probs)
            log_probs_out = torch.zeros_like(actions, dtype=torch.float32)
            for sample_dim in range(actions.shape[0]):
                log_probs_out[sample_dim, :] = torch.gather(log_probs, dim=-1, index=actions[sample_dim, :].unsqueeze(-1).long()).squeeze()

        return log_probs_out
