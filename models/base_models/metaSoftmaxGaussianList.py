import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy
from operator import itemgetter

EPSILON = 1e-6


class MetaSoftmaxGaussianList(nn.Module):
    """
        
    We discretize the action space into bins along each action dimension.
    Softmax outputs an action as a tuple of integer like [8, 5, 2], to be fed into NN. 
    
    We make meta policy by first deciding which action interval to pick,
    and then sample from the Gaussian residing in this interval.
    Updates are done by masking out other intervals.
    """
    def __init__(self, num_actions, action_min, action_max, num_bins=10):
        super(MetaSoftmaxGaussianList, self).__init__()

        self.num_actions = num_actions
        """our interface is uniform, so we discretize the action space here"""
        self.num_bins = num_bins
        self.action_max, self.action_min = np.array(action_max), np.array(action_min)
        self.action_unit_interval = (self.action_max[0] - self.action_min[0]) / num_bins
        
        """
        logits are used to compute softmax probabilities for intervals (like a classifictaion task)
        we choose sample an interval, and then use its index to find the corresponding Gaussian
        """
        self.logits = nn.Parameter(torch.FloatTensor(np.random.rand(num_actions, num_bins)*0.1), requires_grad=True)
      
        """make action_dims copies of mu and sigma for Gaussian policies"""
        self.gauss_params = {}
        # for i in range(num_actions):
        #     for j in range(num_bins):
        #         self.gauss_params[f"action {i}, bin {j}"] = nn.Parameter(torch.FloatTensor([2*j+1, 1]), requires_grad=True)
        for i in range(num_actions):
                self.gauss_params[f"action {i}"] = [nn.Parameter(torch.FloatTensor([j, 1]), requires_grad=True) for j in range(int(self.action_min[0]), int(self.action_max[0]), int(self.action_unit_interval))]
        self.box_lower = torch.FloatTensor(np.arange(self.action_min[0], self.action_max[0], self.action_unit_interval))
        self.box_upper = self.box_lower + self.action_unit_interval

    def forward(self):
        """
        need to make sure whether we need to return parameters from both levels
        """
        return self.logits, self.gauss_params

    def sample(self, num_samples=1):

        logits, gauss_params = self.forward()
        
        """
        logits is of size (action_dims, num_bins)
        probs is of size (batch_size, action_dims, num_bins), 
        actions_bins is of size (batch_size, action_dims)
        see https://github.com/pytorch/pytorch/issues/43250
        """
        probs = F.softmax(logits, dim=-1)
        rep_probs = probs.repeat(num_samples, 1, 1)
        policy = torch.distributions.Categorical(probs=rep_probs)
        interval_idx = policy.sample()
        
        """
        work with each dimension of action independently
        """
        softmax_all_logprob = F.log_softmax(logits, dim=-1)        
        all_dim_actions = torch.zeros(num_samples, self.num_actions)
        log_prob_out = torch.zeros((num_samples, self.num_actions))
        for i in range(self.num_actions):
            selected_pairs = itemgetter(*interval_idx[:, i])(gauss_params[f"action {i}"])
            try:
                stacked_pairs = torch.stack(selected_pairs)
            except TypeError:
                stacked_pairs = selected_pairs[None, :]
            normal = Normal(stacked_pairs[:, 0], stacked_pairs[:, 1])

            actions = normal.sample()
            try:
                lowerbound = torch.stack(itemgetter(*interval_idx[:, i])(self.box_lower))
                upperbound = torch.stack(itemgetter(*interval_idx[:, i])(self.box_upper))
            except TypeError:
                lowerbound = itemgetter(*interval_idx[:, i])(self.box_lower)
                upperbound = itemgetter(*interval_idx[:, i])(self.box_upper)
            clipped_actions = torch.clamp(actions, min=lowerbound, max=upperbound)
            
            all_dim_actions[:, i] = clipped_actions
            
            gauss_logprob = normal.log_prob(actions)
            softmax_logprob = itemgetter(interval_idx[:, i])(softmax_all_logprob[i, :])
            
            log_prob_out[:, i] = gauss_logprob + softmax_logprob

        """average over action dim"""                
        log_prob_out = log_prob_out.mean(dim=1)
        
        if num_samples == 1:
            all_dim_actions = all_dim_actions.squeeze(0)
                        
        return all_dim_actions.float(), log_prob_out, logits.argmax(dim=-1), None
    
    

    def rsample(self, num_samples=1):
        logits, gauss_params = self.forward()
    
        probs = F.softmax(logits, dim=-1)
        rep_probs = probs.repeat(num_samples, 1, 1)
        policy = torch.distributions.Categorical(probs=rep_probs)
        interval_idx = policy.sample()
        
        softmax_all_logprob = F.log_softmax(logits, dim=-1)        
        all_dim_actions = torch.zeros(num_samples, self.num_actions)
        log_prob_out = torch.zeros((num_samples, self.num_actions))
        for i in range(self.num_actions):
            selected_pairs = itemgetter(*interval_idx[:, i])(gauss_params[f"action {i}"])
            
            stacked_pairs = torch.stack(selected_pairs)
            normal = Normal(stacked_pairs[:, 0], stacked_pairs[:, 1])

            actions = normal.rsample()
            lowerbound = torch.stack(itemgetter(*interval_idx[:, i])(self.box_lower))
            upperbound = torch.stack(itemgetter(*interval_idx[:, i])(self.box_upper))
            clipped_actions = torch.clamp(actions, min=lowerbound, max=upperbound)
            
            all_dim_actions[:, i] = clipped_actions
            
            gauss_logprob = normal.log_prob(actions)
            softmax_logprob = itemgetter(interval_idx[:, i])(softmax_all_logprob[i, :])
            
            log_prob_out[:, i] = gauss_logprob + softmax_logprob
               
        log_prob_out = log_prob_out.mean(dim=1)
        
        if num_samples == 1:
            all_dim_actions = all_dim_actions.squeeze(0)
                        
        return all_dim_actions.float(), log_prob_out, logits.argmax(dim=-1), None

    def all_log_prob(self):
        logits = self.forward()
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def log_prob(self, actions):
        """
        Returns the log probability of taking actions in states.
        """
        logits, gauss_params = self.forward()
        softmax_all_logprobs = F.log_softmax(logits, dim=-1)
        log_probs_out = torch.zeros((actions.shape[0], self.num_actions))
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
        return super(MetaSoftmaxGaussianList, self).to(device)
