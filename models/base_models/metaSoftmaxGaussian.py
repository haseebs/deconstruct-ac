import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import copy


EPSILON = 1e-6


class MetaSoftmaxGaussian(nn.Module):
    """
        
    We discretize the action space into bins along each action dimension.
    Softmax outputs an action as a tuple of integer like [8, 5, 2], to be fed into NN. 
    
    We make meta policy by first deciding which action interval to pick,
    and then sample from the Gaussian residing in this interval.
    Updates are done by masking out other intervals.
    """
    def __init__(self, num_actions, action_min, action_max, num_bins=10):
        super(MetaSoftmaxGaussian, self).__init__()

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
        mu = np.arange(self.action_min[0], self.action_max[0], self.action_unit_interval).reshape(-1, 1) + 1.
        sigma = np.ones((num_bins, 1))
        params = np.concatenate((mu, sigma), axis=1)
        """make action_dims copies of mu and sigma for Gaussian policies"""
        self.gauss_params = nn.Parameter(torch.FloatTensor(params).repeat(num_actions, 1, 1), requires_grad=True)
        self.bounding_boxes = torch.FloatTensor(np.arange(self.action_min[0], self.action_max[0], self.action_unit_interval))[None, :]

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
        interval_idx is used to retrieve Gaussian and bounding box
        """
        param_idx = interval_idx[:, :, None].repeat(1, 1, 2)
        box_idx = interval_idx
        selected_gaussians = torch.concat([gauss_params.gather(dim=1, index=param_idx[i, :, :][None, :, :]) for i in range(num_samples)])
        selected_boxes_lower = torch.concat([self.bounding_boxes.gather(dim=1, index=box_idx[i, :][None, :]) for i in range(num_samples)])
        selected_boxes_upper = selected_boxes_lower + self.action_unit_interval
        
        mean = selected_gaussians[:, :, 0]
        std = selected_gaussians[:, :, 1].exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)
                    
        """
        note that mean shape is (num_samples, num_action)
        so calling normal.sample() returns (num_samples, num_action)
        """
        actions = normal.sample()
        
        """
        IMPORTANT!
        Note we are simply truncating the samples by the bounding box.
        This is different from enforcing the distribution stays within the range
        e.g, let 95% of Gaussian probability mass falls within the range        
        """
        actions = torch.clamp(actions, min=selected_boxes_lower, max=selected_boxes_upper)
        gauss_logprob = normal.log_prob(actions)
        
        if num_samples == 1:
            actions = actions.squeeze(0)
        
        softmax_logprob = F.log_softmax(logits, dim=-1)
        
        """        
        computing log policy comprises two stages:
        1. softmax interval log-prob
        2. Gaussian log-prob
        these two terms should be added (multiplied/chained in log)
        
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
                log_prob_out[sample_dim] += softmax_logprob[action_dim, interval_idx[sample_dim, action_dim].long()]
        
        """add softmax logprob and gaussian logprob"""
        log_prob_out += gauss_logprob
        
        return actions.float(), log_prob_out, logits.argmax(dim=-1), None
    

    def rsample(self, num_samples=1):
        logits, gauss_params = self.forward()

        probs = F.softmax(logits, dim=-1)
        rep_probs = probs.repeat(num_samples, 1, 1)
        policy = torch.distributions.Categorical(probs=rep_probs)
        interval_idx = policy.sample()
        
        param_idx = interval_idx[:, :, None].repeat(1, 1, 2)
        box_idx = interval_idx
        selected_gaussians = torch.concat([gauss_params.gather(dim=1, index=param_idx[i, :, :][None, :, :]) for i in range(num_samples)])
        selected_boxes_lower = torch.concat([self.bounding_boxes.gather(dim=1, index=box_idx[i, :][None, :]) for i in range(num_samples)])
        selected_boxes_upper = selected_boxes_lower + self.action_unit_interval
        
        mean = selected_gaussians[:, :, 0]
        std = selected_gaussians[:, :, 1].exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)
            
        actions = normal.rsample()
        actions = torch.clamp(actions, min=selected_boxes_lower, max=selected_boxes_upper)
        gauss_logprob = normal.log_prob(actions)      
        if num_samples == 1:
            actions = actions.squeeze(0)        
        softmax_logprob = F.log_softmax(logits, dim=-1)
        log_prob_out = torch.zeros((num_samples, ))
        for sample_dim in range(num_samples):
            for action_dim in range(self.num_actions):
                log_prob_out[sample_dim] += softmax_logprob[action_dim, interval_idx[sample_dim, action_dim].long()]
        
        log_prob_out += gauss_logprob
        return actions.float(), log_prob_out, logits.argmax(dim=-1), None

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
        return super(MetaSoftmaxGaussian, self).to(device)
