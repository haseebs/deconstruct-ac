import torch
import numpy as np
from scipy.special import gamma, digamma

def KL_Gaussian(policy_1, policy_2):
    """
    KL(p_1 || p_2) for two Gaussian distributions
    
    input: 
    rescaled_mean_1, rescaled_mean_2: rescaled mean in the range [action_min, action_max]
    log_sigma_1, log_sigma_2: log std deviation
    
    we have num_actions dimensions so we compute it wholesale and then average over action dim
    """
    rescaled_mean_1, logsigma_1 = policy_1.mean.detach(), torch.exp(policy_1.log_std.detach())
    rescaled_mean_2, logsigma_2 = policy_2.mean.detach(), torch.exp(policy_2.log_std.detach())
    kl_div = (logsigma_2 - logsigma_1 + (torch.exp(logsigma_1) ** 2 + (rescaled_mean_1 - rescaled_mean_2) ** 2) / (2 * torch.exp(logsigma_2) ** 2) - 0.5).mean()
    return kl_div


def KL_Beta(policy_1, policy_2):
    """
    KL(p_1 || p_2) for two Beta distributions
    
    we need to compute special functions by scipy so we assume the input are detached numpy arrays rather than torch
    gamma functions can run into numerical issues so we need to do clampping

    Beta distribution is defined as B(mean, shape), where:
    mean_1, mean_2: these are typically denoted alpha in literature, they control the mean
    shape_1, shape_2: these are typically denoted beta in literature, they control the shape
    alpha > 0, beta > 0
    
    we have num_actions dimensions so we compute it wholesale and then average over action dim
    """
    # mean_1, shape_1 = policy_1.mean.detach(), policy_1.shape.detach()
    # mean_2, shape_2 = policy_2.mean.detach(), policy_2.shape.detach()
    mean_1, shape_1 = [param.detach() for param in policy_1.forward()]
    mean_2, shape_2 = [param.detach() for param in policy_2.forward()]
    upperbound = 1e128
    gmms1 = np.minimum(gamma(mean_1 + shape_1), upperbound)
    gmms2 = np.minimum(gamma(mean_2 + shape_2), upperbound)
    gms1 = np.minimum(gamma(shape_1), upperbound)
    gms2 = np.minimum(gamma(shape_2), upperbound)
    gmm1 = np.minimum(gamma(mean_1), upperbound)
    gmm2 = np.minimum(gamma(mean_2), upperbound)
    logbeta = np.log(gmm1) + np.log(gms1) - np.log(gmms1) - np.log(gmm2) - np.log(gms2) + np.log(gmms2)

    dgm1 = digamma(mean_1) - digamma(mean_1 + shape_1)
    dgm2 = digamma(shape_1) - digamma(mean_1 + shape_1)

    kl_div = (mean_1 - mean_2) * torch.Tensor(dgm1) + (shape_1 - shape_2)  * torch.Tensor(dgm2) - torch.Tensor(logbeta)
    kl_div = kl_div.mean()
    if torch.isnan(kl_div):
        print(f"mean 1 {mean_1}, shape 1 {shape_1}, mean 2 {mean_2}, shape 2 {shape_2}")
        raise RuntimeError("NaN detected in Beta KL computation!")

    return kl_div


def KL_Softmax(policy_1, policy_2):    
    """
    KL(p_1 || p_2) for two Softmax distributions
    
    input: 
    logits_1, logits_2 from two softmax
    
    this computation is exact, no empirical average over the action dim,
    since we have discretized the action space into bins already
    """
    logits1, logits2 = policy_1.policy.logits.detach(), policy_1.policy_old.logits.detach()
    p1 = torch.nn.functional.softmax(logits1, dim=1)
    kl_div = torch.sum(p1 * (logits1 - logits2 - torch.logsumexp(logits1, dim=1).unsqueeze(-1) + torch.logsumexp(logits2, dim=1).unsqueeze(-1)))

    return kl_div
