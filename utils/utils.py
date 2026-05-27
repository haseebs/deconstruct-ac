import random

import torch
import numpy as np
import sys
sys.path.insert(0, "../")
from models.base_models.gaussian import Gaussian
from models.base_models.beta import BetaNN
from models.base_models.softmax import SoftmaxPolicyWrapper
from kl_computation.kl_computation import KL_Gaussian, KL_Beta, KL_Softmax

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def kl_selection(policy, *args):
    if isinstance(policy, Gaussian):
        return KL_Gaussian
    elif isinstance(policy, BetaNN):
        return KL_Beta
    elif isinstance(policy, SoftmaxPolicyWrapper):
        return KL_Softmax
    else:
        raise RuntimeError("Unrecognized policy class!")