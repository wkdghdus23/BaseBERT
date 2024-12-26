import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): Random seed value to ensure reproducibility.

    Returns:
        None
    """
    # Sets the seed for Python's built-in random number generator
    random.seed(seed)
    # Sets the seed for NumPy's random number generator
    np.random.seed(seed)
    # Sets the seed for PyTorch's random number generator for CPU operations
    torch.manual_seed(seed)

    # Sets the seed for CUDA operations on GPU to ensure reproducibility across multiple GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # This setting optimizes CUDA kernel selection for performance, which may reduce reproducibility across different runs
    torch.backends.cudnn.benchmark = True
