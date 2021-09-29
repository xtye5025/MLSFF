import os
import warnings
from os.path import join

import torch
import numpy as np
import random


def create_file_dirs(file_path):
    os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)

def set_seed(seed):
    print(f"Using seed: {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device(gpu=None):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
