import transformers
import random
import torch
import numpy

def seed_all(seed: int):
    transformers.set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
