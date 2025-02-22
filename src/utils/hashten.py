import hashlib
import torch

def hashten(tensor: torch.Tensor) -> str:
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()
