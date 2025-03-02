import torch

def basic(model:torch.nn.Module) -> torch.nn.Module:
    return torch.compile(model)

def dummy(model:torch.nn.Module) -> torch.nn.Module:
    return model
