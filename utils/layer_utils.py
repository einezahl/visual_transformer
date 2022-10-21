import copy
from torch import nn


def clones(module: nn.Module, N: int):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
