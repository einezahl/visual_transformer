import torch
import torch.nn.functional as F
from torch import nn


class FilterTokenLayer(nn.Module):
    """
    Single layer of the filter tokenizer architecture.
    It is used for the first layer of the tokenizer architecture.
    """

    def __init__(self, n_token: int, n_channel: int) -> None:
        super().__init__()
        self.w = nn.Linear(n_channel, n_token, False)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x):
        t = F.softmax(self.w(x.transpose(1, 2)), dim=1).transpose(1, 2)
        t = torch.matmul(t, x.transpose(1, 2))
        return t


class RecurrentTokenLayer(nn.Module):
    """
    A single layer of the recurrent tokenizer architecture
    """

    def __init__(self, n_channel: int) -> None:
        super().__init__()
        self.w = nn.Linear(n_channel, n_channel, False)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, t_in):
        w = self.w(t_in)
        t = F.softmax(torch.matmul(w, x), dim=1)
        t = torch.matmul(t, x.transpose(1, 2))
        return t
