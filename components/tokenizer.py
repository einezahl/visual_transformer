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
        self.w_a = nn.Linear(n_channel, n_token, False)
        nn.init.xavier_uniform_(self.w_a.weight)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Forwards pass through the tokinzer layer

        Args:
            feature_map (torch.Tensor): Output of the feature extractor, has the shape (batch_size,
            n_channel, feature_width, feature_height)

        Returns:
            torch.Tensor: Tokenized feature map, has the shape (batch_size, n_token, n_channel)
        """
        visual_token = F.softmax(
            self.w_a(feature_map.transpose(1, 2)), dim=1
        ).transpose(1, 2)
        visual_token = torch.matmul(visual_token, feature_map.transpose(1, 2))
        return visual_token


class RecurrentTokenLayer(nn.Module):
    """
    A single layer of the recurrent tokenizer architecture
    """

    def __init__(self, n_channel: int) -> None:
        super().__init__()
        self.w_t_r = nn.Linear(n_channel, n_channel, False)
        nn.init.xavier_uniform_(self.w_t_r.weight)

    def forward(self, feature_map: torch.Tensor, visual_token_in: torch.Tensor):
        """Forwards pass through the recurrent token layer, the weights for the tokenization are
        dependent on the weights of the previous layer

        Args:
            feature_map (torch.Tensor): Output of the feature extractor, has the shape (batch_size,
            n_channel, feature_width, feature_height)
            visual_token_in (torch.Tensor): visual_token of the previous layer, has the shape
            (batch_size, n_token, n_channel)

        Returns:
            _type_: _description_
        """
        w_r = self.w_t_r(visual_token_in)
        t_out = F.softmax(torch.matmul(w_r, feature_map), dim=1)
        t_out = torch.matmul(t_out, feature_map.transpose(1, 2))
        return t_out
