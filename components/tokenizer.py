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
            feature_map (torch.Tensor): Output of the feature extractor, has
            the shape (batch_size, n_channel, feature_width, feature_height)

        Returns:
            torch.Tensor: Tokenized feature map, has the shape (batch_size,
            n_token, n_channel)
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

    def forward(
        self, feature_map: torch.Tensor, visual_token_in: torch.Tensor
    ) -> torch.Tensor:
        """Forwards pass through the recurrent token layer, the weights for
        the tokenization are dependent on the weights of the previous layer

        Args:
            feature_map (torch.Tensor): Output of the feature extractor, has
            the shape (batch_size, n_channel, feature_width*feature_height)
            visual_token_in (torch.Tensor): visual_token of the previous layer,
            has the shape (batch_size, n_token, n_channel)

        Returns:
            _type_: _description_
        """
        w_r = self.w_t_r(visual_token_in)
        t_out = F.softmax(torch.matmul(w_r, feature_map), dim=1)
        t_out = torch.matmul(t_out, feature_map.transpose(1, 2))
        return t_out


class Tokenizer(nn.Module):
    """
    Combines the different token layer into a full tokenizer. As the
    RecurrentTokenLayer requires the feature map of the previous layer, the
    first layer is a FilterTokenLayer
    """

    def __init__(self, n_token_layer, n_token, n_channel) -> None:
        super().__init__()
        self.first_layer = FilterTokenLayer(n_token, n_channel)
        self.recurrent_layer = [
            RecurrentTokenLayer(n_channel) for _ in range(n_token_layer - 1)
        ]

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Forwards the feature map through recurrent tokenizer layers, as no
        visual tokens are present in the first layer, the first layer is a
        filter tokenizer

        Args:
            feature_map (torch.Tensor): Feature map tensor of the feature
            extractor, has the dimensions (batch_size, n_channel,
            feature_width*feature_height)

        Returns:
            torch.Tensor: Visual token
        """
        visual_token = self.first_layer(feature_map)
        for layer in self.recurrent_layer:
            visual_token = layer(feature_map, visual_token)
        return visual_token

    def to_device(self, device: torch.device) -> None:
        """Moves the tokenizer to the specified device

        Args:
            device (torch.device): Device to move the tokenizer to
        """
        self.first_layer.to(device)
        for r_layer in self.recurrent_layer:
            r_layer.to(device)
