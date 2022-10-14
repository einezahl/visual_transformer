import torch
from torch import nn
from torch.nn import functional


class Projector(nn.Module):
    """
    Projector module, seems like another transformer to me
    """

    def __init__(self, n_channel) -> None:
        super().__init__()
        self.w_query = nn.Linear(n_channel, n_channel)
        self.w_key = nn.Linear(n_channel, n_channel)

    def forward(
        self, feature_map: torch.Tensor, visual_token: torch.Tensor
    ) -> torch.Tensor:
        """Fuse the transformers output with the feature map

        Args:
            feature_map (torch.Tensor): Output of the feature extractor, of shape
            (batch_size, n_channel, feature_width*feature_height)
            visual_token (torch.Tensor): Output of the transformer, of shape
            (batch_size, n_token, n_channel)

        Returns:
            torch.Tensor: Fusion of the transformers output and the feature map,
            of shape (batch_size, n_channel, feature_width*feature_height)
        """
        query = self.w_query(feature_map.transpose(1, 2))
        key = self.w_key(visual_token)
        query_key = functional.softmax(torch.matmul(query, key.transpose(1, 2)), dim=2)
        feature_map_attention = feature_map + torch.matmul(
            query_key, visual_token
        ).transpose(1, 2)
        return feature_map_attention
