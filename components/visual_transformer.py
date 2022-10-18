import torch
from torch import nn

from components.projector import Projector
from components.tokenizer import Tokenizer
from components.transformer import Transformer


class VisualTransformer(nn.Module):
    """
    Visual encoder as described in the paper "Visual Transformers:
    Token-based Image Representation and Processing for Computer Vision"
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        transformer: Transformer,
        projector: Projector
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.projector = projector

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Applies self attention to feature map

        Args:
            feature_map (torch.Tensor): Flattened output of the preprocessor,
            typically a convolutional neural network without the last layers

        Returns:
            torch.Tensor: _description_
        """
        original_shape = feature_map.shape
        preprocessed = torch.flatten(feature_map, start_dim=2, end_dim=3)
        visual_transformer = self.projector(
            preprocessed, self.transformer(self.tokenizer(preprocessed))
        )
        return visual_transformer.reshape(original_shape)
