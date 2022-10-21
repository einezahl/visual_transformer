import torch
from torch import nn

from components.feature_extractor import ResNet18Top
from components.visual_transformer import VisualTransformer
from components.classifier import ResNet18Classifier
from components.projector import Projector
from components.tokenizer import Tokenizer
from components.transformer import Transformer


class VisualTransformerClassifier(nn.Module):
    """A classifier constructed by replacing the last two basic blocks of
    ResNet18 by a visual transformer"""

    def __init__(
        self,
        n_token_layer: int,
        n_token: int,
        n_hidden: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.resnet18_top = ResNet18Top()
        self.n_channel = self.resnet18_top.n_channel
        self.tokenizer = Tokenizer(
            n_token_layer=n_token_layer, n_token=n_token, n_channel=self.n_channel
        )
        self.transformer = Transformer(n_channel=self.n_channel, n_hidden=n_hidden)
        self.projector = Projector(n_channel=self.n_channel)
        self.visual_transformer = VisualTransformer(
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            projector=self.projector,
        )
        self.classifier = ResNet18Classifier(
            in_features=self.n_channel, out_features=n_classes
        )

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            image_batch (torch.Tensor): Batch of input images, of shape
            (batch_size, n_color_channel, image_width, image_height)

        Returns:
            torch.Tensor: Predictions for each class, of shape
            (batch_size, n_classes)
        """
        return self.classifier(self.visual_transformer(self.resnet18_top(image_batch)))

    def to(self, device: torch.device) -> None:
        """_summary_

        Args:
            device (torch.device): _description_
        """
        super().to(device)
        self.resnet18_top.to(device)
        self.tokenizer.to(device)
        self.transformer.to(device)
        self.projector.to(device)
        self.visual_transformer.to(device)
        self.classifier.to(device)
