import torch
from torch import nn


class VisualTransformerClassifier(nn.Module):
    """A classifier constructed by replacing the last two basic blocks of
    ResNet18 by a visual transformer"""

    def __init__(
        self,
        feature_extractor: nn.Module,
        visual_transformer: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.visual_transformer = visual_transformer
        self.classifier = classifier

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            image_batch (torch.Tensor): Batch of input images, of shape
            (batch_size, n_color_channel, image_width, image_height)

        Returns:
            torch.Tensor: Predictions for each class, of shape
            (batch_size, n_classes)
        """
        return self.classifier(
            self.visual_transformer(self.feature_extractor(image_batch))
        )
