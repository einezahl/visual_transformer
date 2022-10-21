import torch
from torch import nn


class ResNet18Classifier(nn.Module):
    """
    Last few layers of the ResNet architecture, for the classification of the
    transformed features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(
            in_features=in_features, out_features=out_features
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Given the feature map generated by the CNN and transformed by the
        visual transformer perform classification

        Args:
            feature_map (torch.Tensor): Output of the visual transformer,
            feature map with self attetion, of shape (batch_size, n_channel,
            feature_width, feature_height)

        Returns:
            torch.Tensor: _description_
        """
        x = self.avgpool(feature_map)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fully_connected(x)
        return x


if __name__ == "__main__":
    classifier = ResNet18Classifier(256, 10)
    for p in classifier.parameters():
        print(p.shape)