from torch import Tensor, nn
from torchvision.models import resnet18


class ResNet18Top(nn.Module):
    """
    The top part of the ResNet architecture, the last two basic blocks are
    replaced by the visual transformer.
    """

    def __init__(self) -> None:
        super().__init__()
        full_resnet18 = resnet18()
        self.features = nn.Sequential(*list(full_resnet18.children())[:-3])
        self.n_channel = 256
        for p in self.features.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_batch: Tensor) -> Tensor:
        """Forwards pass throught the top part of the ResNet18 network

        Args:
            x_in (Tensor): Batch of images, of shape (batch_size,
            n_color_channels, width, height)

        Returns:
            Tensor: Features extracted by the ResNet18 network, of shape
                    (batch_size, n_features, feature_width, feature_height)
        """
        return self.features(image_batch)


if __name__ == "__main__":
    resnet18_top = ResNet18Top()
    for p in resnet18_top.parameters():
        print(p.shape)
