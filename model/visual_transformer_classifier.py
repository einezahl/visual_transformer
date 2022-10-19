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


def create_model(
    n_token_layer: int, n_token: int, n_channel: int, n_hidden: int, n_classes: int
) -> VisualTransformerClassifier:
    """Creates a visual transformer classifier
    Args:
        n_token_layer (int): Number of token layer
        n_token (int): Number of visual token per layer
        n_channel (int): Number of channel of the output of the feature extractor
        n_hidden (int): Number of hidden modules in the transformer
        n_classes (int): Number of classes
    Returns:
        VisualTransformerClassifier: Composite Classifier
    """
    resnet18_top = ResNet18Top()
    tokenizer = Tokenizer(
        n_token_layer=n_token_layer, n_token=n_token, n_channel=n_channel
    )
    transformer = Transformer(n_channel=n_channel, n_hidden=n_hidden)
    projector = Projector(n_channel=n_channel)
    visual_transformer = VisualTransformer(
        tokenizer=tokenizer, transformer=transformer, projector=projector
    )
    classifier = ResNet18Classifier(in_features=n_channel, out_features=n_classes)
    visual_transformer_classifier = VisualTransformerClassifier(
        feature_extractor=resnet18_top,
        visual_transformer=visual_transformer,
        classifier=classifier,
    )
    return visual_transformer_classifier


def create_model_cuda(
    n_token_layer: int,
    n_token: int,
    n_channel: int,
    n_hidden: int,
    n_classes: int,
    device: torch.device,
) -> VisualTransformerClassifier:
    """Creates a visual transformer classifier
    Args:
        n_token_layer (int): Number of token layer
        n_token (int): Number of visual token per layer
        n_channel (int): Number of channel of the output of the feature extractor
        n_hidden (int): Number of hidden modules in the transformer
        n_classes (int): Number of classes
    Returns:
        VisualTransformerClassifier: Composite Classifier
    """
    resnet18_top = ResNet18Top()
    resnet18_top.to(device)
    tokenizer = Tokenizer(
        n_token_layer=n_token_layer, n_token=n_token, n_channel=n_channel
    )
    tokenizer.to_device(device)
    transformer = Transformer(n_channel=n_channel, n_hidden=n_hidden)
    transformer.to(device)
    projector = Projector(n_channel=n_channel)
    projector.to(device)
    visual_transformer = VisualTransformer(
        tokenizer=tokenizer, transformer=transformer, projector=projector
    )
    visual_transformer.to(device)
    classifier = ResNet18Classifier(in_features=n_channel, out_features=n_classes)
    classifier.to(device)
    visual_transformer_classifier = VisualTransformerClassifier(
        feature_extractor=resnet18_top,
        visual_transformer=visual_transformer,
        classifier=classifier,
    )
    return visual_transformer_classifier
