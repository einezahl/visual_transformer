import torch

from components.feature_extractor import ResNet18Top
from components.visual_transformer import VisualTransformer
from components.classifier import ResNet18Classifier
from components.projector import Projector
from components.tokenizer import Tokenizer
from components.transformer import Transformer
from model.visual_transformer_classifier import VisualTransformerClassifier


class TestVisualTransformerClassifier:
    """Visual transformer classifier tests"""

    def test_output_shape(self):
        """Tests the correct output shape of the composite visual transformer classifier"""
        batch_size = 10
        n_channel = 256
        n_token = 16
        n_token_layer = 6
        n_hidden = 6
        image_batch = torch.randn((batch_size, 3, 32, 32))
        resnet18_top = ResNet18Top()
        tokenizer = Tokenizer(
            n_token_layer=n_token_layer, n_token=n_token, n_channel=n_channel
        )
        transformer = Transformer(n_channel=n_channel, n_hidden=n_hidden)
        projector = Projector(n_channel=n_channel)
        visual_transformer = VisualTransformer(
            tokenizer=tokenizer, transformer=transformer, projector=projector
        )
        classifier = ResNet18Classifier(in_features=n_channel, out_features=10)
        visual_transformer_classifier = VisualTransformerClassifier(
            feature_extractor=resnet18_top,
            visual_transformer=visual_transformer,
            classifier=classifier,
        )
        predictions = visual_transformer_classifier(image_batch)
        assert predictions.shape == (batch_size, 10)
