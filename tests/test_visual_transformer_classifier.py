import torch

from model.visual_transformer_classifier import VisualTransformerClassifier


class TestVisualTransformerClassifier:
    """Visual transformer classifier tests"""

    def test_output_shape(self):
        """Tests the correct output shape of the composite visual transformer
        classifier"""
        batch_size = 10
        n_channel = 256
        n_token = 16
        n_token_layer = 6
        n_hidden = 6
        image_batch = torch.randn((batch_size, 3, 32, 32))
        visual_transformer_classifier = VisualTransformerClassifier(
            n_token_layer=n_token_layer,
            n_token=n_token,
            n_channel=n_channel,
            n_hidden=n_hidden,
            n_classes=10,
        )
        predictions = visual_transformer_classifier(image_batch)
        assert predictions.shape == (batch_size, 10)

    def test_output_shape_cuda(self):
        """Tests whether the model is correctly initilized when using a GPU"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 10
        n_channel = 256
        n_token = 16
        n_token_layer = 6
        n_hidden = 6
        image_batch = torch.randn((batch_size, 3, 32, 32))
        image_batch = image_batch.to(device)

        visual_transformer_classifier = VisualTransformerClassifier(
            n_token_layer=n_token_layer,
            n_token=n_token,
            n_channel=n_channel,
            n_hidden=n_hidden,
            n_classes=10,
        )
        visual_transformer_classifier.to(device)
        predictions = visual_transformer_classifier(image_batch)
        assert predictions.shape == (batch_size, 10)
