import torch

from model.visual_transformer_classifier import VisualTransformerClassifier


class TestVisualTransformerClassifier:
    """Visual transformer classifier tests"""

    def test_output_shape(self):
        """Tests the correct output shape of the composite visual transformer
        classifier"""
        batch_size = 10
        n_token = 16
        n_token_layer = 6
        n_hidden = 6
        image_batch = torch.randn((batch_size, 3, 32, 32))
        visual_transformer_classifier = VisualTransformerClassifier(
            n_token_layer=n_token_layer,
            n_token=n_token,
            n_hidden=n_hidden,
            n_classes=10,
        )
        predictions = visual_transformer_classifier(image_batch)
        assert predictions.shape == (batch_size, 10)

    def test_output_shape_cuda(self):
        """Tests whether the model is correctly initilized when using a GPU"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 10
        n_token = 16
        n_token_layer = 6
        n_hidden = 6
        image_batch = torch.randn((batch_size, 3, 32, 32))
        image_batch = image_batch.to(device)

        visual_transformer_classifier = VisualTransformerClassifier(
            n_token_layer=n_token_layer,
            n_token=n_token,
            n_hidden=n_hidden,
            n_classes=10,
        )
        visual_transformer_classifier.to(device)
        predictions = visual_transformer_classifier(image_batch)
        assert predictions.shape == (batch_size, 10)

    def test_torch_parameter(self):
        """Test whether all parameters of the classifier are found by torch"""
        n_token = 16
        n_token_layer = 6
        n_hidden = 32

        visual_transformer_classifier = VisualTransformerClassifier(
            n_token_layer=n_token_layer,
            n_token=n_token,
            n_hidden=n_hidden,
            n_classes=10,
        )

        visual_transformer_classifier_parameter = list(
            visual_transformer_classifier.parameters()
        )
        # ResNet parameters
        visual_transformer_classifier_parameter_dimensions = []
        visual_transformer_classifier_parameter_dimensions.append([64, 3, 7, 7])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64, 64, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64, 64, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64, 64, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64, 64, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([64])
        visual_transformer_classifier_parameter_dimensions.append([128, 64, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128, 128, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128, 64, 1, 1])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128, 128, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128, 128, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([128])
        visual_transformer_classifier_parameter_dimensions.append([256, 128, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256, 128, 1, 1])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256, 3, 3])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256])

        # Tokenizer parameters

        visual_transformer_classifier_parameter_dimensions.append([16, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])

        # Transformer parameters

        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([32, 256])
        visual_transformer_classifier_parameter_dimensions.append([32])
        visual_transformer_classifier_parameter_dimensions.append([32, 256])
        visual_transformer_classifier_parameter_dimensions.append([32])

        # Projector parameters

        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256])
        visual_transformer_classifier_parameter_dimensions.append([256, 256])
        visual_transformer_classifier_parameter_dimensions.append([256])

        # Classifier parameters

        visual_transformer_classifier_parameter_dimensions.append([10, 256])
        visual_transformer_classifier_parameter_dimensions.append([10])

        for i in range(len(visual_transformer_classifier_parameter)):
            assert visual_transformer_classifier_parameter[i].shape == torch.Size(
                visual_transformer_classifier_parameter_dimensions[i]
            )
