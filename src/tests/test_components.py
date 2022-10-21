import torch

from components.classifier import ResNet18Classifier
from components.feature_extractor import ResNet18Top
from components.projector import Projector
from components.tokenizer import FilterTokenLayer, RecurrentTokenLayer, Tokenizer
from components.transformer import Transformer
from components.visual_transformer import VisualTransformer


class TestComponents:
    """In this class the components of the visual transformer are tested in a
    pipeline, where the output of the previous component is the input for the
    next component
    """

    def test_feature_extractor_output_shape(self):
        """Test if the output of the ResNetReduced class has the correct shape
        for the CIFAR dataset"""
        batch_input = torch.randn((10, 3, 32, 32))

        rnreduced = ResNet18Top()
        preprocessed_data = rnreduced(batch_input)

        assert preprocessed_data.shape == (10, 256, 2, 2)

    def test_feature_extractor_torch_parameter(self):
        """Test whether all parameters of ResNet18 network are found by torch"""
        resnet18_top = ResNet18Top()

        parameter = list(resnet18_top.parameters())

        parameter_dimensions = []
        parameter_dimensions.append([64, 3, 7, 7])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64, 64, 3, 3])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64, 64, 3, 3])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64, 64, 3, 3])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64, 64, 3, 3])
        parameter_dimensions.append([64])
        parameter_dimensions.append([64])
        parameter_dimensions.append([128, 64, 3, 3])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128, 128, 3, 3])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128, 64, 1, 1])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128, 128, 3, 3])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128, 128, 3, 3])
        parameter_dimensions.append([128])
        parameter_dimensions.append([128])
        parameter_dimensions.append([256, 128, 3, 3])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256, 256, 3, 3])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256, 128, 1, 1])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256, 256, 3, 3])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256, 256, 3, 3])
        parameter_dimensions.append([256])
        parameter_dimensions.append([256])

        for i, param in enumerate(parameter):
            assert param.shape == torch.Size(parameter_dimensions[i])

    def test_tokenizer_components_output_shape(self):
        """Test if the output of the components of the tokenizer have the
        correct shape"""
        feature_map_input = torch.randn(10, 256, 4)

        filter_token_layer = FilterTokenLayer(16, 256)
        first_layer_output = filter_token_layer(feature_map_input)

        assert first_layer_output.shape == (10, 16, 256)

        recurrent_token_layer = RecurrentTokenLayer(256)
        second_layer_output = recurrent_token_layer(
            feature_map_input, first_layer_output
        )

        assert second_layer_output.shape == (10, 16, 256)

    def test_tokenizer_component_torch_parameter(self):
        """Test whether all parameters of the components of the tokenizer are
        found by torch"""
        filter_token_layer = FilterTokenLayer(16, 256)
        recurrent_token_layer = RecurrentTokenLayer(256)

        filter_token_layer_parameter = list(filter_token_layer.parameters())
        recurrent_token_layer_parameter = list(recurrent_token_layer.parameters())

        filter_token_layer_parameter_dimensions = []
        filter_token_layer_parameter_dimensions.append([16, 256])

        recurrent_token_layer_parameter_dimensions = []
        recurrent_token_layer_parameter_dimensions.append([256, 256])

        for i, param in enumerate(filter_token_layer_parameter):
            assert param.shape == torch.Size(filter_token_layer_parameter_dimensions[i])

        for i, param in enumerate(recurrent_token_layer_parameter):
            assert param.shape == torch.Size(
                recurrent_token_layer_parameter_dimensions[i]
            )

    def test_tokenizer_output_shape(self):
        """Test if the output of the tokenizer has the correct shape"""
        feature_map_input = torch.randn(10, 256, 4)

        tokenizer = Tokenizer(n_token_layer=6, n_channel=256, n_token=16)
        tokenizer_output = tokenizer(feature_map_input)

        assert tokenizer_output.shape == (10, 16, 256)

    def test_tokenizer_torch_parameter(self):
        """Test whether all parameters of the tokenizer are found by torch"""
        tokenizer = Tokenizer(n_token_layer=6, n_channel=256, n_token=16)

        tokenizer_parameter = list(tokenizer.parameters())

        tokenizer_parameter_dimensions = []
        tokenizer_parameter_dimensions.append([16, 256])
        tokenizer_parameter_dimensions.append([256, 256])
        tokenizer_parameter_dimensions.append([256, 256])
        tokenizer_parameter_dimensions.append([256, 256])
        tokenizer_parameter_dimensions.append([256, 256])
        tokenizer_parameter_dimensions.append([256, 256])

        for i, param in enumerate(tokenizer_parameter):
            assert param.shape == torch.Size(tokenizer_parameter_dimensions[i])

    def test_transformer_output_shape(self):
        """Test if the output of the transformer has the correct shape"""
        visual_token_in = torch.randn(10, 16, 256)
        transformer = Transformer(n_channel=256, n_hidden=6)

        transformer_output = transformer(visual_token_in)

        assert transformer_output.shape == (10, 16, 256)

    def test_transformer_torch_parameter(self):
        """Test whether all parameters of the transformer are found by torch"""
        transformer = Transformer(n_channel=256, n_hidden=32)

        transformer_parameter = list(transformer.parameters())

        transformer_parameter_dimensions = []
        transformer_parameter_dimensions.append([256, 256])
        transformer_parameter_dimensions.append([256, 256])
        transformer_parameter_dimensions.append([32, 256])
        transformer_parameter_dimensions.append([32])
        transformer_parameter_dimensions.append([32, 256])
        transformer_parameter_dimensions.append([32])

        for i, param in enumerate(transformer_parameter):
            assert param.shape == torch.Size(transformer_parameter_dimensions[i])

    def test_projector_output_shape(self):
        """Test if the output of the projector has the correct shape"""
        feature_map_input = torch.randn(10, 256, 4)
        visual_token_in = torch.randn(10, 16, 256)

        projector = Projector(n_channel=256)
        projector_output = projector(feature_map_input, visual_token_in)

        assert projector_output.shape == (10, 256, 4)

    def test_projector_torch_parameter(self):
        """Test whether all parameters of the projector are found by torch"""
        projector = Projector(n_channel=256)

        projector_parameter = list(projector.parameters())

        projector_parameter_dimensions = []
        projector_parameter_dimensions.append([256, 256])
        projector_parameter_dimensions.append([256])
        projector_parameter_dimensions.append([256, 256])
        projector_parameter_dimensions.append([256])

        for i, param in enumerate(projector_parameter):
            assert param.shape == torch.Size(projector_parameter_dimensions[i])

    def test_classifier_output_shape(self):
        """Test if the output of the classifier has the correct shape"""
        feature_map_input = torch.randn(10, 256, 2, 2)

        classifier = ResNet18Classifier(256, 100)
        classifier_output = classifier(feature_map_input)

        assert classifier_output.shape == (10, 100)

    def test_classifier_torch_parameter(self):
        """Test whether all parameters of the classifier are found by torch"""
        classifier = ResNet18Classifier(256, 10)

        classifier_parameter = list(classifier.parameters())

        classifier_parameter_dimensions = []
        classifier_parameter_dimensions.append([10, 256])
        classifier_parameter_dimensions.append([10])

        for i, param in enumerate(classifier_parameter):
            assert param.shape == torch.Size(classifier_parameter_dimensions[i])

    def test_visual_transformer_output_shape(self):
        """Test if the output of the classifier has the correct shape"""
        feature_map_input = torch.randn(10, 256, 2, 2)

        tokenizer = Tokenizer(n_token_layer=6, n_channel=256, n_token=16)
        transformer = Transformer(n_channel=256, n_hidden=6)
        projector = Projector(n_channel=256)
        visual_transformer = VisualTransformer(
            tokenizer=tokenizer, transformer=transformer, projector=projector
        )
        visual_transformer_output = visual_transformer(feature_map_input)

        assert visual_transformer_output.shape == (10, 256, 2, 2)

    def test_visual_transformer_torch_parameter(self):
        """Test whether all parameters of the classifier are found by torch"""

        tokenizer = Tokenizer(n_token_layer=6, n_channel=256, n_token=16)
        transformer = Transformer(n_channel=256, n_hidden=32)
        projector = Projector(n_channel=256)
        visual_transformer = VisualTransformer(
            tokenizer=tokenizer, transformer=transformer, projector=projector
        )

        visual_transformer_parameter = list(visual_transformer.parameters())

        visual_transformer_parameter_dimensions = []
        visual_transformer_parameter_dimensions.append([16, 256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256, 256])

        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([32, 256])
        visual_transformer_parameter_dimensions.append([32])
        visual_transformer_parameter_dimensions.append([32, 256])
        visual_transformer_parameter_dimensions.append([32])

        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256])
        visual_transformer_parameter_dimensions.append([256, 256])
        visual_transformer_parameter_dimensions.append([256])

        for i, param in enumerate(visual_transformer_parameter):
            print(i)
            assert param.shape == torch.Size(visual_transformer_parameter_dimensions[i])
