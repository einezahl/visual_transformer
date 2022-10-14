import torch

from components.feature_extractor import ResNetReduced
from components.tokenizer import FilterTokenLayer, RecurrentTokenLayer, Tokenizer
from components.transformer import Transformer


class TestComponents:
    """In this class the components of the visual transformer are tested in a pipeline, where
    the output of the previous component is the input for the next component
    """

    def test_feature_extractor(self):
        """Test if the output of the ResNetReduced class has the correct shape for the CIFAR
        dataset"""
        batch_input = torch.randn((10, 3, 32, 32))
        rnreduced = ResNetReduced()
        preprocessed_data = rnreduced(batch_input)
        assert preprocessed_data.shape == (10, 256, 2, 2)

    def test_tokenizer_components(self):
        """Test if the output of the components of the tokenizer have the correct shape"""
        feature_map_input = torch.randn(10, 256, 4)
        filter_token_layer = FilterTokenLayer(16, 256)
        first_layer_output = filter_token_layer(feature_map_input)
        assert first_layer_output.shape == (10, 16, 256)
        recurrent_token_layer = RecurrentTokenLayer(256)
        second_layer_output = recurrent_token_layer(
            feature_map_input, first_layer_output
        )
        assert second_layer_output.shape == (10, 16, 256)

    def test_tokenizer(self):
        """Test if the output of the tokenizer has the correct shape"""
        feature_map_input = torch.randn(10, 256, 4)
        tokenizer = Tokenizer(n_token_layer=6, n_channel=256, n_token=16)
        tokenizer_output = tokenizer(feature_map_input)
        assert tokenizer_output.shape == (10, 16, 256)

    def test_transformer(self):
        """Test if the output of the transformer has the correct shape"""
        visual_token_in = torch.randn(10, 16, 256)
        transformer = Transformer(n_channel=256, n_hidden=6)
        transformer_output = transformer(visual_token_in)
        assert transformer_output.shape == (10, 16, 256)
