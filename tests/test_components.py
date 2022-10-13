from components.feature_extractor import ResNetReduced
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def test_feature_extractor():
    data = CIFAR10(download=True, root="./data", transform=ToTensor(), train=False)
    data_loader = DataLoader(data, batch_size=10, shuffle=False)

    first_batch, _ = next(iter(data_loader))

    rnreduced = ResNetReduced()
    preprocessed_data = rnreduced(first_batch)
    assert preprocessed_data.shape == (10, 256, 2, 2)
