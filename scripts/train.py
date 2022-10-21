import hydra
from hydra.core.config_store import ConfigStore
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

from config import CifarConf
from model.visual_transformer_classifier import VisualTransformerClassifier
from utils.trainer import Trainer


cs = ConfigStore.instance()
cs.store(name="cifar_conf", node=CifarConf)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: CifarConf) -> None:
    batch_size = cfg.params.batch_size
    epochs = cfg.params.epochs
    lr = cfg.params.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = CIFAR10(
        root=cfg.paths.data, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_data = CIFAR10(
        root=cfg.paths.data, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    classifier = VisualTransformerClassifier(
        n_token_layer=6,
        n_token=16,
        n_channel=256,
        n_hidden=16,
        n_classes=10,
    )
    classifier.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    trainer = Trainer(classifier, loss, optimizer, device)

    trainer.train(train_loader, epochs)

    print("Finished Training")

    os.makedirs(cfg.paths.model, exist_ok=True)
    path = os.path.join(cfg.paths.model, "model.pth")
    torch.save(classifier.state_dict(), path)

    correct = 0
    total = 0

    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            output = classifier(image_batch)
            _, predicted = torch.max(output.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")


if __name__ == "__main__":
    main()
