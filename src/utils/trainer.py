import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        writer: SummaryWriter,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.writer = writer

    def train(self, train_loader: DataLoader, epoch: int) -> None:
        for i in range(epoch):
            self.train_epoch(train_loader, i)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        running_loss = 0.0
        for i, (image_batch, label_batch) in enumerate(train_loader):
            image_batch = image_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image_batch)
            loss_value = self.loss(output, label_batch)
            loss_value.backward()
            self.optimizer.step()
            running_loss += loss_value.item()
        self.writer.add_scalar("Train/loss", running_loss, epoch)
        print(f"[{epoch + 1}] loss: {running_loss / 100:.3f}")
