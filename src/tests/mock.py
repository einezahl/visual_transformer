import torch
from torch import nn


class MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.c)


class MockLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


class MockOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(1.0, alpha=1.0)

    def zero_grad(self):
        pass


class MockDataSet(torch.utils.data.Dataset):
    def __init__(self, data=None, labels=None, size=100):
        self.data = torch.randn(100)
        self.labels = torch.randn(100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MockDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, num_workers)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from utils.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel()
    loss = MockLoss()
    optimizer = MockOptimizer(model.parameters())
    dataset = MockDataSet()
    data_loader = MockDataLoader(dataset)
    epochs = 10
    trainer = Trainer(model, loss, optimizer, device)
    trainer.train(data_loader, epochs)

    assert trainer.model.c.data == 1000.0
    print(trainer.model.c.data == 1000.0)
