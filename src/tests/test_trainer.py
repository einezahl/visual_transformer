import torch

from tests.mock import MockModel, MockLoss, MockOptimizer, MockDataSet, MockDataLoader
from utils.trainer import Trainer


class TestTrainer:
    def test_train(self):
        size = 100
        epochs = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MockModel()
        loss = MockLoss()
        optimizer = MockOptimizer(model.parameters())
        dataset = MockDataSet(size=size)
        data_loader = MockDataLoader(dataset)
        trainer = Trainer(model, loss, optimizer, device)
        trainer.train(data_loader, epochs)
        assert trainer.model.c.data == size * epochs
