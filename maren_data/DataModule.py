from torch.utils.data import DataLoader

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        data_dir: str = "data",
        batch_size: int = 256,
    ):
        super().__init__()
        print('initializing data module')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader=testloader

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader

    def predict_dataloader(self):
        return NotImplementedError
