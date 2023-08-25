import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# config
torch.set_float32_matmul_precision("medium")
LATENT_DIMS = 10
NUM_EPOCHS = 100000
NUM_WORKERS = os.cpu_count()
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
PERSISTENT_WORKERS = True
AMSGRAD = True
strategy = DDPStrategy()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, num_workers: int, persistent_workers: bool
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )


class LinearEncoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dims),
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class LinearDecoder(nn.Module):
    def __init__(self, latent_dims: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class LinearAutoEncoder(pl.LightningModule):
    def __init__(self, latent_dims: int, lr: float, amsgrad: bool = False) -> None:
        super().__init__()
        self.encoder = LinearEncoder(latent_dims)
        self.decoder = LinearDecoder(latent_dims)

        # CONFIG
        self.lr = lr
        self.amsgrad = amsgrad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def _run_step(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

    def step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(-1, 28 * 28)
        x_hat = self._run_step(x)

        loss = F.mse_loss(x_hat, x, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, amsgrad=self.amsgrad)
        return optimizer


if __name__ == "__main__":
    # data
    dm = MnistDataModule(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )

    # model
    model = LinearAutoEncoder(LATENT_DIMS, LEARNING_RATE, AMSGRAD)

    # training
    trainer = pl.Trainer(
        strategy=strategy,
        profiler="simple",
        accelerator="gpu",
        devices=-1,
        precision="32",
        max_epochs=NUM_EPOCHS,
        # enable_progress_bar=True,
        check_val_every_n_epoch=50,
        accumulate_grad_batches=10,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)
