# from datetime import date
import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# config
torch.set_float32_matmul_precision("medium")
LATENT_DIMS = 10
NUM_EPOCHS = 1000
NUM_WORKERS = os.cpu_count()
LEARNING_RATE = 3e-4
BATCH_SIZE = 256 * 4
KL_COEFF = 4e-5
PERSISTENT_WORKERS = True
# strategy = DDPStrategy()

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


class ConvolutionalVariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(64, latent_dims)
        self.linear3 = nn.Linear(64, latent_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.flatten(self.encoder(x), start_dim=1)

        mu, log_var = self.linear2(encoded), self.linear3(encoded)
        return mu, log_var


class ConvolutionalVariationalDecoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalDecoder, self).__init__()
        self.linear = nn.Linear(latent_dims, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.decoder(x)


class ConvolutionalVariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        latent_dims: int,
        kl_coeff: float = KL_COEFF,
        learning_rate: float = LEARNING_RATE,
    ):
        super().__init__()
        self.encoder = ConvolutionalVariationalEncoder(latent_dims)
        self.decoder = ConvolutionalVariationalDecoder(latent_dims)

        # config
        self.kl_coeff = kl_coeff
        self.lr = learning_rate

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        _, _, sample = self.sample(mu, log_var)

        return self.decoder(sample)

    def _run_step(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        p, q, sample = self.sample(mu, log_var)
        return sample, self.decoder(sample), p, q

    def step(self, batch, batch_idx):
        x, _ = batch
        sample, x_hat, p, q = self._run_step(x)

        log_qz = q.log_prob(sample)
        log_pz = p.log_prob(sample)

        kl = self.kl_coeff * (log_qz - log_pz).mean()

        loss = F.mse_loss(x_hat, x, reduction="mean") + kl
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
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
    vae = ConvolutionalVariationalAutoencoder(LATENT_DIMS)

    # training
    trainer = pl.Trainer(
        strategy="ddp",  # if model too large: from pl.strategies import deepspeed
        profiler="simple",
        accelerator="gpu",
        devices=-1,
        precision="32",
        max_epochs=NUM_EPOCHS,
        enable_progress_bar=True,
        check_val_every_n_epoch=50,
        accumulate_grad_batches=10,
        log_every_n_steps=5,
    )
    trainer.fit(vae, dm)
