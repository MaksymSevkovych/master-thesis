import time
from random import seed

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# config
LATENT_DIMS = 3
NUM_EPOCHS = 1000
NUM_WORKERS = 64
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
SEED = 0
ALPHA = 1
FILE_NAME = f"conv_vae_{NUM_EPOCHS}_epochs_{LATENT_DIMS}_dims_{LEARNING_RATE}_lr_{ALPHA}_alpha.pt"  # noqa: E501

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianSampler(pl.LightningModule):
    def __init__(self) -> None:
        super(GaussianSampler, self).__init__()
        self.N = torch.distributions.Normal(0, 1)  # initialise standard normal
        self.N.loc = self.N.loc.to(DEVICE)
        self.N.scale = self.N.scale.to(DEVICE)

        # Kullback-Leibler divergence loss:
        self.kl = 0

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor):
        # mu + sigma * z ~ N(mu, sigma), if z ~ N(0, 1)

        z = mu + sigma * self.N.sample(mu.shape)

        # self.kl = torch.sum(-0.5 * (1 + sigma - mu.pow(2) - torch.exp(sigma)))
        # NOTE: what if we distinguish between the hidden normal distributions??
        # sigma_goal = torch.tensor([1] * sigma.shape[1])
        # mu_goal = torch.tensor([0] * mu.shape[1])
        # mu_goal = torch.tensor(list(range(mu.shape[1])))

        self.kl = torch.sum(-torch.log(sigma) + 1 / 2 * (sigma.pow(2) + mu.pow(2) - 1))
        # self.kl = torch.sum(
        #     torch.log(sigma / sigma_goal)
        #     + (sigma_goal.pow(2) + (mu_goal - mu).pow(2)) / (2 * sigma.pow(2))
        #     - 1 / 2
        # )
        return z


class ConvolutionalVariationalEncoder(pl.LightningModule):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1).to(
                DEVICE
            ),  # N, 1, 28, 28 -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1).to(
                DEVICE
            ),  # N, 16, 14, 14 -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7).to(DEVICE),  # N, 32, 7, 7 -> N, 64, 1, 1
            nn.ReLU(),
        ).to(DEVICE)
        self.linear2 = nn.Linear(64, latent_dims).to(DEVICE)
        self.linear3 = nn.Linear(64, latent_dims).to(DEVICE)

    def forward(self, x: torch.tensor) -> torch.tensor:
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)

        mu, sigma = self.linear2(encoded), torch.exp(self.linear3(encoded))

        return mu, sigma


class ConvolutionalVariationalDecoder(pl.LightningModule):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalDecoder, self).__init__()
        self.linear = nn.Linear(latent_dims, 64).to(DEVICE)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  #  N, 64, 1, 1 -> N, 32, 7, 7
            nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3), #  N, 32, 7, 7 -> N, 16, 13, 13 THE DIMENSIONS WOULD NOT ADD UP!! # noqa: E501
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  #  N, 32, 7, 7 -> N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, stride=2, padding=1, output_padding=1
            ),  #  N, 16, 14, 14 -> N, 1, 28, 28
            nn.Sigmoid(),  # IMPORTANT! Depending on data we might need different activation here! # noqa: E501
        ).to(DEVICE)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return F.relu(self.decoder(x))


class ConvolutionalVariationalAutoencoder(pl.LightningModule):
    def __init__(self, latent_dims: int):
        super().__init__()
        self.encoder = ConvolutionalVariationalEncoder(latent_dims).to(DEVICE)
        self.sampler = GaussianSampler().to(DEVICE)
        self.decoder = ConvolutionalVariationalDecoder(latent_dims).to(DEVICE)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma).to(DEVICE)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x) + self.sampler.kl
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)


def train(model, optimizer, data_loader, num_epochs=10):
    outputs = []
    mse_fn = torch.nn.MSELoss(reduction="sum")

    print("training started")
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        for img, _ in data_loader:
            img = img.to(DEVICE)
            recon = model(img)

            mse = mse_fn(recon, img)
            kl = model.sampler.kl

            loss = ALPHA * mse + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.perf_counter()

        duration = end_time - start_time

        print(
            f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Duration: {duration:.1f} secs"
        )
        outputs.append((epoch, img, recon))
    return outputs, model


if __name__ == "__main__":
    # seed
    seed(SEED)
    # data
    data = datasets.MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    vae = ConvolutionalVariationalAutoencoder(LATENT_DIMS)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    print("model initialized")

    _, vae = train(
        vae,
        optimizer=optimizer,
        data_loader=data_loader,
        num_epochs=NUM_EPOCHS,
    )

    with open(FILE_NAME, "wb") as f:
        torch.save(vae.state_dict(), f)
