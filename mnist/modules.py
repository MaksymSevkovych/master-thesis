import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(Encoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(784, 512).to(DEVICE),
            nn.ReLU(),
            nn.Linear(512, 124).to(DEVICE),
        )
        self.linear2 = nn.Linear(512, latent_dims).to(DEVICE)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(latent_dims, 124).to(DEVICE),
            nn.ReLU(),
            nn.Linear(124, 512).to(DEVICE),
        )
        self.linear2 = nn.Linear(512, 784).to(DEVICE)

    def forward(self, z: torch.Tensor):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(784, 512).to(DEVICE),
            nn.ReLU(),
            nn.Linear(512, 124).to(DEVICE),
        )
        self.linear2 = nn.Linear(124, latent_dims).to(DEVICE)
        self.linear3 = nn.Linear(124, latent_dims).to(DEVICE)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = (
            self.N.loc
        )  # .cuda() # hack to get sampling on the GPU -> can't access it on MacBook
        self.N.scale = (
            self.N.scale
        )  # .cuda() # hack to get sampling on the GPU -> can't access it on MacBook
        self.kl = 0

    def forward(self, x: torch.Tensor) -> torch.float64:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Autoencoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)
