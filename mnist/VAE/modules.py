import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearEncoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(784, 512).to(DEVICE),
            nn.ReLU(),
            nn.Linear(512, 124).to(DEVICE),
        )
        self.linear2 = nn.Linear(124, latent_dims).to(DEVICE)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
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
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.encoder(x))


class ConvolutionalDecoder(nn.Module):
    def __init__(self):
        super(ConvolutionalDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7).to(DEVICE),  #  N, 64, 1, 1 -> N, 32, 7, 7
            nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3), #  N, 32, 7, 7 -> N, 16, 13, 13 THE DIMENSIONS WOULD NOT ADD UP!! # noqa: E501
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1).to(
                DEVICE
            ),  #  N, 32, 7, 7 -> N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1).to(
                DEVICE
            ),  #  N, 16, 14, 14 -> N, 1, 28, 28
            nn.Sigmoid(),  # IMPORTANT! Depending on data we might need different activation here! # noqa: E501
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.decoder(x))


class LinearDecoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(LinearDecoder, self).__init__()
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


class LinearVariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(LinearVariationalEncoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(784, 512).to(DEVICE),
            nn.ReLU(),
            nn.Linear(512, 124).to(DEVICE),
        )
        self.linear2 = nn.Linear(124, latent_dims).to(DEVICE)
        self.linear3 = nn.Linear(124, latent_dims).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.float64:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu, sigma = self.linear2(x), self.linear3(x)
        return mu, sigma


class GaussianSampler(nn.Module):
    def __init__(self) -> None:
        super(GaussianSampler, self).__init__()
        self.N = torch.distributions.Normal(0, 1)  # initialise standard normal
        self.N.loc = (
            self.N.loc
        )  # .cuda() # hack to get sampling on the GPU -> can't access it on MacBook
        self.N.scale = (
            self.N.scale
        )  # .cuda() # hack to get sampling on the GPU -> can't access it on MacBook

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


class ConvolutionalVariationalEncoder(nn.Module):
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
        )
        self.linear2 = nn.Linear(64, latent_dims).to(DEVICE)
        self.linear3 = nn.Linear(64, latent_dims).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)

        mu, sigma = self.linear2(encoded), torch.exp(self.linear3(encoded))

        return mu, sigma


class ConvolutionalVariationalDecoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalDecoder, self).__init__()
        self.linear = nn.Linear(latent_dims, 64).to(DEVICE)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7).to(DEVICE),  #  N, 64, 1, 1 -> N, 32, 7, 7
            nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3), #  N, 32, 7, 7 -> N, 16, 13, 13 THE DIMENSIONS WOULD NOT ADD UP!! # noqa: E501
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1).to(
                DEVICE
            ),  #  N, 32, 7, 7 -> N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1).to(
                DEVICE
            ),  #  N, 16, 14, 14 -> N, 1, 28, 28
            nn.Sigmoid(),  # IMPORTANT! Depending on data we might need different activation here! # noqa: E501
        )

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return F.relu(self.decoder(x))


class LinearAutoencoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(LinearAutoencoder, self).__init__()
        self.encoder = LinearEncoder(latent_dims)
        self.decoder = LinearDecoder(latent_dims)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self) -> None:
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = ConvolutionalEncoder()
        self.decoder = ConvolutionalDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LinearVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(LinearVariationalAutoencoder, self).__init__()
        self.encoder = LinearVariationalEncoder(latent_dims)
        self.sampler = GaussianSampler()
        self.decoder = LinearDecoder(latent_dims)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        return self.decoder(z)


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(ConvolutionalVariationalAutoencoder, self).__init__()
        self.encoder = ConvolutionalVariationalEncoder(latent_dims)
        self.sampler = GaussianSampler()
        self.decoder = ConvolutionalVariationalDecoder(latent_dims)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        return self.decoder(z)
