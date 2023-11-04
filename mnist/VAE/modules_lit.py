import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# config
LATENT_DIMS = 3
LR = 3e-4
NUM_WORKERS = 8  # on MacBook M2
NUM_EPOCHS = 200
BATCH_SIZE = 64
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class LinearEncoder(pl.LightningModule):
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


class ConvolutionalEncoder(pl.LightningModule):
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


class ConvolutionalDecoder(pl.LightningModule):
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


class LinearDecoder(pl.LightningModule):
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


class LinearVariationalEncoder(pl.LightningModule):
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


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, latent_dims: int):
        super(LinearAutoencoder, self).__init__()
        self.encoder = LinearEncoder(latent_dims)
        self.decoder = LinearDecoder(latent_dims)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)


class ConvolutionalAutoencoder(pl.LightningModule):
    def __init__(self) -> None:
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = ConvolutionalEncoder()
        self.decoder = ConvolutionalDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LinearVariationalAutoencoder(pl.LightningModule):
    def __init__(self, latent_dims: int):
        super(LinearVariationalAutoencoder, self).__init__()
        self.encoder = LinearVariationalEncoder(latent_dims)
        self.sampler = GaussianSampler()
        self.decoder = LinearDecoder(latent_dims)

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        return self.decoder(z)


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
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x) + model.sampler.kl
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        mu, sigma = self.encoder(x)
        z = self.sampler(mu, sigma)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)


def plot_latent_3D_convolutional(model, data_loader, num_batches=100):
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        mu, sigma = model.encoder(img)
        z = model.sampler(mu, sigma)
        z = z.to("cpu").detach().numpy()

        # Data for three-dimensional scattered points
        xdata = z[:, 0]
        ydata = z[:, 1]
        zdata = z[:, 2]

        # Plot the data
        plot = ax.scatter(xdata, ydata, zdata, c=label, cmap="tab10", marker="o")

        # Label the axes, config the plot
        ax.grid(False)
        ax.set_title("Encoder Output")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    plt.show()


if __name__ == "__main__":
    # data
    dataset = datasets.MNIST(
        "", train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(
        mnist_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # model
    model = ConvolutionalVariationalAutoencoder(LATENT_DIMS)

    # training
    trainer = pl.Trainer(
        accelerator="mps",
        precision="16-mixed",
        limit_train_batches=0.5,
        max_epochs=NUM_EPOCHS,
    )
    trainer.fit(model, train_loader, val_loader)

    # save model
    with open("./master-thesis/mnist/VAE/lightning.pt", "wb") as f:
        torch.save(model.state_dict(), f)

    # load model
    # with open("./master-thesis/mnist/VAE/lightning.pt", "rb") as f:
    #     model.load_state_dict(torch.load(f))

    # # plot latentç
    # plot_latent_3D_convolutional(model, train_loader, 50)
