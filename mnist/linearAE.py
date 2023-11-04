# Import dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VAE.visualisations import plot_latent3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Define a linear AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N (batch size), 784 (28x28)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # N, 3
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),  # N, 784
            nn.Sigmoid(),  # IMPORTTANT! Depending on data we might need different activation here! # noqa: E501
        )

    # NOTE: Last activation: [0, 1] -> nn.ReLU(), [-1, 1] -> nn.Tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Train the Autoencoder
def train(autoencoder, optimizer, criterion, data_loader, num_epochs=10):
    outputs = []

    for epoch in range(num_epochs):
        for img, _ in data_loader:
            img = img.reshape(-1, 28 * 28)
            recon = autoencoder(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))
    return outputs, autoencoder


if __name__ == "__main__":
    # Get data
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    data_loader = DataLoader(mnist_data, batch_size=64, shuffle=True)

    # Analyze data
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    print(torch.min(images), torch.max(images))

    # Instantiate neural net and loss
    autoencoder = AutoEncoder()
    criterion = nn.MSELoss()

    # Config training
    num_epochs = 12
    optimizer = optim.Adam(autoencoder.parameters(), lr=5e-3, weight_decay=1e-5)

    outputs, autoencoder = train(
        autoencoder, optimizer, criterion, data_loader, num_epochs
    )

    plot_latent3D(autoencoder, data_loader, 50)
