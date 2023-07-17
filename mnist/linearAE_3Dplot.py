# Import dependencies
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

    def forward(self, x):
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


# Plot the reconstructed images
def plot_latent3D(autoencoder, data_loader, num_batches=100):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (img, label) in enumerate(data_loader):
        img = img.reshape(-1, 28 * 28)
        z = autoencoder.encoder(img).to(DEVICE)
        z = z.to("cpu").detach().numpy()

        # Data for three-dimensional scattered points
        xdata = z[:, 0]
        ydata = z[:, 1]
        zdata = z[:, 2]

        plot = ax.scatter(xdata, ydata, zdata, c=label, cmap="tab10", marker="o")

        ax.grid(False)
        ax.set_title("Encoder Output")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    plt.show()


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
