import torch
from modules import VariationalAutoencoder
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualisations import plot_reconstructed_2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(autoencoder, optimizer, data_loader, num_epochs=10):
    outputs = []

    for epoch in range(num_epochs):
        for img, _ in data_loader:
            optimizer.zero_grad()

            recon = autoencoder(img)
            loss = ((img - recon) ** 2).sum() + autoencoder.encoder.kl

            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))
    return outputs, autoencoder


if __name__ == "__main__":
    # config
    latent_dims = 2

    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

    vae = VariationalAutoencoder(latent_dims)

    optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    outputs, vae = train(
        vae, optimizer=optimizer, data_loader=data_loader, num_epochs=20
    )

    plot_reconstructed_2D(vae, (-3, 3), (-3, 3))
