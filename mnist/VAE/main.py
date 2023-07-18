import torch
from modules import LinearVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualisations import plot_latent_2D_linear

if __name__ == "__main__":
    # config
    latent_dims = 2
    num_epochs = 10

    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

    vae = LinearVariationalAutoencoder(latent_dims)
    with open(
        f"./master-thesis/mnist/VAE/linear_vae_{num_epochs}_epochs_{latent_dims}_dims.pt",
        "rb",
    ) as f:
        vae.load_state_dict(torch.load(f))

    # print(
    #     f"VAE parameters: mu: {vae.encoder.mu.detach()}, sigma: {vae.encoder.sigma.detach()}" # noqa: E501
    # )
    plot_latent_2D_linear(vae, data_loader, num_batches=50)
    # plot_latent3D(vae, data_loader, 50)
    # plot_latent3D_single_point(vae, data_loader)
    # plot_reconstructed_2D(vae, (-3, 3), (-3, 3))
