from random import seed

import torch
from modules import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualisations import plot_latent_3D_convolutional, plot_reconstructed_for_point

if __name__ == "__main__":
    # config
    latent_dims = 3
    num_epochs = 40

    seed(0)
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

    vae = ConvolutionalVariationalAutoencoder(latent_dims)
    with open(
        f"./master-thesis/mnist/VAE/conv_vae_{num_epochs}_epochs_{latent_dims}_dims.pt",
        "rb",
    ) as f:
        vae.load_state_dict(torch.load(f))

    # print(
    #     f"VAE parameters: mu: {vae.encoder.mu.detach()}, sigma: {vae.encoder.sigma.detach()}" # noqa: E501
    # )
    # plot_latent_2D_convolutional(vae, data_loader, num_batches=50)
    plot_latent_3D_convolutional(vae, data_loader, 50)
    # plot_latent3D_single_point(vae, data_loader)
    # plot_reconstructed_2D(vae, (-1.5, 1.5), (-1.5, 1.5))

    point = torch.tensor([[1, -1, 0]], dtype=torch.float)
    plot_reconstructed_for_point(vae, point)
