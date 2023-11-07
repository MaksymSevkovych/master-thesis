import os

import torch
from isa_conv_VAE import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualise import (
    inference_convolutional,
    plot_latent_2D_convolutional,
    plot_reconstructed_2D,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG

BASE_PATH = os.path.dirname(__file__)
BATCH_SIZE = 64
LATENT_DIMS = 2
NUM_BATCHES = 50
LR = 3e-4


if __name__ == "__main__":
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    file_path = os.path.join(
        BASE_PATH,
        "version_0/checkpoints/epoch=9999-step=20000.ckpt",
    )

    model = ConvolutionalVariationalAutoencoder.load_from_checkpoint(
        file_path, map_location=torch.device(DEVICE), latent_dims=LATENT_DIMS, lr=LR
    )

    # plot
    plot_latent_2D_convolutional(model, data_loader, 100)
    plot_reconstructed_2D(model, (-8, 4), (-1, 10), 10)
    inference_convolutional(model, data_loader, 10)