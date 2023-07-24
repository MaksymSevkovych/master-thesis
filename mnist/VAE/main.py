import os
from random import seed

import torch
from modules import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualisations import inference, plot_latent_3D_convolutional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # config
    LATENT_DIMS = 3
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    SEED = 0
    ALPHA = 0.5
    FILE_PATH = "./master-thesis/mnist/VAE"
    FILE_NAME = f"conv_vae_{NUM_EPOCHS}_epochs_{LATENT_DIMS}_dims_{LEARNING_RATE}_lr_{ALPHA}_alpha.pt"  # noqa: E501

    seed(SEED)
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    vae = ConvolutionalVariationalAutoencoder(LATENT_DIMS)
    with open(
        os.path.join(FILE_PATH, FILE_NAME),
        "rb",
    ) as f:
        vae.load_state_dict(torch.load(f))

    # print(
    #     f"VAE parameters: mu: {vae.encoder.mu.detach()}, sigma: {vae.encoder.sigma.detach()}" # noqa: E501
    # )
    # plot_latent_2D_convolutional(vae, data_loader, num_batches=50)
    # plot_reconstructed_2D(vae, (-1.5, 1.5), (-1.5, 1.5))
    plot_latent_3D_convolutional(vae, data_loader, 80)

    # point = torch.tensor([[1, 0.4, 0.4]], dtype=torch.float)
    # plot_reconstructed_for_point(vae, point)

    # INFERENCE: find a sample for each of the 10 classes
    # -> encode each sample
    # -> use the encodings to generate new samples
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    inference(vae, data_loader, 10)
