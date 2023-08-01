import os

import torch
from isa29.isa_runner import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualisations import inference, plot_latent_3D_convolutional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # config
    LATENT_DIMS = 3
    NUM_EPOCHS = 5000
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    KL_COEFF = 1
    FILE_PATH = "./master-thesis/mnist/VAE/binaries"
    FILE_NAME = f"conv_vae_{NUM_EPOCHS}_epochs_{LATENT_DIMS}_dims_{LEARNING_RATE}_lr_{KL_COEFF}_alpha.pt"  # noqa: E501

    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    file_path = os.path.join(
        FILE_PATH,
        "100000_steps_new/version_0/checkpoints",
        "epoch=99999-step=200000.ckpt",
    )

    vae = ConvolutionalVariationalAutoencoder.load_from_checkpoint(
        file_path, map_location=torch.device(DEVICE), latent_dims=LATENT_DIMS
    )

    # plot_latent_2D_convolutional(vae, data_loader, num_batches=50)
    # plot_reconstructed_2D(vae, (-1.5, 1.5), (-1.5, 1.5))
    plot_latent_3D_convolutional(vae, data_loader, 80)

    # point = torch.tensor([[1, 0.4, 0.4]], dtype=torch.float)
    # plot_reconstructed_for_point(vae, point)

    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)

    inference(vae, data_loader, 10)
