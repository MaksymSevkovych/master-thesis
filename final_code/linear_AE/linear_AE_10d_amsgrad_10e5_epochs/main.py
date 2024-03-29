import os

import torch
from isa_linear_AE import LinearAutoEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visualise import inference_linear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG

BASE_PATH = os.path.dirname(__file__)
BATCH_SIZE = 64
LATENT_DIMS = 10
NUM_BATCHES = 50
LR = 3e-4


if __name__ == "__main__":
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    file_path = os.path.join(
        BASE_PATH,
        "version_0/checkpoints/epoch=99999-step=300000.ckpt",
    )

    model = LinearAutoEncoder.load_from_checkpoint(
        file_path, map_location=torch.device(DEVICE), latent_dims=LATENT_DIMS, lr=LR
    )

    # plot
    inference_linear(model, data_loader, 10)
