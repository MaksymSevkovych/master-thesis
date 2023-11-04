import os

import torch
from isa_conv_AE import ConvolutionalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG

BASE_PATH = "/Users/maksym/Uni/master/coding/master-thesis/final_code/convolutional_AE/convolutional_AE_2d"  # noqa: E501
BATCH_SIZE = 1
LATENT_DIMS = 2
LR = 3e-4
MAX_AMOUNT = 2
ENCODINGS_FILE = "encodings.pickle"

if __name__ == "__main__":
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    file_path = os.path.join(
        BASE_PATH,
        "version_0/checkpoints/epoch=9999-step=30000.ckpt",
    )

    model = ConvolutionalAutoencoder.load_from_checkpoint(
        file_path, map_location=torch.device(DEVICE), latent_dims=LATENT_DIMS, lr=LR
    )

    for img, label in data_loader:
        if torch.flatten(model.encoder(img).detach())[1] <= 0:
            print(torch.flatten(model.encoder(img).detach())[1])
