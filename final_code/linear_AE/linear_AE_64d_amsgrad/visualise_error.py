import os

import matplotlib.pyplot as plt
import torch
from isa_linear_AE import LinearAutoEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = os.path.dirname(__file__)
BATCH_SIZE = 1
LATENT_DIMS = 64
LR = 3e-4


def compute_errors(model: torch.nn.Module, data_loader: DataLoader) -> dict:
    images = []
    for digit in tqdm(range(10)):
        images_for_digit = []
        for img, label in data_loader:
            if label != digit:
                continue
            images_for_digit.append(img)
        images.append(images_for_digit)

    errors = {}
    for index, imgs in enumerate(images):
        error = 0
        for img in tqdm(imgs):
            recon = model(torch.flatten(img))
            recon = recon.reshape(28, 28)
            error += torch.sqrt(
                torch.pow(torch.sub(img.detach(), recon.detach()), 2).sum()
            )

        error = error / len(imgs)
        print(f"The error for digit {index} is: {error}")
        errors.update({index: error})
    return errors


def plot_errors(
    errors: dict,
) -> None:
    for x, y in errors.items():
        plt.bar(x, y)

    plt.xticks(list(range(10)))
    plt.set_cmap("tab10")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_errors.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))

    plt.show()


if __name__ == "__main__":
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    file_path = os.path.join(
        BASE_PATH,
        "version_0/checkpoints/epoch=9999-step=30000.ckpt",
    )

    model = LinearAutoEncoder.load_from_checkpoint(
        file_path, map_location=torch.device(DEVICE), latent_dims=LATENT_DIMS, lr=LR
    )

    plot_errors(compute_errors(model, data_loader))
