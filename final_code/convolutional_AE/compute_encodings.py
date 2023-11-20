import os
import pickle

import pandas as pd
import torch
from isa_conv_AE import ConvolutionalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG

BASE_PATH = os.path.dirname(__file__)
BATCH_SIZE = 1
LATENT_DIMS = 2
LR = 3e-4
MAX_AMOUNT = 5000
ENCODINGS_FILE = "encodings.pickle"


def compute_encodings(model: torch.nn.Module, max_amount: int) -> dict:
    encodings = {}
    for label in range(10):
        encodings_per_digit = []

        pbar = tqdm(total=max_amount)
        for img, lbl in data_loader:
            if len(encodings_per_digit) == max_amount:
                break
            if lbl != label:
                continue
            encodings_per_digit.append(torch.flatten(model.encoder(img).detach()))
            pbar.update(1)

        print(f"completed digit: {label}!")
        encodings.update({label: encodings_per_digit})

    return encodings


def create_dataframes(encodings: dict, max_amount: int) -> dict[pd.DataFrame]:
    dfs = {}

    for label, encodings_per_digit in encodings.items():
        cat = torch.empty((64, 0))
        for enc in encodings_per_digit:
            cat = torch.cat((cat, enc.unsqueeze(1)), 1)
        df = pd.DataFrame(cat)
        df["average"] = torch.stack(encodings_per_digit).sum(0) / max_amount

        dfs.update({f"example {label}": df})

    return dfs


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

    encodings = compute_encodings(model, MAX_AMOUNT)
    dfs = create_dataframes(encodings, MAX_AMOUNT)

    with open(os.path.join(BASE_PATH, ENCODINGS_FILE), "wb") as writer:
        pickle.dump(dfs, writer)
