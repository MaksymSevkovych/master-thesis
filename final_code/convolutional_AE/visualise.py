import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_reconstructed_for_point(
    autoencoder: Module,
    point: torch.Tensor,
):
    x_hat = autoencoder.decoder(point)
    x_hat = x_hat.reshape(28, 28).to(DEVICE).detach()
    plt.imshow(x_hat)
    plt.show()


# Plot 10 generated digits
def inference_convolutional_ae(
    model: Module,
    data_loader: DataLoader,
    amount: int,
) -> None:
    width = 28
    images, recons = [], []
    recon = []
    img_to_plot = []

    for num in range(10):
        for imgs, labels in data_loader:
            if len(recon) >= 10:
                continue
            for img, label in zip(imgs, labels):
                if len(recon) >= 10:
                    continue
                if label != num:
                    continue
                img_rec = model(img).to(DEVICE).detach()
                recon.append(img_rec)
                img_to_plot.append(img)
        recons.append(recon)
        images.append(img_to_plot)
        recon, img_to_plot = [], []

    fig = plt.figure(figsize=(14, 7))
    fig.add_subplot(121)
    img = torch.zeros((amount * width, amount * width))

    for i, imgs in enumerate(images):
        for j, image in enumerate(imgs):
            img[
                (amount - 1 - i) * width : (amount - 1 - i + 1) * width,
                j * width : (j + 1) * width,
            ] = image
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    fig.add_subplot(122)
    img = torch.zeros((amount * width, amount * width))

    for i, recs in enumerate(recons):
        for j, recon in enumerate(recs):
            img[
                (amount - 1 - i) * width : (amount - 1 - i + 1) * width,
                j * width : (j + 1) * width,
            ] = recon
    plt.xticks([])
    plt.yticks([])
    # plt.title("Inference of autoencoder", fontsize=FONTSIZE_INFERENCE)
    plt.imshow(img)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_inference.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()


# Plot 10 generated digits
def generate_convolutional(
    model: Module,
    data_loader: DataLoader,
    amount: int,
) -> None:
    recons = []
    for digit in range(10):
        reconstructions_for_digit = []
        for imgs, labels in data_loader:
            if len(reconstructions_for_digit) == amount:
                break
            for index, (img, label) in enumerate(zip(imgs, labels)):
                if label != digit:
                    continue

                encoded = model.encoder(img).flatten(-1)
                nudged = [entry[0][0] + 5 * torch.rand(1) for entry in encoded.detach()]
                encoded = torch.tensor(nudged)

                encoded = encoded.unsqueeze(1).unsqueeze(1)

                reconstructions_for_digit.append(model.decoder(encoded))
                if len(reconstructions_for_digit) == amount:
                    recons.append(reconstructions_for_digit)

                    break
    width = 28

    plt.figure(figsize=(7, 5))
    img = torch.zeros((amount * width, amount * width))

    for i, recs in enumerate(recons):
        for j, recon in enumerate(recs):
            img[
                (amount - 1 - i) * width : (amount - 1 - i + 1) * width,
                j * width : (j + 1) * width,
            ] = recon.detach()
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_generated.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()
