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


# Plot the latent 3D space
def plot_latent_3D_convolutional(
    model: Module, data_loader: DataLoader, num_batches=100
) -> None:
    # Define the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        mu, log_var = model.encoder(img)
        _, _, z = model.sample(label, mu, log_var)
        z = z.to("cpu").detach().numpy()

        # Data for three-dimensional scattered points
        xdata = z[:, 0]
        ydata = z[:, 1]
        zdata = z[:, 2]

        # Plot the data
        plot = ax.scatter(xdata, ydata, zdata, c=label, cmap="tab10", marker="o")

        # Label the axes, config the plot
        ax.grid(False)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    # plt.title("Latent space of encoder", fontsize=FONTSIZE_LATENT)
    plt.show()


# Plot 10 generated digits
def inference_convolutional(
    model: Module,
    data_loader: DataLoader,
    amount: int,
) -> None:
    images = []
    recons = []
    for digit in range(10):
        images_for_digit = []
        reconstructions_for_digit = []
        for imgs, labels in data_loader:
            mu, log_var = model.encoder(imgs)
            if len(reconstructions_for_digit) == 10:
                break
            for index, (img, label) in enumerate(zip(imgs, labels)):
                if label != digit:
                    continue

                images_for_digit.append(img)
                reconstructions_for_digit.append(
                    model.decoder(
                        model.sample(
                            labels, mu[index].detach(), log_var[index].detach()
                        )[2].unsqueeze(0)
                    )
                )
                if len(reconstructions_for_digit) == 10:
                    recons.append(reconstructions_for_digit)
                    images.append(images_for_digit)
                    break
    width = 28

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
            ] = recon.detach()
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_inference.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()
