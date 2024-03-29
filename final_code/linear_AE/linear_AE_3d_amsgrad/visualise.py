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
def plot_latent_3D_linear(
    autoencoder: Module,
    data_loader: DataLoader,
    num_batches=100,
) -> None:
    # Define the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        img = img.reshape(-1, 28 * 28)
        z = autoencoder.encoder(img)
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
    plt.show()


def plot_latent3D_single_point(
    autoencoder: Module,
    data_loader: DataLoader,
):
    # Define the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    plotted = {}
    for images, labels in data_loader:
        for image, label in zip(images, labels):
            if label.item() in plotted:
                continue
            plotted.update({label.item(): image.tolist()})
            if len(plotted) == 10:
                break

    data = torch.empty(0)
    labels = list(plotted.keys())
    for img in plotted.values():
        # Feed the data into the model
        img = torch.tensor(img).reshape(-1, 28 * 28)
        z = autoencoder.encoder(img).to(DEVICE)
        z = z.to("cpu").detach()
        data = torch.cat((data, z), 0)

    # Data for three-dimensional scattered points
    data = data.numpy()
    xdata = data[:, 0]
    ydata = data[:, 1]
    zdata = data[:, 2]

    # Plot the data
    plot = ax.scatter(xdata, ydata, zdata, c=labels, cmap="tab10")

    # Label the axes, config the plot
    ax.grid(False)
    ax.set_title("One point of each class")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    # Add a colorbar
    fig.colorbar(plot, ax=ax)
    plt.show()


# Plot 10 generated digits
def inference_linear(
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
                img_to_plot.append(img)
                img = img.reshape(-1, 28 * 28)
                img_rec = model(img).to(DEVICE).detach()
                img_rec = img_rec.reshape(28, 28)
                recon.append(img_rec)
        recons.append(recon)
        images.append(img_to_plot)
        recon = []
        img_to_plot = []

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

    plt.imshow(img)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_inference.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()
