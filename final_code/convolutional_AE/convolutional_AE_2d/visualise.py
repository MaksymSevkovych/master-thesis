import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FONTSIZE_LATENT = 30
FONTSIZE_INFERENCE = 20


def plot_reconstructed_2D(
    autoencoder: Module,
    r0: tuple[int, int] = (-5, 10),
    r1: tuple[int, int] = (-10, 5),
    n=12,
) -> None:
    w = 28
    img = torch.zeros((n * w, n * w))
    for i, y in enumerate(torch.linspace(*r1, n)):
        for j, x in enumerate(torch.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(DEVICE)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to(DEVICE).detach()
            img[(n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()


def plot_reconstructed_for_point(
    autoencoder: Module,
    point: torch.Tensor,
):
    x_hat = autoencoder.decoder(point)
    x_hat = x_hat.reshape(28, 28).to(DEVICE).detach()
    plt.imshow(x_hat)
    plt.show()


# Plot the latent 3D space
def plot_latent_3D_vae(model: Module, data_loader: DataLoader, num_batches=100) -> None:
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        mu, log_var = model.encoder(img)
        _, _, z = model.sample(mu, log_var)
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
    plt.title("Latent space of encoder", fontsize=FONTSIZE_LATENT)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_latent.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()


def plot_latent3D_single_point(
    autoencoder: Module,
    data_loader: DataLoader,
):
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
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
