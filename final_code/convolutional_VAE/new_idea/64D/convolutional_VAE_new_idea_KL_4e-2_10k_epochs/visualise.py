import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FONTSIZE_LATENT = 30
FONTSIZE_RECONSTRUCTION = 30
FONTSIZE_INFERENCE = 20


# Plot the latent 2D space
def plot_latent_2D_linear(
    autoencoder: Module,
    data_loader: DataLoader,
    num_batches=150,
) -> None:
    # Define the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        img = img.reshape(-1, 28 * 28)
        z = autoencoder.encoder(img.to(DEVICE))
        z = z.to("cpu").detach().numpy()

        # Data for two-dimensional scattered points
        xdata = z[:, 0]
        ydata = z[:, 1]

        # Plot the data
        plot = ax.scatter(xdata, ydata, c=label, cmap="tab10", marker="o")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    # plt.title("Latent space of encoder", fontsize=FONTSIZE_LATENT)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_latent.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()


# Plot the latent 2D space
def plot_latent_2D_convolutional(
    autoencoder: Module,
    data_loader: DataLoader,
    num_batches=150,
) -> None:
    # Define the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i, (img, label) in enumerate(data_loader):
        # Feed the data into the model
        z = autoencoder.encoder(img.to(DEVICE))
        z = z.to("cpu").detach().numpy()

        # Data for two-dimensional scattered points
        xdata = z[:, 0]
        ydata = z[:, 1]

        # Plot the data
        plot = ax.scatter(xdata, ydata, c=label, cmap="tab10", marker="o")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    # plt.title("Latent space of encoder", fontsize=FONTSIZE_LATENT)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_latent.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()


def plot_reconstructed_2D(
    autoencoder: Module,
    r0: tuple[int, int] = (-5, 10),
    r1: tuple[int, int] = (-10, 5),
    n=10,
) -> None:
    plt.figure(figsize=(7, 5))
    w = 28
    img = torch.zeros((n * w, n * w))
    for i, y in enumerate(torch.linspace(*r1, n)):
        for j, x in enumerate(torch.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(DEVICE)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to(DEVICE).detach()
            img[(n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    # plt.title("Reconstruction of latent space", fontsize=FONTSIZE_RECONSTRUCTION)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_reconstruction.png"  # noqa: E501
    plt.savefig(os.path.join(dir_path, file_name))
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
    # plt.title("Latent space of encoder", fontsize=FONTSIZE_LATENT)
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


# Plot 10 generated digits
def inference_linear(
    model: Module,
    data_loader: DataLoader,
    amount: int,
) -> None:
    width = 28
    recons = []
    recon = []

    for num in range(10):
        for imgs, labels in data_loader:
            if len(recon) >= 10:
                continue
            for img, label in zip(imgs, labels):
                if len(recon) >= 10:
                    continue
                if label != num:
                    continue
                img = img.reshape(-1, 28 * 28)
                img_rec = model(img).to(DEVICE).detach()
                img_rec = img_rec.reshape(28, 28)
                recon.append(img_rec)
        recons.append(recon)
        recon = []

    img = torch.zeros((amount * width, amount * width))

    for i, recs in enumerate(recons):
        for j, recon in enumerate(recs):
            img[
                (amount - 1 - i) * width : (amount - 1 - i + 1) * width,
                j * width : (j + 1) * width,
            ] = recon
    # plt.title("Inference of autoencoder", fontsize=FONTSIZE_INFERENCE)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = (
        f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_inference.png"
    )
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()


def generate_convolutional_optimal(
    model: Module,
    data_loader: DataLoader,
    latent_dims: int,
    amount: int,
) -> None:
    parameters = {}

    for digit in tqdm(range(10)):
        mean_mu, mean_log_var, counter = (
            torch.zeros(latent_dims),
            torch.zeros(latent_dims),
            0,
        )
        for imgs, labels in data_loader:
            mu, log_var = model.encoder(imgs)
            for index, label in enumerate(labels):
                if label != digit:
                    continue

                mean_mu += mu[index].detach()
                mean_log_var += log_var[index].detach()
                counter += 1

        parameters.update(
            {digit: {"mean": mean_mu / counter, "log_var": mean_log_var / counter}}
        )

    recons = []
    for digit in range(10):
        reconstructions_for_digit = []
        for _ in range(10):
            mu, log_var = parameters[digit]["mean"], parameters[digit]["log_var"]
            mu_new = mu + 0  # torch.rand(latent_dims)

            reconstructions_for_digit.append(
                model.decoder(
                    model.sample([0] * latent_dims, mu_new, log_var)[2].unsqueeze(0)
                )
            )

        recons.append(reconstructions_for_digit)
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
    file_name = f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_generated_optimal.png"  # noqa: E501
    plt.savefig(os.path.join(dir_path, file_name))
    plt.show()
