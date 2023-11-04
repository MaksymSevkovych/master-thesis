import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Plot the latent 2D space
def plot_latent_2D_linear(autoencoder, data_loader, num_batches=150):
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
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
    plt.show()


# Plot the latent 2D space
def plot_latent_2D_convolutional(autoencoder, data_loader, num_batches=150):
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
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
    plt.show()


def plot_reconstructed_2D(
    autoencoder: Module,
    r0: tuple[int, int] = (-5, 10),
    r1: tuple[int, int] = (-10, 5),
    n=12,
):
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
def plot_latent3D(autoencoder, data_loader, num_batches=100):
    # Define the figure
    fig = plt.figure(figsize=(12, 7))
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
        ax.set_title("Encoder Output")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    plt.show()


# Plot the latent 3D space
def plot_latent_3D_convolutional(model, data_loader, num_batches=100):
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
        ax.set_title("Encoder Output")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

        # Add a colorbar
        if i > num_batches:
            fig.colorbar(plot, ax=ax)
            break
    plt.show()


def plot_latent3D_single_point(autoencoder, data_loader):
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

    print(label)
    # Plot the data
    plot = ax.scatter(xdata, ydata, zdata, c=labels, cmap="tab10")  # , marker="O")

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
def inference(model: Module, data_loader: DataLoader, amount: int) -> None:
    generating_data = {}
    for img, label in data_loader:
        if len(generating_data) == 10:
            break

        if label in generating_data:
            continue

        mu, log_var = model.encoder(img)

        generating_data.update(
            {
                label.item(): {
                    "mu": mu.detach(),
                    "log_var": log_var.detach(),
                }
            }
        )

    generating_data_sorted = {
        label: generating_data[label] for label in list(range(10))
    }

    width = 28
    img = torch.zeros((amount * width, amount * width))

    for i, params in generating_data_sorted.items():
        mu, log_var = params.values()
        samples = []
        for _ in range(10):
            _, _, sample = model.sample(mu, log_var)
            samples.append(sample)
        # samples = [model.sample(mu, log_var)[2] for _ in range(10)]
        recons = [model.decoder(sample) for sample in samples]
        recons = [recon.reshape(28, 28).to(DEVICE).detach() for recon in recons]
        for j, recon in enumerate(recons):
            img[
                (amount - 1 - i) * width : (amount - 1 - i + 1) * width,
                j * width : (j + 1) * width,
            ] = recon
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
