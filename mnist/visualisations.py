import matplotlib.pyplot as plt
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_latent_2D_linear(autoencoder, data_loader, num_batches=150):
    for i, (img, label) in enumerate(data_loader):
        img = img.reshape(-1, 28 * 28)
        z = autoencoder.encoder(img.to(DEVICE))
        z = z.to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=label, cmap="tab10", marker=".")
        if i > num_batches:
            plt.colorbar()
            break


def plot_reconstructed_2D(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
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
