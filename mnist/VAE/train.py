from random import seed

import torch
import torch.optim as optim
from modules import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, data_loader, num_epochs=10):
    outputs = []
    bce_fn = torch.nn.BCELoss(reduction="sum")

    for epoch in range(num_epochs):
        for img, _ in data_loader:
            recon = model(img)
            # loss = ((img - recon) ** 2).sum() + model.encoder.kl
            bce = bce_fn(recon, img)
            kl = model.sampler.kl

            print(f"Epoch: {epoch+1}, BCE: {bce}, KL: {kl}")
            loss = 0.7 * bce + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        outputs.append((epoch, img, recon))
    return outputs, model


if __name__ == "__main__":
    # config
    LATENT_DIMS = 3
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    SEED = 0

    # seed
    seed(SEED)
    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    vae = ConvolutionalVariationalAutoencoder(LATENT_DIMS)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    outputs, vae = train(
        vae, optimizer=optimizer, data_loader=data_loader, num_epochs=NUM_EPOCHS
    )

    with open(
        f"./master-thesis/mnist/VAE/conv_vae_{NUM_EPOCHS}_epochs_{LATENT_DIMS}_dims_{LEARNING_RATE}_lr.pt",
        "wb",
    ) as f:
        torch.save(vae.state_dict(), f)
