import torch
import torch.optim as optim
from modules import ConvolutionalVariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, data_loader, num_epochs=10):
    outputs = []

    for epoch in range(num_epochs):
        for img, _ in data_loader:
            optimizer.zero_grad()

            recon = model(img)
            loss = ((img - recon) ** 2).sum() + model.encoder.kl

            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))
    return outputs, model


if __name__ == "__main__":
    # config
    latent_dims = 3
    num_epochs = 40

    # data
    transform = transforms.ToTensor()
    data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    data_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)

    vae = ConvolutionalVariationalAutoencoder(latent_dims)
    optimizer = optim.Adam(vae.parameters(), lr=1e-5)

    outputs, vae = train(
        vae, optimizer=optimizer, data_loader=data_loader, num_epochs=num_epochs
    )

    with open(
        f"./master-thesis/mnist/VAE/conv_vae_{num_epochs}_epochs_{latent_dims}_dims.pt",
        "wb",
    ) as f:
        torch.save(vae.state_dict(), f)
