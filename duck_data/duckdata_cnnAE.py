# Import dependencies
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# Get Duckeneers Data locally

base_path = (
    "/Users/maksym/git/duckeneers/palettscan/setup/Data/datasets/card_detection/data"
)
dataset_dict = load_dataset(
    "imagefolder",
    data_files={
        "train": os.path.join(base_path, "train", "**"),
        "test": os.path.join(base_path, "test", "**"),
        "valid": os.path.join(base_path, "valid", "**"),
    },
)


# Define a linear AutoEncoder
class LinearAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N (batch size), 2064*3088
        self.encoder = nn.Sequential(
            nn.Linear(2064 * 3088, 2**20),  # N, 20264*3088 -> N, 2**20
            nn.ReLU(),
            nn.Linear(2**20, 2**14),
            nn.ReLU(),
            nn.Linear(2**14, 2**7),
            nn.ReLU(),
            nn.Linear(128, 20),  # N, 20
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 128),  # N, 20
            nn.ReLU(),
            nn.Linear(2**7, 2**14),
            nn.ReLU(),
            nn.Linear(2**14, 2**20),
            nn.ReLU(),
            nn.Linear(2**20, 2064 * 3088),  # N, 784
            nn.Sigmoid(),  # IMPORTTANT! Depending on data we might need different activation here!  # noqa: E501
        )

    # NOTE: Last activation: [0, 1] -> nn.ReLU(), [-1, 1] -> nn.Tanh

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Get data
    transform = transforms.ToTensor()
    images = [transform(sample["image"]) for sample in tqdm(dataset_dict["train"])]
    print(torch.min(images[0]), torch.max(images[0]))

    # Instantiate neural net and optimizer
    model = LinearAutoEncoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the Autoencoder
    num_epochs = 10
    outputs = []
    for epoch in range(num_epochs):
        for img in images:
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))

    # Plot the reconstructed images
    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()

        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])
