# Import dependenciies
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

# Global variables
MACHINE = "cpu"  # change to "cuda" if operating on a GPU

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, (3, 3)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to(MACHINE)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# Train flow
if __name__ == "__main__":
    for epoch in tqdm(range(10)):  # train for 10 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to(MACHINE), y.to(MACHINE)
            y_hat = clf(X)
            loss = loss_fn(y_hat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} loss is {loss.item()}")

    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)
