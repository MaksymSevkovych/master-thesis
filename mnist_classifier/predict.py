# Import dependenciies
import torch
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from train import ImageClassifier

# Global variables
MACHINE = "cpu"  # change to "cuda" if operating on a GPU

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to(MACHINE)


# Train flow
if __name__ == "__main__":
    # Load trained model
    with open("./src/model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    # Make prediction
    img = train[0][0]
    img_tensor = ToTensor()(img).unsqueeze(0).to(MACHINE)

    print(torch.argmax(clf(img_tensor)))
    img.show()
