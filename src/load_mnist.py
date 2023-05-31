# import torch
import torchvision.datasets as datasets


def visualize_data(images: list) -> None:
    return


mnist_trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=None
)

images = [mnist_trainset[i] for i in range(10)]
