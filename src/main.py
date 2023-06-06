import os

import torchvision.datasets as datasets

from visualise import generate_plots

if __name__ == "__main__":
    visualisation_path = "./visualisations"

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )

    amount_of_images = 24
    images = [mnist_trainset[i][0] for i in range(amount_of_images)]
    labels = [mnist_trainset[i][1] for i in range(amount_of_images)]

    if not os.path.exists(visualisation_path):
        os.makedirs(visualisation_path)

    generate_plots(images, labels, visualisation_path)
