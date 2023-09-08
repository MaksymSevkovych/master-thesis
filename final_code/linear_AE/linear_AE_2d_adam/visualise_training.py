import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_PATH = f"./final_code/linear_AE/{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}"  # noqa: E501
CSV_PATH = "version_0/metrics.csv"
IMG_NAME = f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_training_progress.png"  # noqa: E501


def plot_training_progress(
    base_path: str = BASE_PATH,
    csv_path: str = CSV_PATH,
    img_name: str = IMG_NAME,
) -> None:
    csv_file_path = os.path.join(base_path, csv_path)
    df = pd.read_csv(csv_file_path)
    df = df.drop("val_loss", axis=1)

    x_1, x_2 = df["epoch"][:2000], df["epoch"][2000:]
    y_1, y_2 = df["train_loss"][:2000], df["train_loss"][2000:]

    fig = plt.figure(figsize=(12, 7))
    sns.set_theme()

    fig.add_subplot(121)
    sns.lineplot(x=x_1, y=y_1)
    sns.lineplot(x=x_1[100::100], y=y_1[100::100])

    fig.add_subplot(122)
    sns.lineplot(x=x_2, y=y_2)
    sns.lineplot(x=x_2[0::300], y=y_2[0::300])

    img_path = os.path.join(base_path, img_name)
    plt.savefig(img_path)

    plt.show()


if __name__ == "__main__":
    plot_training_progress(BASE_PATH, CSV_PATH, IMG_NAME)
