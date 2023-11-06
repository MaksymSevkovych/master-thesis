import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_PATH = os.path.dirname(__file__)
CSV_PATH = "version_0/metrics.csv"
IMG_NAME = f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_training_progress.png"  # noqa: E501


def sanitize_data(arr: list) -> list:
    sanitized = []
    for loss in arr:
        if np.isnan(loss):
            sanitized.append(sanitized[-1])
            continue
        sanitized.append(loss)

    return sanitized


def calc_moving_average(arr: list, window_size: int) -> list:
    arr = sanitize_data(arr)
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    while i < len(arr):
        if i in range(window_size):
            window = arr[0 : i + window_size + 1]
            window_average = sum(window) / len(window)
            moving_averages.append(window_average)
            i += 1
            continue

        if i in range(len(arr) - window_size, len(arr)):
            window = arr[i - window_size :]
            window_average = sum(window) / len(window)
            moving_averages.append(window_average)
            i += 1
            continue

        window = arr[i - window_size : i + window_size + 1]
        window_average = sum(window) / len(window)

        moving_averages.append(window_average)

        i += 1

    return moving_averages


def plot_training_progress(
    base_path: str = BASE_PATH,
    csv_path: str = CSV_PATH,
    img_name: str = IMG_NAME,
) -> None:
    csv_file_path = os.path.join(base_path, csv_path)
    df = pd.read_csv(csv_file_path)
    df = df.drop("val_loss", axis=1)
    df["moving average"] = calc_moving_average(df["train_loss"], 50)

    x_1, x_2 = df["epoch"][:2000], df["epoch"][2000:]
    y_1, y_2 = df["train_loss"][:2000], df["train_loss"][2000:]
    avg_1, avg_2 = df["moving average"][:2000], df["moving average"][2000:]

    fig = plt.figure(figsize=(14, 7))
    sns.set_theme()

    fig.add_subplot(121)
    sns.lineplot(x=x_1, y=y_1)
    sns.lineplot(x=x_1, y=avg_1)

    fig.add_subplot(122)
    sns.lineplot(x=x_2, y=y_2)
    sns.lineplot(x=x_2, y=avg_2)

    img_path = os.path.join(base_path, img_name)
    plt.savefig(img_path)

    plt.show()


if __name__ == "__main__":
    plot_training_progress(BASE_PATH, CSV_PATH, IMG_NAME)
