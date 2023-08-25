import os

import pandas as pd
import plotly.express as px

BASE_PATH = "final_code/linear_AE_amsgrad/"
CSV_PATH = "10000_steps_linear_AE/version_0/metrics.csv"
IMG_NAME = "training_progress.png"


def plot_training_progress(
    base_path: str = BASE_PATH, csv_path: str = CSV_PATH, img_name: str = IMG_NAME
) -> None:
    csv_file_path = os.path.join(BASE_PATH, CSV_PATH)
    df = pd.read_csv(csv_file_path)
    df = df.drop("val_loss", axis=1)

    fig = px.line(df, x="epoch", y="train_loss", title="training progress")

    img_path = os.path.join(BASE_PATH, IMG_NAME)
    fig.write_image(img_path)


if __name__ == "__main__":
    plot_training_progress(BASE_PATH, CSV_PATH, IMG_NAME)
