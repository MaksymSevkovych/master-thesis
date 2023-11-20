import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

BASE_PATH = os.path.dirname(__file__)
ENCODINGS_FILE = "encodings.pickle"

with open(os.path.join(BASE_PATH, ENCODINGS_FILE), "rb") as reader:
    dfs = pickle.load(reader)

for df in dfs.values():
    for i in range(20):
        df.rename(columns={i: f"example {i}"}, inplace=True)


averages = {}
for i in range(10):
    averages.update({f"digit: {i}": dfs[f"example {i}"]["average"]})

df_averages = pd.DataFrame(averages)
df_averages.plot(
    y=list(range(5)),
    kind="bar",
    figsize=(14, 7),
)
dir_path = os.path.dirname(os.path.realpath(__file__))
file_name = (
    f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_latent_0.png"
)
plt.savefig(os.path.join(dir_path, file_name))
plt.show()

df_averages.plot(
    y=list(range(5, 10)),
    kind="bar",
    figsize=(14, 7),
)
dir_path = os.path.dirname(os.path.realpath(__file__))
file_name = (
    f"{os.path.basename(os.path.dirname(os.path.realpath(__file__)))}_latent_1.png"
)
plt.savefig(os.path.join(dir_path, file_name))
plt.show()
