import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

BASE_PATH = "/Users/maksym/Uni/master/coding/master-thesis/final_code/convolutional_AE/convolutional_AE_2d"  # noqa: E501
ENCODINGS_FILE = "encodings.pickle"

with open(os.path.join(BASE_PATH, ENCODINGS_FILE), "rb") as reader:
    dfs = pickle.load(reader)

for df in dfs.values():
    for i in range(20):
        df.rename(columns={i: f"example {i}"}, inplace=True)

# for label, df in dfs.items():
#     for index in range(1):
#         fig = plt.figure(figsize=(14, 7))

#         ax = fig.add_subplot()
#         df.plot(
#             y=[f"example {index}", "average"],
#             kind="bar",
#             # xticks=list(range(64)),
#             ax=ax,
#             title=f"Label: {label}",
#         )
#         plt.show()

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
