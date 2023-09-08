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


values = {
    "first": dfs[0]["average"],
    "second": dfs[1]["average"],
    "third": dfs[2]["average"],
}
test = pd.DataFrame(values)
test.plot(y=["first", "second", "third"], kind="bar", figsize=(14, 7))
# sns.barplot(x=list(range(64)), y=[avg1.tolist(), avg2.tolist()])
plt.show()
