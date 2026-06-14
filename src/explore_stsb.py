import io
from urllib.request import urlopen

import pandas as pd

STS_URLS = {
    "train": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-train-tr.csv"
}

# Download the STSB sample set.
with urlopen(STS_URLS["train"], timeout=30) as response:
    dataset_file = response.read()

df = pd.read_csv(
    io.StringIO(
        dataset_file.decode("utf-8")
    )
)

# Inspect the dataset shape and columns.
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)