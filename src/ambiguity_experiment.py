from pathlib import Path

from pandas.errors import EmptyDataError

import fasttext
import pandas as pd
import numpy as np
import os
import sys


model_path = (
	Path(__file__).resolve().parents[1]
	/ "models"
	/ "cc.tr.300.bin"
	/ "cc.tr.300.bin"
)

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading fastText model from: {model_path}")
model = fasttext.load_model(str(model_path))
print("Model loaded.")

# Load the ambiguity word list.
data_path = Path(__file__).resolve().parents[1] / "data" / "ambiguity.csv"
try:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
except EmptyDataError as exc:
    raise ValueError(f"Ambiguity data file is empty: {data_path}") from exc

if df.empty:
    raise ValueError(f"Ambiguity data file has no rows: {data_path}")


# Print nearest neighbors for each target word.
for _, row in df.iterrows():

    word = row["word"]

    print("\n")
    print("=" * 50)
    print("WORD:", word)
    print("=" * 50)

    neighbors = model.get_nearest_neighbors(word)
    

    for score, neighbor in neighbors[:10]:
        print(repr(neighbor), "score:", round(score, 3))
