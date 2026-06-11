from pathlib import Path

from pandas.errors import EmptyDataError

import fasttext
import pandas as pd
import numpy as np

model_path = (
	Path(__file__).resolve().parents[1]
	/ "models"
	/ "cc.tr.300.bin"
	/ "cc.tr.300.bin"
)

model = fasttext.load_model(str(model_path))

data_path = Path(__file__).resolve().parents[1] / "data" / "ambiguity.csv"
try:
    df = pd.read_csv(data_path)
except EmptyDataError as exc:
    raise ValueError(f"Ambiguity data file is empty: {data_path}") from exc

if df.empty:
    raise ValueError(f"Ambiguity data file has no rows: {data_path}")


for _, row in df.iterrows():

    word = row["word"]

    print("\n")
    print("=" * 50)
    print("WORD:", word)
    print("=" * 50)

    neighbors = model.get_nearest_neighbors(word)

    for score, neighbor in neighbors[:10]:
        print(neighbor, round(score, 3))
