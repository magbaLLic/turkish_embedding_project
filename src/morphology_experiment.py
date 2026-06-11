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

data_path = Path(__file__).resolve().parents[1] / "data" / "morphology.csv"
try:
    df = pd.read_csv(data_path)
except EmptyDataError as exc:
    raise ValueError(f"Morphology data file is empty: {data_path}") from exc

if df.empty:
    raise ValueError(f"Morphology data file has no rows: {data_path}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )


results = []

for _, row in df.iterrows():

    base = row["base"]
    derived = row["derived"]

    base_vec = model.get_word_vector(base)
    derived_vec = model.get_word_vector(derived)

    score = cosine_similarity(base_vec, derived_vec)

    results.append({
        "base": base,
        "derived": derived,
        "score": score
    })

results_df = pd.DataFrame(results)

print(results_df)


output_path = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "morphology_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)

print("\nAverage similarity:")
print(results_df["score"].mean())