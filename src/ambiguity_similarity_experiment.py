from pathlib import Path

from pandas.errors import EmptyDataError

import fasttext
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "cc.tr.300.bin" / "cc.tr.300.bin"
data_path = project_root / "data" / "ambiguity_similarity.csv"
results_path = project_root / "results" / "ambiguity_similarity_results.csv"

model = fasttext.load_model(str(model_path))

# Load the ambiguity similarity pairs.
try:
    df = pd.read_csv(data_path)
except EmptyDataError as exc:
    raise ValueError(f"Ambiguity similarity data file is empty: {data_path}") from exc

if df.empty:
    raise ValueError(f"Ambiguity similarity data file has no rows: {data_path}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# Compute similarity scores for both meanings.
results = []

for _, row in df.iterrows():

    target = row["target"]
    meaning1 = row["meaning1"]
    meaning2 = row["meaning2"]

    target_vec = model.get_word_vector(target)

    m1_vec = model.get_word_vector(meaning1)
    m2_vec = model.get_word_vector(meaning2)

    sim1 = cosine_similarity(target_vec, m1_vec)
    sim2 = cosine_similarity(target_vec, m2_vec)

    results.append({
        "target": target,
        "meaning1": meaning1,
        "meaning1_score": sim1,
        "meaning2": meaning2,
        "meaning2_score": sim2
    })

results_df = pd.DataFrame(results)

results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_path, index=False)

# Save the final table.
print(f"Results saved to: {results_path}")