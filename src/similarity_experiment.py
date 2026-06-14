
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

# Load the similarity benchmark pairs.
data_path = Path(__file__).resolve().parents[1] / "data" / "similarity.csv"
results_path = Path(__file__).resolve().parents[1] / "results" / "similarity_results.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Similarity data file not found: {data_path}")

try:
    df = pd.read_csv(data_path)
except EmptyDataError as exc:
    raise ValueError(f"Similarity data file is empty: {data_path}") from exc


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# Score each word pair.
results = []

for _, row in df.iterrows():

    word1 = row["word1"]
    word2 = row["word2"]
    category = row["category"]

    v1 = model.get_word_vector(word1)
    v2 = model.get_word_vector(word2)

    score = cosine_similarity(v1, v2)

    results.append({
        "word1": word1,
        "word2": word2,
        "category": category,
        "score": score
    })

results_df = pd.DataFrame(results)


results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(str(results_path), index=False)

# Save the final CSV.
print(f"Results saved to: {results_path}")