from pathlib import Path

import fasttext
import numpy as np
import pandas as pd

# -------------------------
# Load model
# -------------------------

model_path = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cc.tr.300.bin"
    / "cc.tr.300.bin"
)

model = fasttext.load_model(str(model_path))

# -------------------------
# Load dataset
# -------------------------

data_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "analogies.csv"
)

try:
    df = pd.read_csv(data_path)
except Exception as exc:
    raise ValueError(f"Analogies data file error: {data_path}") from exc

if df.empty:
    raise ValueError(f"Analogies data file has no rows: {data_path}")

# -------------------------
# Cosine similarity
# -------------------------

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# -------------------------
# Analogy
# -------------------------

for _, row in df.iterrows():

    a = row["a"]
    b = row["b"]
    c = row["c"]
    expected = row["expected"]

    target_vector = (
        model.get_word_vector(a)
        - model.get_word_vector(b)
        + model.get_word_vector(c)
    )

    expected_vector = model.get_word_vector(expected)

    score = cosine_similarity(
        target_vector,
        expected_vector
    )

    print("\n")
    print("=" * 60)
    print(f"{a} - {b} + {c}")
    print(f"Expected: {expected}")
    print(f"Similarity: {score:.4f}")