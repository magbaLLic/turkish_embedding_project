from pathlib import Path

import fasttext
import numpy as np

model_path = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cc.tr.300.bin"
    / "cc.tr.300.bin"
)

model = fasttext.load_model(str(model_path))

# Compare two sample words.
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

v1 = model.get_word_vector("kedi")
v2 = model.get_word_vector("masa")

score = cosine_similarity(v1, v2)

# Print the sample score.
print(score)