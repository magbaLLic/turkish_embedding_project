import fasttext
import pandas as pd
import numpy as np

# Model yükle
model = fasttext.load_model("../models/cc.tr.300.bin")

# Dataset yükle
df = pd.read_csv("../data/ambiguity_similarity.csv")

# Cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

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

print(results_df)

results_df.to_csv(
    "../results/ambiguity_similarity_results.csv",
    index=False
)

print("\nResults saved!")