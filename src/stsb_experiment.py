from pathlib import Path
import io
import re
from urllib.request import urlopen

import fasttext
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# DATASET

STS_URL = (
    "https://raw.githubusercontent.com/"
    "emrecncelik/sts-benchmark-tr/main/sts-train-tr.csv"
)

print("Downloading STSB Turkish dataset...")

with urlopen(STS_URL, timeout=30) as response:
    dataset_file = response.read()

df = pd.read_csv(
    io.StringIO(
        dataset_file.decode("utf-8")
    )
)

print(f"Dataset loaded: {len(df)} rows")

print(len(df))
# MODEL

model_path = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cc.tr.300.bin"
    / "cc.tr.300.bin"
)

print("Loading fastText model...")
model = fasttext.load_model(str(model_path))
print("Model loaded.")


# HELPERS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )


def tokenize(sentence):
    return re.findall(
        r"\w+",
        str(sentence).lower(),
        flags=re.UNICODE
    )


def sentence_embedding(sentence):

    words = tokenize(sentence)

    if not words:
        return np.zeros(300)

    vectors = [
        model.get_word_vector(word)
        for word in words
    ]

    return np.mean(vectors, axis=0)


# 
# EXPERIMENT
# 

results = []

for idx, row in df.iterrows():

    sent1 = row["sentence1_tr"]
    sent2 = row["sentence2_tr"]

    human_score = row["score"]

    emb1 = sentence_embedding(sent1)
    emb2 = sentence_embedding(sent2)

    model_score = cosine_similarity(
        emb1,
        emb2
    )

    results.append({
        "human_score": human_score,
        "model_score": model_score,
        "sentence1": sent1,
        "sentence2": sent2
    })

    if idx % 500 == 0:
        print(f"Processed {idx} rows...")


results_df = pd.DataFrame(results)


# SAVE RESULTS

output_path = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "stsb_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(f"\nResults saved to:\n{output_path}")


# CORRELATION

correlation, p_value = spearmanr(
    results_df["human_score"],
    results_df["model_score"]
)

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print(f"Spearman Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.8f}")


# SAMPLE OUTPUT

print("\nSample predictions:\n")

print(
    results_df[
        ["human_score", "model_score"]
    ].head(1000).to_string(index=False)
)