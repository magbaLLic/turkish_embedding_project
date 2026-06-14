from pathlib import Path

import fasttext

model_path = (
	Path(__file__).resolve().parents[1]
	/ "models"
	/ "cc.tr.300.bin"
	/ "cc.tr.300.bin"
)
model = fasttext.load_model(str(model_path))

# Inspect a few nearest neighbors.
neighbors = model.get_nearest_neighbors("kedi")

print(neighbors)