from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


results_path = Path(__file__).resolve().parents[1] / "results" / "similarity_results.csv"

# Load the saved similarity results.
if not results_path.exists():
	raise FileNotFoundError(f"Results file not found: {results_path}")

try:
	df = pd.read_csv(results_path)
except EmptyDataError as exc:
	raise ValueError(f"Results file is empty: {results_path}") from exc

required_cols = {"category", "score"}
missing = required_cols - set(df.columns)
if missing:
	raise ValueError(f"Results file missing required columns: {sorted(missing)}")

# Average the score per category.
grouped = df.groupby("category")["score"].mean()

print(grouped)