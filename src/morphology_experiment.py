from pathlib import Path

from pandas.errors import EmptyDataError
import pandas as pd

data_path = Path(__file__).resolve().parents[1] / "data" / "morphology.csv"

if not data_path.exists():
	raise FileNotFoundError(f"Morphology data file not found: {data_path}")

try:
	df = pd.read_csv(data_path)
except EmptyDataError:
	raise ValueError(f"Morphology data file is empty: {data_path}")

print(f"Loaded morphology data: {len(df)} rows")
print("morphology_experiment not implemented yet")
