from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


sns.set_theme(
	style="whitegrid",
	context="talk",
	palette="colorblind"
)

# Use a clean, light chart style.
plt.rcParams.update({
	"figure.facecolor": "white",
	"axes.facecolor": "#fbfcff",
	"axes.edgecolor": "#d7dde6",
	"axes.titleweight": "bold",
	"axes.labelweight": "medium",
	"grid.color": "#dde3ec",
	"grid.linewidth": 0.8,
	"font.family": "DejaVu Sans",
})


project_root = Path(__file__).resolve().parents[1]
results_path = project_root / "results" / "stsb_results.csv"

if not results_path.exists():
	raise FileNotFoundError(
		f"STSB results file not found: {results_path}. Run stsb_experiment.py first."
	)

results_df = pd.read_csv(results_path)

required_columns = {"human_score", "model_score"}
missing_columns = required_columns - set(results_df.columns)

if missing_columns:
	raise ValueError(
		f"STSB results file is missing required columns: {sorted(missing_columns)}"
	)

results_dir = results_path.parent

model_hist_color = "#4c78a8"
human_hist_color = "#f58518"


# Keep repeated styling in one place.
def finalize_plot(ax):
	ax.tick_params(axis="both", labelsize=11)
	sns.despine(ax=ax, offset=5)
	return ax

# Compare human and model scores.
fig, ax = plt.subplots(figsize=(8, 6))

hb = ax.hist2d(
	results_df["human_score"],
	results_df["model_score"],
	bins=30,
	cmap="viridis",
	cmin=1
)

sns.regplot(
	data=results_df,
	x="human_score",
	y="model_score",
	scatter=False,
	ax=ax,
	color="#7aa6c2",
	line_kws={"linewidth": 2.0, "alpha": 0.85}
)

colorbar = fig.colorbar(hb[3], ax=ax, pad=0.02)
colorbar.set_label("Point density")

correlation, _ = spearmanr(results_df["human_score"], results_df["model_score"])

ax.set_xlabel("Human Similarity Score")
ax.set_ylabel("FastText Cosine Similarity")
ax.set_title("Human Judgments vs FastText Similarity")
ax.set_xlim(results_df["human_score"].min() - 0.1, results_df["human_score"].max() + 0.1)
ax.set_ylim(results_df["model_score"].min() - 0.05, results_df["model_score"].max() + 0.05)
ax.text(
	0.03,
	0.97,
	f"Spearman r = {correlation:.3f}",
	transform=ax.transAxes,
	va="top",
	fontsize=11,
	bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d9dde3", "alpha": 0.95}
)
finalize_plot(ax)
fig.tight_layout()
fig.savefig(results_dir / "scatter_plot.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Plot the model score distribution.
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
	results_df["model_score"], # pyright: ignore[reportArgumentType]
	bins=30,
	stat="density",
	kde=True,
	color=model_hist_color,
	alpha=0.78,
	ax=ax,
	line_kws={"linewidth": 2}
)
ax.axvline(results_df["model_score"].mean(), color=model_hist_color, linestyle="--", linewidth=1.8, alpha=0.9)
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of FastText Similarity Scores")
finalize_plot(ax)
fig.tight_layout()
fig.savefig(results_dir / "similarity_distribution.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Plot the human score distribution.
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
	results_df["human_score"], # pyright: ignore[reportArgumentType]
	bins=30,
	stat="density",
	kde=True,
	color=human_hist_color,
	alpha=0.78,
	ax=ax,
	line_kws={"linewidth": 2}
)
ax.axvline(results_df["human_score"].mean(), color=human_hist_color, linestyle="--", linewidth=1.8, alpha=0.9)
ax.set_xlabel("Human Similarity Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Human Similarity Scores")
finalize_plot(ax)
fig.tight_layout()
fig.savefig(results_dir / "human_distribution.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Plots saved to: {results_dir}")
