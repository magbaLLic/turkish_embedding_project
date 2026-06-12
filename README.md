# Turkish Word Embeddings: Semantic Similarity and Relationship Analysis

## Overview

This project investigates how effectively Turkish word embeddings capture semantic similarity and semantic relationships between words.

The study evaluates whether distributional semantic representations can distinguish between semantically related and unrelated words and whether they encode linguistic relationships such as gender, synonymy, antonymy, and hierarchical relations.

## Research Questions

1. Do Turkish word embeddings assign higher similarity scores to semantically related words than to unrelated words?

2. Can word embeddings capture semantic relationships such as:
   - Synonymy
   - Antonymy
   - Hypernym–Hyponym relations
   - Gender relations

3. How well do vector representations reflect semantic structure in Turkish?

## Dataset

The evaluation dataset consists of manually curated Turkish word pairs grouped into categories:

- Similar words
- Related words
- Unrelated words
- Synonyms
- Antonyms
- Hypernym–Hyponym pairs
- Gender pairs

Example:

| Word 1 | Word 2 | Relation |
|---------|---------|----------|
| kadın | erkek | Gender |
| köpek | hayvan | Hypernym |
| mutlu | neşeli | Synonym |
| sıcak | soğuk | Antonym |
| elma | demokrasi | Unrelated |

## Methodology

### Semantic Similarity

Cosine similarity is computed between word vectors.

Higher similarity scores are expected for semantically related words.

### Semantic Relationship Analysis

The project evaluates whether embeddings capture:

- Synonym relations
- Antonym relations
- Hierarchical relations
- Gender relations

### Visualization

Word embeddings are projected into two dimensions using:

- PCA
- t-SNE

Visualizations help identify semantic clusters and relationships.

## Technologies

- Python
- FastText
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
turkish_embedding_project/
│
├── data/
│   ├── similarity_pairs.csv
│   ├── relation_pairs.csv
│   └── embeddings/
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── similarity_analysis.py
│   ├── relation_analysis.py
│   └── visualization.py
│
├── results/
│   ├── similarity_scores.csv
│   └── figures/
│
├── README.md
├── LICENSE
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/turkish_embedding_project.git
cd turkish_embedding_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run similarity analysis:

```bash
python src/similarity_analysis.py
```

Run semantic relation analysis:

```bash
python src/relation_analysis.py
```

Generate visualizations:

```bash
python src/visualization.py
```

## Expected Findings

- Related words should have higher cosine similarity than unrelated words.
- Word embeddings should capture meaningful semantic structure.
- Semantic relations may emerge through vector space organization.
- Visualization should reveal clustering of semantically related concepts.

## Course Information

**Course:** Computational Semantics and Discourse Processing

**Project Topic:** Semantic Similarity and Meaning Relationships in Turkish Word Embeddings

## Authors

Irem Yurdakul
