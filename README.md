# 🕷️ Spidey Tracker — K-NN Classification & K-Means Clustering

A machine learning project that uses the [Spidey Tracker: Spiderman Dataset](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset) to classify Spider-Man sightings and discover location clusters.

## Dataset

- **Source:** [Kaggle - Spidey Tracker](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset)
- **Size:** 86,631 sighting reports + supporting tables
- **License:** CC0 (Public Domain)

The dataset contains synthetic Spider-Man sighting reports across a New York-inspired city. Each sighting has location coordinates, timestamps, confidence scores, and a verified/unverified label.

## Project Structure

```
mlassignment1/
├── SpideyTracker_KNN_KMeans.ipynb   # Main Colab notebook
├── README.md                         # This file
└── .gitignore
```

## How to Run

1. Open the notebook in Google Colab
2. Run all cells — the dataset downloads automatically via `kagglehub`
3. No local setup needed

## What This Project Does

### K-NN Classification
Predicts whether a Spider-Man sighting is **verified or not** (`is_verified` column).

### K-Means Clustering
Groups sighting locations into clusters to find patterns in where Spider-Man is spotted.

## Methodology & Optimization Steps

The model is improved step-by-step, with accuracy measured after each change:

| # | Optimization Step | What Changed |
|---|---|---|
| 1 | Baseline | Default KNN (K=5, no scaling) |
| 2 | Feature Scaling | StandardScaler normalization |
| 3 | K Tuning | Test K=1 to 30, pick the best |
| 4 | Distance Metric | Compare euclidean, manhattan, chebyshev |
| 5 | Weighted Voting | uniform vs distance-weighted neighbors |
| 6 | Feature Selection | ANOVA F-test to keep top features |
| 7 | Cross-Validation | 5-fold CV for reliable accuracy |

## Libraries Used

- `numpy` — numerical operations
- `pandas` — data loading and manipulation
- `matplotlib` — plotting
- `seaborn` — statistical visualizations
- `scikit-learn` — KNN, KMeans, preprocessing, evaluation
