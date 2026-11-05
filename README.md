# 🕷️ Spidey Tracker — K-NN Classification & K-Means Clustering

A machine learning project that uses the [Spidey Tracker: Spiderman Dataset](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset) to classify Spider-Man sightings and discover location clusters.

## Dataset

- **Source:** [Kaggle - Spidey Tracker](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset)
- **Size:** 86,631 sighting reports + supporting tables (crimes, locations, weather, witnesses)
- **License:** CC0 (Public Domain)

The dataset contains synthetic Spider-Man sighting reports across a New York-inspired city. Each sighting includes location coordinates, timestamps, confidence scores, and a verified/unverified label. Some sightings are real, some are mistakes, and some are deliberate hoaxes.

### Files Used

| File | Rows | Purpose |
|---|---|---|
| `sightings.csv` | 86,631 | Main dataset — used for K-NN classification and K-Means clustering |

## Project Structure

```
mlassignment1/
├── SpideyTracker_KNN_KMeans.ipynb   # Main Colab notebook (run this)
├── README.md                         # This file
└── .gitignore
```

## How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `SpideyTracker_KNN_KMeans.ipynb` (or open from GitHub)
3. Click **Runtime → Run all**
4. The dataset downloads automatically via `kagglehub` — no manual download needed
5. All outputs (tables, charts, metrics) appear inline

**No local setup required.** Everything runs in Colab with pre-installed libraries.

## What This Project Does

### Part 1: K-NN Classification

**Task:** Predict whether a Spider-Man sighting is verified or not (`is_verified` = 0 or 1).

The model is improved step-by-step, with accuracy measured after **every single change**:

| Step | What Changed | Why |
|---|---|---|
| 1. Baseline | Default KNN (K=5, no scaling, minkowski) | Starting point to measure against |
| 2. Feature Scaling | Applied StandardScaler | KNN uses distances — unscaled features bias results |
| 3. K Tuning | Tested K=1 to 30, picked best | Default K=5 may not be optimal for this data |
| 4. Distance Metric | Compared euclidean, manhattan, chebyshev | Different metrics suit different data shapes |
| 5. Weighted Voting | Compared uniform vs distance weights | Closer neighbors should matter more |
| 6. Feature Selection | ANOVA F-test, kept top features | Removing noise features can improve accuracy |
| 7. Cross-Validation | 5-fold CV on final model | Single split can be misleading — CV is more reliable |

Each step includes:
- The accuracy number
- The change from the previous step (+ or -)
- A visualization (confusion matrix, bar chart, or line plot)

At the end, a **bar chart and line chart** show the full improvement progression across all 7 steps.

### Part 2: K-Means Clustering & Mission 05 Secret Base Discovery

**Task 1:** Discover natural geographic patrol clusters in Spider-Man sightings (unsupervised).
**Task 2 (Mission 05):** Solve the unscored Kaggle detective mission to pinpoint Spider-Man's secret operating base.

| Step | What it Does |
|---|---|
| Elbow Method | Plots inertia vs K to find the "elbow" point |
| Silhouette Score | Measures how well-separated clusters are |
| Cluster Visualization | Scatter plot of sightings colored by cluster, with cluster centers marked |
| Init Comparison | Compares random init vs K-Means++ initialization |
| Secret Base Discovery | Identifies morning departure and nighttime return hotspots to unmask the base |

#### 🕵️ Secret Base Discovery Findings:
- **Primary District:** `Brooklyn Central 0`
- **Borough:** `Brooklyn`
- **Estimated Coordinates:** `40.58975° N, -74.01696° W`
- **Why this is the base:**
  1. **Daily Departures:** Highest concentration of first-recorded daily sightings (45 mornings).
  2. **Daily Returns:** Highest concentration of final nighttime sightings (35 nights).
  3. **Demographics:** According to `locations.csv`, this zone has a high residential score (0.838) and low police patrol activity (0.412), making it the ideal quiet home base.

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Charts and plots |
| `seaborn` | Statistical visualizations, heatmaps |
| `scikit-learn` | KNeighborsClassifier, KMeans, StandardScaler, SelectKBest, cross_val_score, metrics |
| `kagglehub` | Download dataset from Kaggle |

## Visualizations Included

1. Target variable distribution (bar + pie chart)
2. Feature correlation heatmap
3. Feature distributions (histograms)
4. Baseline confusion matrix
5. K vs Accuracy curve
6. Distance metric comparison bar chart
7. Number of features vs accuracy curve
8. Feature importance horizontal bar chart
9. Baseline vs Optimized confusion matrix side-by-side
10. Elbow method curve
11. Silhouette score vs K
12. Cluster scatter plot with centers
13. Spider-Man secret operating base map (Mission 05)
14. Progressive improvement bar chart
15. Accuracy trajectory line chart

## Notebook Sections

| Section | Description |
|---|---|
| 1. Setup & Imports | Install kagglehub, import libraries |
| 2. Download & Load | Download from Kaggle, load sightings.csv |
| 3. EDA | Explore data shape, types, missing values, distributions |
| 4. Preprocessing | Select features, handle nulls, train/test split |
| 5. K-NN Classification | 7-step progressive optimization |
| 6. K-Means Clustering | Location clustering with elbow + silhouette |
| 7. Improvement Summary | Full progression table + charts |
| 8. Conclusion | Key takeaways |
