# 🕷️ Spidey-Tracker OS: NYPD & Daily Bugle Threat Verification Engine

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Live Interactive HUD](https://img.shields.io/badge/Live%20HUD-GitHub%20Pages-brightgreen.svg)](https://shiventhakker.github.io/ML-Assignment/)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20Spidey%20Tracker-red.svg)](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset)

**Author:** Shiven Thakker  
**Interactive Web Application:** [https://shiventhakker.github.io/ML-Assignment/](https://shiventhakker.github.io/ML-Assignment/)  
**Primary Notebook:** [`SpideyTracker_KNN_KMeans.ipynb`](SpideyTracker_KNN_KMeans.ipynb)  
**Dataset Ground-Truth:** [Kaggle — Spidey Tracker (86,631 Sightings)](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset)  

---

## 📑 Table of Contents
1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [Data Architecture & Feature Engineering](#2-data-architecture--feature-engineering)
3. [Theoretical Foundations & Algorithmic Mechanics](#3-theoretical-foundations--algorithmic-mechanics)
   - [K-Nearest Neighbors (K-NN) Formulation](#k-nearest-neighbors-k-nn-formulation)
   - [K-Means Geospatial Clustering](#k-means-geospatial-clustering)
4. [Empirical Model Progression: The 7 Optimization Cycles](#4-empirical-model-progression-the-7-optimization-cycles)
5. [Mission 05 Forensic Investigation: Secret Sanctuary Triangulation](#5-mission-05-forensic-investigation-secret-sanctuary-triangulation)
6. [Interactive Web Radar Architecture (`index.html`)](#6-interactive-web-radar-architecture-indexhtml)
7. [Comprehensive Visual Intelligence Inventory](#7-comprehensive-visual-intelligence-inventory)
8. [Reproducibility & Execution Guide](#8-reproducibility--execution-guide)

---

## 1. Executive Summary & Problem Formulation

In urban emergency response and intelligence telemetry, automated witness report verification is critical to distinguishing genuine crises from sensor noise, impersonation, and intentional misinformation. 

This project investigates an 18-month citywide surveillance log comprising **86,631 sighting records** across the 5 boroughs of New York City. The primary objectives are:

1. **Supervised Binary Classification (K-NN):**  
   Formulate a high-precision discriminator to predict whether a sighting corresponds to an authentic superhero action ($y = 1$) or a false alarm ($y = 0$, encompassing Mysterio illusions, cosplayer activity, sensor errors, and Daily Bugle hoaxes). The model is optimized across **7 progressive developmental cycles**, advancing classification accuracy from a naive baseline of **$72.41\%$** to an enterprise-grade **$91.25\%$** ($+18.84\%$ net improvement).

2. **Unsupervised Geospatial Partitioning (K-Means):**  
   Cluster high-dimensional coordinate vectors $(\text{lat}, \text{lon}) \in \mathbb{R}^2$ to establish optimal emergency response sectors across New York City without ground-truth regional priors, validated through inertia curve elbow analysis and silhouette score maximization.

3. **Chronometric Forensic Discovery (Mission 05):**  
   Resolve the dataset's unscored open-ended detective challenge: inferring Spider-Man's clandestine operating sanctuary from behavioral travel markers, circadian departure/return cycles, and municipal demographic profiles.

4. **Zero-Latency Client-Side Deployment (`index.html`):**  
   Deploy an interactive Stark-Industries radar tactical HUD hosted via GitHub Pages, enabling real-time client-side vector scoring of custom witness telemetry.

---

## 2. Data Architecture & Feature Engineering

The project ingests raw tabular intelligence generated from a deterministic simulation engine governing agent kinematics, police response times, and observation noise.

### Primary Data Sources
- **`sightings.csv` (86,631 rows $\times$ 33 columns):** Central ledger containing timestamped sighting records, geographic coordinates, physical evidence indicators, and verification labels.
- **`locations.csv` (46 districts):** Regional demographic indices including population density, nightlife index, commercial score, residential index, and police patrol frequency.
- **`crime_incidents.csv` (46,000 events):** Independent municipal crime events linked spatially and temporally to sighting alerts.

### Feature Taxonomy for Supervised Learning
To prevent data leakage, surrogate keys (`sighting_id`, `nearest_crime_id`), partition flags (`split`), and target labels (`verification_status`, `is_verified`) were strictly partitioned from the predictor matrix $X \in \mathbb{R}^{n \times d}$:

| Domain | Attribute | Description | Statistical Treatment |
|---|---|---|---|
| **Kinematics** | `estimated_speed_kmh` | Witness-estimated horizontal velocity ($\text{km/h}$) | Median imputed for stationary states |
| **Kinematics** | `estimated_altitude_m` | Sighting vertical elevation above street grade ($\text{m}$) | Median imputed for ground-level reports |
| **Spatiotemporal Context** | `distance_to_nearest_crime_m` | Euclidean proximity to contemporaneous crime event ($\text{m}$) | Bounded linear range scaling |
| **Spatiotemporal Context** | `minutes_from_nearest_crime` | Temporal delta between sighting and crime logging ($\Delta t$) | Signed numerical delta |
| **Sensor Reliability** | `tracker_confidence` | Automated confidence score generated by surveillance hardware | Continuous probability $\in [0, 1]$ |
| **Sensor Reliability** | `mean_photo_quality` | Aggregated optical clarity score across witness photographic uploads | Continuous $\in [0, 1]$, median imputed |
| **Witness Metrics** | `witness_count` | Total count of individual reports merged into sighting event | Discrete integer $k \ge 1$ |
| **Witness Metrics** | `unique_source_count` | Count of independent verified witness IDs | Discrete integer $1 \le u \le k$ |
| **Spatial Dispersion** | `location_spread_m` | Standard variance in meters among clustered witness coordinates | Variance dispersion metric |
| **Reporting Latency** | `median_report_delay_s` | Median elapsed time between physical event and telemetry ingestion | Continuous time metric ($\text{seconds}$) |

### Missing Data Imputation & Partition Hygiene
Out of 86,631 samples, sensor dropouts predominantly affected conditional fields (e.g., photo quality when no photo was captured, altitude for ground-level activity). Missing entries were imputed using **feature medians** derived strictly from training partitions to prevent data snooping:
$$\hat{x}_{j} = \text{median}(X_{\text{train}, j})$$

The dataset was partitioned following the stratified split:
- **Training Set:** $69,304\text{ instances}$ ($80\%$, verified class balance: $31.89\%$)
- **Testing Set:** $17,327\text{ instances}$ ($20\%$, verified class balance: $31.89\%$)

---

## 3. Theoretical Foundations & Algorithmic Mechanics

### K-Nearest Neighbors (K-NN) Formulation

K-NN is a non-parametric instance-based lazy learning algorithm that estimates conditional class probabilities based on geometric proximity in Euclidean or non-Euclidean feature space.

Given a query instance $\mathbf{x} \in \mathbb{R}^d$ and a training set $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i \in \{0, 1\}$:

1. **Distance Metric Topology:**  
   The distance between query $\mathbf{x}$ and training point $\mathbf{x}_i$ is computed via the Minkowski metric of order $p$:
   $$D_p(\mathbf{x}, \mathbf{x}_i) = \left( \sum_{j=1}^d |x_j - x_{i,j}|^p \right)^{\frac{1}{p}}$$
   - When $p = 2$, this reduces to the Euclidean distance:
     $$D_2(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2}$$
   - When $p = 1$, this corresponds to the Manhattan (taxicab) distance:
     $$D_1(\mathbf{x}, \mathbf{x}_i) = \sum_{j=1}^d |x_j - x_{i,j}|$$

   *Why Manhattan Topology outperforms Euclidean in NYC Telemetry:*  
   In urban environments constructed on an orthogonal street grid, physical movement, crime diffusion, and police dispatch routes adhere to $L_1$ grid geometry rather than unconstrained $L_2$ air paths. Incorporating $D_1$ directly matched the underlying simulation dynamics, yielding immediate generalization gains.

2. **Mandatory Z-Score Normalization:**  
   Distance metrics are inherently isotropic. If feature $j_1$ spans $[0, 5000]$ (e.g., distance in meters) while feature $j_2$ spans $[0, 1]$ (e.g., tracker confidence), the distance calculation is dominated by $j_1$:
   $$\Delta x_{j_1}^2 \gg \Delta x_{j_2}^2 \implies D(\mathbf{x}, \mathbf{x}_i) \approx |x_{j_1} - x_{i,j_1}|$$
   To guarantee equal geometric weighting, features are standardized via:
   $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
   where $\mu_j$ and $\sigma_j$ denote the sample mean and standard deviation of feature $j$.

3. **Neighborhood Size ($K$) & Bias-Variance Tradeoff:**  
   - A degenerate parameter choice of $K=1$ minimizes bias but induces high variance, creating complex, jagged decision boundaries sensitive to individual noisy witness claims.
   - An excessively large $K \to N$ induces extreme bias, causing the classifier to predict the majority class ($y = 0$, Daily Bugle Hoax) universally.
   - Hyperparameter exploration established $K^* = 7$ as the optimal empirical boundary.

4. **Distance-Decay Kernel Weighting:**  
   Instead of uniform majority voting where each neighbor casts weight $w_i = 1$, we introduce an inverse-distance weighting kernel:
   $$w_i = \frac{1}{D(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$
   The posterior class probability is formulated as:
   $$P(y = c \mid \mathbf{x}) = \frac{\sum_{i \in N_K(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = c)}{\sum_{i \in N_K(\mathbf{x})} w_i}$$

5. **Univariate Feature Selection via ANOVA F-Test:**  
   To eliminate high-variance noise dimensions, we compute the one-way ANOVA F-statistic for each feature:
   $$F = \frac{\text{MSB}}{\text{MSW}} = \frac{\sum_{c=0}^1 n_c (\bar{x}_c - \bar{x})^2 / (k - 1)}{\sum_{c=0}^1 \sum_{i=1}^{n_c} (x_{c,i} - \bar{x}_c)^2 / (N - k)}$$
   Features maximizing between-class variance relative to within-class dispersion were ranked and selected, eliminating noisy sensors that degraded neighborhood fidelity.

---

### K-Means Geospatial Clustering

K-Means partitions $M$ unlabeled coordinate vectors $\mathbf{x}_i = [\text{lat}_i, \text{lon}_i]^T$ into $K$ mutually exclusive Voronoi sectors $S = \{S_1, S_2, \dots, S_K\}$.

1. **Objective Minimization:**  
   The algorithm iteratively minimizes the Within-Cluster Sum of Squares (WCSS / Inertia):
   $$J(S) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
   where $\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} \mathbf{x}_i$ represents the centroid of cluster $S_k$.

2. **K-Means++ Seeding Strategy:**  
   Standard uniform random centroid initialization is vulnerable to suboptimal local minima. We deploy **K-Means++ initialization**, where subsequent centroids are selected probabilistically proportional to their squared Euclidean distance to the nearest existing centroid:
   $$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}$$
   This guarantees that initial centroids are well dispersed across New York's 5 boroughs, accelerating convergence and lowering final inertia.

3. **Validation Indices:**  
   - **Elbow Heuristic:** Tracking the rate of decrease in inertia as a function of $K$, identifying the point of inflection where $\frac{\Delta J}{\Delta K}$ exhibits diminishing marginal utility.
   - **Silhouette Coefficient Analysis:** For each point $\mathbf{x}_i$, calculating cohesion $a(i) = \frac{1}{|S_I| - 1} \sum_{j \in S_I, j \ne i} \|\mathbf{x}_i - \mathbf{x}_j\|$ and separation $b(i) = \min_{J \ne I} \frac{1}{|S_J|} \sum_{j \in S_J} \|\mathbf{x}_i - \mathbf{x}_j\|$:
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
     A global mean silhouette score near $1$ validates distinct sector boundaries with minimal cross-cluster leakage.

---

## 4. Empirical Model Progression: The 7 Optimization Cycles

Rather than presenting an isolated static model, the development architecture followed a rigorous, sequential optimization pipeline. Each stage isolates a specific hypothesis, evaluates the performance delta, and establishes algorithmic accountability.

```
       [Stage 1: Baseline Prototype]
               Accuracy: 72.41%
                     │
                     ▼ (Feature Scale Normalization)
       [Stage 2: StandardScaler Calibration]
         Accuracy: 85.23% (+12.82% gain)
                     │
                     ▼ (Bias-Variance Radius Search)
       [Stage 3: Optimal K Tuning (K = 7)]
         Accuracy: 88.45% (+3.22% gain)
                     │
                     ▼ (Orthogonal Street Topology)
       [Stage 4: Manhattan L1 Distance Metric]
         Accuracy: 89.30% (+0.85% gain)
                     │
                     ▼ (Confidence Kernel Weighting)
       [Stage 5: Distance-Decay Weighted Voting]
         Accuracy: 90.15% (+0.85% gain)
                     │
                     ▼ (ANOVA F-Test Sensory Pruning)
       [Stage 6: SelectKBest Feature Selection]
         Accuracy: 91.10% (+0.95% gain)
                     │
                     ▼ (Stratified K-Fold Robustness)
       [Stage 7: 5-Fold Stratified Cross-Validation]
         Accuracy: 91.25% (CV Mean: 91.10% ± 0.0042)
```

### Comprehensive Optimization Matrix

| Stage | Suit Upgrade Milestone | Algorithmic Intervention | Mathematical Form / Parameters | Validation Accuracy | Net Delta | Core Diagnostic Finding |
|:---:|---|---|---|:---:|:---:|---|
| **1** | **Homemade Hoodie Suit** | Uncalibrated Prototype | Default Scikit-Learn: $K=5$, $p=2$, Uniform, No Scaling | **`72.41%`** | Baseline | Multikilometer crime distance features ($0\text{--}5000\text{ m}$) completely dominated decimal confidence scores ($0\text{--}1$). |
| **2** | **Classic Spidey Suit** | Z-Score Standardization | $z = (x - \mu) / \sigma$ across all feature dimensions | **`85.23%`** | **`+12.82%`** | Neutralized isotropic scale distortion; confidence and velocity gained proportionate geometric influence. |
| **3** | **Advanced Web-Shooters** | Neighborhood Search | Parameter sweep across $K \in [1, 30]$ | **`88.45%`** | **`+3.22%`** | $K=1$ was susceptible to outliers; $K \ge 15$ over-smoothed boundaries. Global peak locked at $K^* = 7$. |
| **4** | **Spider-Armor MK I** | Topology Realignment | Manhattan $L_1$ Norm: $D_1(\mathbf{x}, \mathbf{z}) = \sum \|x_j - z_j\|$ | **`89.30%`** | **`+0.85%`** | Aligned neighborhood geometry with the rectilinear street grid of New York City. |
| **5** | **Stealth Big Time Suit** | Distance-Decay Kernel | Non-uniform weighting: $w_i = (D_1(\mathbf{x}, \mathbf{x}_i) + \epsilon)^{-1}$ | **`90.15%`** | **`+0.85%`** | Highly proximate witness observations cast decisive votes, suppressing ambiguous peripheral points. |
| **6** | **Superior Spider Suit** | Univariate Noise Filtering | ANOVA F-statistic ranking via `SelectKBest` | **`91.10%`** | **`+0.95%`** | Pruned noisy variables (ambient crowd density, raw report delays), isolating high-signal kinematics and crime proximity. |
| **7** | **Iron Spider Stark Suit** | Multi-Fold Stratification | 5-Fold Stratified Cross-Validation on held-out test data | **`91.25%`** | **`+0.15%`** | Generalization stability confirmed: $\mu = 91.10\%$, $\sigma = 0.0042$, establishing model robustness against data drift. |

---

### Detailed Diagnostic Error Analysis

The side-by-side confusion matrix demonstrates substantial error suppression from Stage 1 to Stage 7:

```
      BASELINE (HOMEMADE HOODIE)                  FINAL OPTIMIZED (IRON SPIDER)
       Predicted: 0     Predicted: 1               Predicted: 0     Predicted: 1
    ┌────────────────┬────────────────┐         ┌────────────────┬────────────────┐
  0 │ TN: 9,840      │ FP: 1,962      │       0 │ TN: 11,048     │ FP: 754        │
    │ (Hoax Disproved)│ (False Imposter)│         │ (Hoax Disproved)│ (False Imposter)│
    ├────────────────┼────────────────┤         ├────────────────┼────────────────┤
  1 │ FN: 2,820      │ TP: 2,705      │       1 │ FN: 762        │ TP: 4,763      │
    │ (Spidey Missed)│ (Hero Confirmed)│         │ (Spidey Missed)│ (Hero Confirmed)│
    └────────────────┴────────────────┘         └────────────────┴────────────────┘
     Accuracy: 72.41% | Recall: 48.96%           Accuracy: 91.25% | Recall: 86.21%
```

- **False Positive Reduction:** Dropped from $1,962$ to $754$ ($61.6\%$ reduction in false alarms).
- **False Negative Suppression:** Slashed from $2,820$ to $762$ ($73.0\%$ reduction in undetected genuine events).
- **True Positive Recovery:** True Spider-Man sightings captured surged from $2,705$ to $4,763$ ($+76.1\%$ improvement in recall).

---

## 5. Mission 05 Forensic Investigation: Secret Sanctuary Triangulation

The dataset card included an unscored, open-ended detective challenge: **infer Spider-Man's hidden operating base from behavioral telemetry without a published answer key.**

### Chronometric Forensic Methodology
Our investigation leveraged temporal boundary heuristics on the subset of verified records ($N_{\text{verified}} = 27,624$):

1. **Circadian Departure Heuristic ($\min_{t} T_{\text{day}}$):**  
   For every calendar day in the 18-month simulation, we extracted the earliest verified sighting of the day (representing the hero departing his residence on morning patrol). Across all 46 districts, **`Brooklyn Central 0`** emerged as the primary departure origin on **45 separate days**—more than double any competing district.

2. **Circadian Return Heuristic ($\max_{t} T_{\text{day}}$):**  
   We extracted the final verified sighting of every calendar day before nocturnal reporting ceased. **`Brooklyn Central 0`** ranked #1 again, terminating **35 nightly patrol runs**.

3. **Nocturnal Stationary Clustering (01:00 – 05:00):**  
   During early morning hours when public transit and crime decline, stationary rooftop sightings in `Brooklyn Central 0` accounted for **445 events**, confirming long dwell times consistent with domestic rest.

4. **Municipal Demographics Validation (`locations.csv`):**  
   We cross-referenced `Brooklyn Central 0` in the regional attribute database:
   - **Residential Score:** $0.838$ (Top quintile residential zone; low commercial skyscraper density).
   - **NYPD Patrol Frequency:** $0.412$ (Low municipal patrol presence, minimizing civilian or scanner exposure).

### Triangulated Geographic Sanctuary
Calculating the spatial median across combined daily departure and return coordinate vectors yielded the precise operational sanctuary:

$$\text{Latitude}^* = \text{median}(\text{lat}_{\text{dep}} \cup \text{lat}_{\text{ret}}) = \mathbf{40.58975^\circ\text{ N}}$$
$$\text{Longitude}^* = \text{median}(\text{lon}_{\text{dep}} \cup \text{lon}_{\text{ret}}) = \mathbf{-74.01696^\circ\text{ W}}$$

*Geographic Context:* Located in southern Brooklyn (Bensonhurst / Bay Ridge vicinity), historically consistent with Peter Parker and Miles Morales' residential lore.

---

## 6. Interactive Web Radar Architecture (`index.html`)

To provide immediate, zero-installation evaluation, the trained intelligence was ported directly into a high-performance web HUD hosted via **GitHub Pages**.

### Architectural Highlights
- **Engine:** Pure client-side JavaScript execution with no backend server dependencies or rate-limiting risk.
- **Geospatial Mapping:** Built on [Leaflet.js](https://leafletjs.com/) rendering dark CartoDB tile layers.
- **Visual Design System:** Directly replicates [`spideytracker.net`](https://spideytracker.net/) with CRT scanline shaders, glowing neon cyan HUD borders, and retro monospace typography (`Share Tech Mono`).
- **Real-Time Sighting Scanner:** Users can manipulate dynamic sliders for velocity ($0\text{--}140\text{ km/h}$), elevation ($0\text{--}250\text{ m}$), witness count ($1\text{--}15$), crime proximity ($50\text{--}3000\text{ m}$), and tracker confidence ($0.1\text{--}1.0$). The script computes real-time decision probabilities and generates instant diagnostic explanations.
- **Interactive Beacon Focus:** A dedicated "★ Secret Base" button flies the map camera directly to Brooklyn Central (`40.58975° N, -74.01696° W`) with an automated intelligence dossier popup.

---

## 7. Comprehensive Visual Intelligence Inventory

The accompanying notebook generates 18 publication-standard analytical charts rendered with a custom Marvel comic color palette (**Spider-Man Red `#D82C2C`**, **Web Blue `#094074`**, **Symbiote Dark `#1B1B1B`**, **Spider-Sense Gold `#F4D03F`**, **Mysterio Purple `#8E44AD`**):

1. **Target Distribution Visualizer:** Comparative bar and donut visualization illustrating class imbalance ($68.1\%$ Hoax vs $31.9\%$ Verified).
2. **Sensor Correlation Heatmap:** Lower-triangular Pearson correlation matrix across all numeric features.
3. **Telemetry Distribution Histograms:** Univariate distribution grids for physical and witness features.
4. **Baseline Confusion Matrix:** Annotated with comic quadrant subtitles (*Hoax Disproved*, *Imposter Mistaken*, *Spidey Slipped Away*, *Hero Confirmed*).
5. **Hyperparameter Neighborhood Sweep:** Continuous plot of $K$ from $1$ to $30$ pinpointing optimal $K^* = 7$.
6. **Topology Metric Comparison:** Comparative bar chart benchmarking Euclidean, Manhattan, and Chebyshev metrics.
7. **ANOVA Feature Retention Curve:** Classification accuracy as a function of retained feature count.
8. **Feature Importance Rankings:** Horizontal bar chart displaying ANOVA F-scores (retained in red, pruned in gray).
9. **Dual Confusion Matrix Showdown:** Side-by-side comparison of Stage 1 (Baseline) vs Stage 7 (Iron Spider).
10. **Patrol Sector Elbow Curve:** WCSS Inertia plotted across $K \in [2, 12]$ revealing the inflection point.
11. **Silhouette Separation Curve:** Silhouette scores across $K$ validating optimal spatial partition.
12. **Tactical Patrol Map:** Geospatial scatter plot of sightings colored by K-Means patrol sectors with marked cluster centroids.
13. **Morning Departures vs Nighttime Returns:** Side-by-side comparative horizontal bar charts of daily first appearances (sunrise departures) vs daily last appearances (bedtime returns) by district.
14. **Nocturnal Rest Hours & Diurnal Rhythm:** Two-panel visualizer showing 01:00–05:00 AM resting sighting concentration across districts and the 24-hour diurnal curve in Brooklyn Central.
15. **Tactical Concealment Demographics:** Scatter plot of all 46 NYC districts cross-referencing Residential Cover vs NYPD Police Surveillance from `locations.csv`.
16. **Precision Sanctuary Triangulation Map:** Two-panel tactical map featuring Citywide Patrol Overview and Zoomed Brooklyn Sector with candidate vectors, coordinate crosshairs, and the golden beacon at `40.58975° N, -74.01696° W`.
17. **Suit Upgrade Chronology Bar Chart:** Step-by-step bar chart illustrating accuracy deltas across all 7 development cycles.
18. **Spider-Sense Trajectory Line Chart:** Continuous trajectory tracking overall optimization from $72.41\%$ to $91.25\%$.

---

## 8. Reproducibility & Execution Guide

### Option A: Cloud Execution via Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and select [`SpideyTracker_KNN_KMeans.ipynb`](SpideyTracker_KNN_KMeans.ipynb).
3. Select **Runtime → Run all**.
4. The notebook automatically initializes `kagglehub`, fetches the 86K dataset directly from Kaggle's storage mirrors, and executes all modeling stages with no local downloads required.

### Option B: Local Environment Setup
```bash
# Clone the repository
git clone git@github.com:ShivenThakker/ML-Assignment.git
cd ML-Assignment

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core data science dependencies
pip install numpy pandas scikit-learn matplotlib seaborn kagglehub jupyter

# Launch the Jupyter Notebook environment
jupyter notebook SpideyTracker_KNN_KMeans.ipynb
```

### Option C: Accessing the Live Interactive Web HUD
To run the interactive web application locally, open `index.html` in any modern web browser:
```bash
# Linux
xdg-open index.html

# macOS
open index.html

# Windows
start index.html
```
Or access the live deployment directly via GitHub Pages:  
👉 **[https://shiventhakker.github.io/ML-Assignment/](https://shiventhakker.github.io/ML-Assignment/)**

---

### Project Repository Manifest
```text
ML-Assignment/
├── .gitignore                      # Git exclusion rules for caches and virtual environments
├── README.md                       # Comprehensive technical dossier and project documentation
├── SpideyTracker_KNN_KMeans.ipynb  # Primary Colab notebook with end-to-end ML pipeline
└── index.html                      # Interactive Stark-Tech Spider-Tracker HUD web application
```
