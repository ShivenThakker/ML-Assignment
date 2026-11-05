# 🕷️ Spidey-Tracker OS: NYPD & Daily Bugle Threat Verification Engine

```text
       /\             /\
      /  \  __   __  /  \
     /    \(  \_/  )/    \
    /  /\  \       /  /\  \
   /  /  \  \     /  /  \  \
  (  (    \  )   (  /    )  )
   \  \    \(     )/    /  /
    \  \___/  ===  \___/  /
     \_      /   \      _/
       \____/     \____/
```

**Interactive Web Application:** [https://shiventhakker.github.io/ML-Assignment/](https://shiventhakker.github.io/ML-Assignment/)  
**Notebook:** [`SpideyTracker_KNN_KMeans.ipynb`](SpideyTracker_KNN_KMeans.ipynb)  
**Dataset:** [Kaggle — Spidey Tracker (86,631 Sightings)](https://www.kaggle.com/datasets/umuttuygurr/spidey-tracker-spiderman-dataset)  
**License:** CC0 (Public Domain)

---

## 🎯 Mission Overview

For 18 months, a city-wide scanner across New York's 5 boroughs has been logging reports of a masked vigilante. Some sightings are genuine web-slinger interventions, but many are **Mysterio illusions, cosplayers, or Daily Bugle sensationalist hoaxes**.

This project provides an end-to-end Machine Learning pipeline to:
1. **Calibrate Spider-Sense (K-NN):** Filter authentic sightings from hoaxes through a progressive **7-stage Suit Upgrade** cycle, taking accuracy from **72.41% to 91.25%**.
2. **Patrol Grid Analysis (K-Means):** Discover optimal tactical patrol sectors across the city using Elbow and Silhouette analysis.
3. **Solve Mission 05 (Secret Sanctuary):** Triangulate Peter Parker's secret operating base using daily departure and return chronometry.
4. **Deploy Live Tactical HUD (`index.html`):** An interactive radar web page inspired by [`spideytracker.net`](https://spideytracker.net/).

---

## 🚀 Live Interactive Radar Web Page (`index.html`)

Inspired directly by the official Sony Pictures / Marvel tracker interface:
- **Dark Tactical Leaflet Map:** Displays NYC with custom-plotted K-Means patrol sectors.
- **Pulsing Secret Sanctuary Beacon:** Highlights Spider-Man's discovered base in Brooklyn Central.
- **Real-Time Spider-Sense Scanner:** Test custom witness parameters (Speed, Altitude, Witnesses, Crime Proximity) and run the live K-NN model in your browser.
- **Suit Upgrades Terminal:** Visual evolution from the Homemade Hoodie Suit to the Iron Spider Stark-Tech Suit.
- **Animated Radar Sweep Widget:** Live radar sweep HUD in the corner.

> [!TIP]
> To view the live tracker web page, enable **GitHub Pages** on this repository (Settings → Pages → Branch: `main` → `/root`), or simply open `index.html` locally in any browser.

---

## ⚡ Suit Upgrade Chronology (Model Optimization Progression)

Each optimization step represents a tangible suit upgrade with measurable accuracy improvements:

| Stage | Suit Upgrade Milestone | Technique Applied | Test Accuracy | Net Gain | Tactical Rationale |
|---|---|---|---|---|---|
| **1** | 🧵 **Homemade Hoodie Suit** | Default K-NN ($K=5$, Minkowski, No Scaling) | `72.41%` | Baseline | Raw starting baseline; features on large scales dominate unfairly. |
| **2** | 🕸️ **Classic Spidey Suit** | `StandardScaler` Normalization | `85.23%` | **+12.82%** | Eliminates dimensional scale distortion across multi-thousand-meter readings. |
| **3** | 🎯 **Advanced Web-Shooters** | Optimal Neighborhood Radius ($K=7$) | `88.45%` | **+3.22%** | Swept $K=1..30$; eliminates single-neighbor overfitting and broad-neighborhood blurring. |
| **4** | 🏙️ **Spider-Armor MK I** | Manhattan Grid Metric (L1 Norm) | `89.30%` | **+0.85%** | Aligns metric geometry with NYC's rectangular street grid. |
| **5** | ⚖️ **Stealth Big Time Suit** | Distance-Decay Proximity Weighting | `90.15%` | **+0.85%** | Gives closer, high-certainty witness reports greater voting authority. |
| **6** | 🔬 **Superior Spider Suit** | Karen AI Feature Selection (`SelectKBest`) | `91.10%` | **+0.95%** | Pruned noisy sensors (crowd density, raw timestamps) via ANOVA F-tests. |
| **7** | 🛡️ **Iron Spider Stark Suit** | 5-Fold Stratified Cross-Validation | `91.25%` | **+0.15%** | Validated stability across all folds ($\mu=0.9110 \pm 0.0042$). |

**Total Progression:** `72.41%` → `91.25%` (**+18.84% net accuracy boost**).

---

## 🕵️ Mission 05: Unmasking Spider-Man's Secret Operating Base

The Kaggle dataset challenged researchers with an open-ended detective problem: *Infer Spider-Man's operating base from behavioral clues.*

### 🔍 Forensic Evidence:
1. **Morning Patrol Departures:** Analyzing the earliest verified sighting of every single day revealed **`Brooklyn Central 0`** as the top departure origin (**45 separate mornings**).
2. **Nighttime Returns:** The final recorded sighting of the night consistently terminated in **`Brooklyn Central 0`** (**35 nights**).
3. **Resting Hours (1 AM – 5 AM):** The greatest concentration of stationary/rooftop sightings during sleep hours is centered in this district (445 sightings).
4. **District Demographics (`locations.csv`):** `Brooklyn_Central_0` features a **residential concealment score of 0.838** and a **low police patrol score of 0.412**, creating the ideal safe haven.

### 📍 Triangulated Sanctuary Coordinates:
- **Borough:** `Brooklyn`
- **District:** `Brooklyn Central 0`
- **Latitude:** `40.58975° N`
- **Longitude:** `-74.01696° W`
*(Southern Brooklyn / Bay Ridge & Bensonhurst area)*

---

## 🎨 Marvel Spider-Man Color Palette

All plots use a cohesive comic-book palette:
- **Spider-Man Red (`#D82C2C`):** Verified sightings and top-ranked features.
- **Web-Shooter Blue (`#094074`):** Accuracy curves and baseline comparisons.
- **Symbiote Black (`#1B1B1B`):** Cluster centroids, borders, and dark grid lines.
- **Spider-Sense Gold (`#F4D03F`):** Optimal tuning points and the secret base beacon.
- **Mysterio Purple (`#8E44AD`):** Daily Bugle hoaxes and unverified reports.

---

## 📊 Visualizations Included

1. Target Distribution (Verified Hero vs Daily Bugle Hoax — Bar & Donut)
2. Feature Correlation Heatmap (Spider-Sense Sensor Interlink)
3. Sensor Telemetry Distributions (Histograms)
4. Baseline Homemade Suit Confusion Matrix (with Quadrant Subtitles)
5. Neighborhood Radius Calibration (K vs Accuracy Curve)
6. Distance Metric Comparison (Euclidean vs Manhattan vs Chebyshev)
7. Sensor Pruning Curve (ANOVA F-Test Accuracy vs Feature Count)
8. Feature Importance Rankings (Red = Retained, Gray = Pruned)
9. Baseline vs Stark-Tech Confusion Matrices Side-by-Side
10. Patrol Sector Elbow Curve
11. Sector Separation Silhouette Curve
12. K-Means Tactical Patrol Sectors Map (with Centroids)
13. Mission 05 Secret Sanctuary Triangulation Map
14. Suit Upgrade Chronology Bar Chart
15. Spider-Sense Learning Trajectory Line Chart

---

## 🛠️ How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload [`SpideyTracker_KNN_KMeans.ipynb`](SpideyTracker_KNN_KMeans.ipynb) (or clone this repo).
3. Click **Runtime → Run all**.
4. The notebook downloads the Kaggle dataset automatically via `kagglehub` and generates all visual intelligence reports inline.
