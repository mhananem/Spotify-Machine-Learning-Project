# 🎵 Hype Check - Predicting Spotify Song Popularity Using Machine Learning

> *Can we predict whether a song will become popular on Spotify by analyzing how it sounds and where it lives across platforms?*

**Authors:** Hanane Mamalik & Kanak Yadav  
**Program:** Ironhack Data Analytics Bootcamp — Berlin, 2026  
**Presentation:** [View Google Slides](https://docs.google.com/presentation/d/1uBOyjTdkTMEvdgi6xWfE6fVsWyRD9f4xoPUxmHrBzLc/edit?usp=sharing)

---

## 📌 Project Overview

Hype Check is a supervised machine learning classification project that predicts whether a song will be **popular on Spotify** (Popularity ≥ 70, top 30%) using a combination of **audio DNA** (danceability, speechiness, tempo, etc.) and **platform presence signals** (Shazam counts, AirPlay spins, YouTube views, TikTok views).

Four ML algorithms were built, tuned, and benchmarked:

| Model | Default Accuracy | Tuned Accuracy |
|---|---|---|
| 🏆 **XGBoost** | **0.7674** | 0.7637 |
| Ensemble (Voting) | — | 0.7637 |
| Decision Tree | 0.7473 | 0.7289 |
| KNN | 0.6557 | 0.5788 |
| Random Forest | 0.6374 | 0.6190 |

**Best model: XGBoost (default) at 76.74% accuracy.**

---

## 📂 Repository Structure

```
Spotify-Machine-Learning-Project/
│
├── data/
│   ├── processed/
│   │   ├── final_spotify.csv               # Final merged & feature-engineered dataset
│   │   └── spotify_kaggle_dataset_clean.csv  # Cleaned Kaggle dataset
│   └── raw/
│       ├── audio_features.csv              # Audio features from RapidAPI (Musixae)
│       ├── isrc_to_spotify_id.csv          # ISRC → Spotify ID mapping
│       ├── spotify_dataset.csv             # Original Kaggle dataset
│       └── spotify_with_genres.csv         # Dataset with genre labels
│
├── notebooks/
│   ├── 1_kaggle_data_cleaning.ipynb        # Data cleaning & preprocessing
│   ├── 2_data_load_api_preprocessing.ipynb # API data collection & merging
│   └── 3_modeling.ipynb                    # Feature engineering, model building, tuning & evaluation
│
└── README.md
```

---

## 🗃️ Data Sources

### Dataset 1 — Most Streamed Spotify Songs 2024 (Kaggle)
~4,600 tracks with:
- Spotify streams, playlist & chart counts
- YouTube views & likes, TikTok views & likes
- Shazam counts, AirPlay spins
- Genre, ISRC code

### Dataset 2 — Audio Features via RapidAPI (Musixae)
Fetched using ISRC → Spotify ID → audio features pipeline:
- Danceability, Energy, Loudness, Valence, Tempo
- Speechiness, Acousticness, Instrumentalness, Liveness
- Key, Mode, Time Signature, Duration (ms)

> **Note:** Spotify's native audio features endpoint was deprecated in November 2024. Musixae API was used as the alternative.

### Final Dataset
- **2,727 tracks × 28 features** after merging and cleaning
- Target: `is_popular` (binary) — 1,957 not popular / 770 popular
- Train/test split: 80/20 stratified

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (KNN, DT, RF, Voting Ensemble), XGBoost |
| Hyperparameter tuning | GridSearchCV with 5-Fold Cross-Validation |
| Data collection | Spotify API, Musixae/RapidAPI |
| Notebook environment | Jupyter Notebook |

---

## ⚙️ Feature Engineering

**18 features across two layers:**

**Audio DNA (how it sounds)**  
`Danceability`, `Energy`, `Valence`, `Loudness`, `Tempo`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Key`, `Mode`, `Time_signature`, `Duration_ms`

**Platform Presence (where it lives)**  
`YouTube_Views`, `TikTok_Views`, `Shazam_Counts`, `AirPlay_Spins`, `Genre` (label-encoded), `Artist_Track_Count` (engineered)

**Target Variable**  
`is_popular` = 1 if `spotify_popularity ≥ 70`, else 0

---

## 🔍 Key Findings

### What drives song popularity?

Feature importance averaged across Decision Tree, Random Forest, and XGBoost:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Shazam Counts | 0.4546 |
| 2 | AirPlay Spins | 0.1476 |
| 3 | Speechiness | 0.1441 |
| 4 | Danceability | 0.0841 |
| 5 | Duration_ms | 0.0451 |

**Where a song lives matters more than how it sounds.** Platform discoverability (Shazam, AirPlay) is by far the dominant predictor — but of audio features, Speechiness and Danceability still hold meaningful predictive power.

### Why did tuning hurt?

GridSearchCV with 5-fold CV was applied to all four models. Tuning decreased accuracy across the board. The default XGBoost (76.7%) outperformed every tuned version — suggesting the default hyperparameters were already near-optimal for this dataset, and tuning caused slight overfitting to the CV folds.

### Model limitations

- **Low recall on the Popular class (~28–34%):** models are conservative — precise when they call a song popular, but miss many actual hits
- **Survivorship bias:** the dataset skews toward charting/mainstream songs, underrepresenting indie and niche music
- Popularity ≠ quality — the model predicts commercial success metrics, not artistic merit

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Kanak2208/Spotify-Machine-Learning-Project.git
cd Spotify-Machine-Learning-Project
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter requests
```

### 3. Set up data

Download the [Most Streamed Spotify Songs 2024](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024/data) dataset from Kaggle and place the CSV in `data/`.

Audio features were collected via RapidAPI (Musixae). You'll need a RapidAPI key to reproduce the collection step — or use the pre-processed file already in `data/`.

### 4. Run the notebooks

```bash
jupyter notebook
```

Open the notebooks in order:

1. `1_kaggle_data_cleaning.ipynb` — clean and explore the Kaggle dataset
2. `2_data_load_api_preprocessing.ipynb` — fetch audio features via API and merge datasets
3. `3_modeling.ipynb` — feature engineering, model training, tuning, and evaluation

If you're using the pre-processed data, you can jump straight to `3_modeling.ipynb`.

---

## 📊 Evaluation Metrics Used

- **Accuracy** — overall % of correct predictions
- **Precision** — when the model says "popular", how often is it right?
- **Recall** — of all actual popular songs, how many did it catch?
- **F1 Score** — harmonic mean of precision and recall
- **5-Fold Cross-Validation** — used during GridSearchCV for reliable generalisation estimates
- **Confusion Matrix** — per-model breakdown of true/false positives and negatives

---

## 🔮 Future Work

1. **Address low recall** — explore SMOTE, class weighting, or threshold tuning to better identify popular songs
2. **Larger & more balanced dataset** — include more independent/underground artists to reduce survivorship bias
3. **NLP on lyrics** — add sentiment analysis and lyrical complexity features
4. **Time-series analysis** — predict not just *if* a song will pop, but *when*

---

## 🌐 Connect

**Kanak Yadav**
- GitHub: [Kanak2208](https://github.com/Kanak2208)
- LinkedIn: [kanakyadav22](https://www.linkedin.com/in/kanakyadav22)

**Hanane Mamalik**
- GitHub: [mhananem](https://github.com/mhananem)
- LinkedIn: [hanane-mamalik](https://www.linkedin.com/in/hanane-mamalik/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
