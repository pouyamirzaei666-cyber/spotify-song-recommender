"""Clean the supplied CSV and build reusable recommendation artifacts."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.recommender import AUDIO_FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "universal_top_spotify_songs.csv"
MODELS_DIR = ROOT / "models"
SAMPLE_SIZE = 12_000
RANDOM_STATE = 42
DBSCAN_EPS = 1.9
DBSCAN_MIN_SAMPLES = 12


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place the CSV in data/ after checking its license."
        )
    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = ["name", "artists", *AUDIO_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Coerce malformed feature values to NaN before dropping incomplete rows.
    for feature in AUDIO_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    clean = df.dropna(subset=required).copy()
    clean["name"] = clean["name"].astype(str).str.strip()
    clean["artists"] = clean["artists"].astype(str).str.strip()
    clean = clean[(clean["name"] != "") & (clean["artists"] != "")]

    # Prefer the Spotify track ID for deduplication when it is available.
    # This prevents one track from appearing more than once when chart snapshots
    # contain minor spelling or artist-credit variations.
    if "spotify_id" in clean.columns:
        has_id = clean["spotify_id"].notna() & clean["spotify_id"].astype(str).str.strip().ne("")
        with_id = clean.loc[has_id].drop_duplicates(subset=["spotify_id"], keep="first")
        without_id = clean.loc[~has_id].drop_duplicates(subset=["name", "artists"], keep="first")
        clean = pd.concat([with_id, without_id], ignore_index=True)
    else:
        clean = clean.drop_duplicates(subset=["name", "artists"], keep="first")

    clean = clean.reset_index(drop=True)
    print(f"Rows before cleaning: {len(df):,}")
    print(f"Rows after cleaning:  {len(clean):,}")
    if len(clean) < 2:
        raise ValueError("At least two valid unique tracks are required.")

    sample = clean.sample(n=min(SAMPLE_SIZE, len(clean)), random_state=RANDOM_STATE).reset_index(drop=True)
    X = sample[AUDIO_FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN is retained for exploratory diagnostics only; it does not filter recommendations.
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)
    sample["cluster"] = labels
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Sample size: {len(sample):,}")
    print(f"DBSCAN clusters (excluding noise): {n_clusters}")
    print("Cluster sizes:")
    print(cluster_counts.to_string())

    # Recommendations search the full sample, independent of DBSCAN cluster labels.
    neighbors = NearestNeighbors(metric="euclidean", algorithm="auto")
    neighbors.fit(X_scaled)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(dbscan, MODELS_DIR / "dbscan_model.joblib")
    joblib.dump(neighbors, MODELS_DIR / "nearest_neighbors.joblib")
    sample.to_csv(MODELS_DIR / "songs.csv", index=False, encoding="utf-8")
    np.save(MODELS_DIR / "X_scaled.npy", X_scaled)
    metadata = {
        "sample_size": int(len(sample)),
        "features": AUDIO_FEATURES,
        "random_state": RANDOM_STATE,
        "recommendation_metric": "euclidean_distance_on_standardized_features",
        "dbscan_cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Artifacts saved to: {MODELS_DIR}")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
