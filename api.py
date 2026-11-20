"""Optional REST API for the music recommendation engine."""
from __future__ import annotations
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from src.recommender import load_artifacts, recommend_by_index, search_songs

app = FastAPI(
    title="Music Discovery API",
    description="Content-based track recommendations using standardized audio features and nearest neighbors.",
    version="1.0.0",
)

@lru_cache(maxsize=1)
def get_artifacts():
    try:
        return load_artifacts()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/")
def root() -> dict:
    return {"message": "Music Discovery API is running", "docs": "/docs"}

@app.get("/search")
def search(query: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)) -> dict:
    artifacts = get_artifacts()
    matches = search_songs(artifacts["songs"], query, limit=limit)
    return {"query": query, "results": [
        {"track_index": int(idx), "name": row["name"], "artists": row["artists"]}
        for idx, row in matches.iterrows()
    ]}

@app.get("/recommend")
def recommend(
    track_index: int = Query(..., ge=0),
    top_n: int = Query(10, ge=1, le=30),
    exclude_same_artist: bool = Query(False),
) -> dict:
    artifacts = get_artifacts()
    songs = artifacts["songs"]
    if track_index >= len(songs):
        raise HTTPException(status_code=404, detail="track_index is outside the available dataset")
    target = songs.iloc[track_index]
    results = recommend_by_index(
        track_index, songs, artifacts["scaled_features"], artifacts["neighbor_model"],
        top_n=top_n, exclude_same_artist=exclude_same_artist,
    )
    return {
        "query_track": {"name": target["name"], "artists": target["artists"]},
        "ranking_method": "euclidean_distance_on_standardized_audio_features",
        "interpretation": "Lower distance indicates closer numeric audio-feature vectors; it is not an accuracy or probability score.",
        "recommendations": results.to_dict(orient="records"),
    }
