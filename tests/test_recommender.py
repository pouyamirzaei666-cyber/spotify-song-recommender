import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from src.recommender import search_songs, recommend_by_index, closest_feature_dimensions


def sample_objects():
    songs = pd.DataFrame({
        "name": ["Track A", "Track B", "Track C", "Track D"],
        "artists": ["Artist 1", "Artist 2", "Artist 1", "Artist 3"],
        "spotify_id": ["a", "b", "c", "d"],
    })
    vectors = np.array([[0.0, 0.0], [0.2, 0.1], [0.0, 0.3], [2.0, 2.0]])
    model = NearestNeighbors(metric="euclidean").fit(vectors)
    return songs, vectors, model


def test_search_matches_title_and_artist_literally():
    songs, _, _ = sample_objects()
    assert len(search_songs(songs, "Track A")) == 1
    assert len(search_songs(songs, "Artist 1")) == 2
    assert search_songs(songs, "not-found").empty
    assert search_songs(songs, "[").empty


def test_recommend_excludes_query_and_sorts_by_distance():
    songs, vectors, model = sample_objects()
    result = recommend_by_index(0, songs, vectors, model, top_n=2)
    assert "Track A" not in result["name"].tolist()
    assert result["feature_distance"].is_monotonic_increasing
    assert result.iloc[0]["name"] == "Track B"


def test_recommend_can_exclude_same_artist():
    songs, vectors, model = sample_objects()
    result = recommend_by_index(0, songs, vectors, model, top_n=2, exclude_same_artist=True)
    assert all(artist != "Artist 1" for artist in result["artists"])


def test_invalid_track_index_raises():
    songs, vectors, model = sample_objects()
    with pytest.raises(IndexError):
        recommend_by_index(10, songs, vectors, model)


def test_closest_dimensions_returns_labels():
    target = np.zeros(9)
    candidate = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    labels = closest_feature_dimensions(target, candidate, top_k=2)
    assert labels == ["Danceability", "Loudness"]
