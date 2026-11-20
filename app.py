"""Streamlit interface for exploring similar music tracks."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from src.recommender import (
    AUDIO_FEATURES,
    closest_feature_dimensions,
    load_artifacts,
    recommend_by_index,
    search_songs,
)

st.set_page_config(
    page_title="Music Discovery | Song Recommender",
    page_icon="🎧",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --accent: #19c5a2; }
    .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(25,197,162,.13), transparent 30%),
          linear-gradient(145deg, #07111b 0%, #0b1c27 52%, #101b2a 100%);
        color: #f1f5f9;
    }
    .block-container { max-width: 1220px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #091722; }
    [data-testid="stMetric"] {
        background: rgba(25,197,162,.08);
        border: 1px solid rgba(25,197,162,.22);
        padding: 12px; border-radius: 14px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], label { color: #e2e8f0 !important; }
    div.stButton > button[kind="primary"] {
        background: #16b998; border: 0; color: #041a1b;
        font-weight: 700; border-radius: 10px;
    }
    div.stButton > button[kind="primary"]:hover { background: #39d5b5; color: #041a1b; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(90,180,171,.25);
        background: rgba(255,255,255,.025); border-radius: 16px;
    }
    .eyebrow { color: #6be5c8; font-size: .76rem; letter-spacing: .13em; font-weight: 800; }
    .hero-note { color: #a9bac8; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalise_text(value: str) -> str:
    """Make title/artist comparisons tolerant of punctuation and case."""
    value = str(value).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


@st.cache_data(ttl=86400, show_spinner=False)
def find_album_artwork(track_name: str, artist_name: str) -> str | None:
    """Find a likely exact album-cover match using Apple's public Search API.

    A result is used only when both title and artist are close matches. If no
    convincing match is found, the UI deliberately shows a neutral placeholder
    instead of an unrelated image.
    """
    query = f"{track_name} {artist_name}"
    params = urlencode({"term": query, "entity": "song", "limit": 12})
    request = Request(
        f"https://itunes.apple.com/search?{params}",
        headers={"User-Agent": "MusicDiscoveryPortfolio/1.0"},
    )
    try:
        with urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    wanted_title = _normalise_text(track_name)
    wanted_artist = _normalise_text(artist_name)
    best_url = None
    best_score = 0.0

    for item in payload.get("results", []):
        candidate_title = _normalise_text(item.get("trackName", ""))
        candidate_artist = _normalise_text(item.get("artistName", ""))
        title_score = SequenceMatcher(None, wanted_title, candidate_title).ratio()
        artist_score = SequenceMatcher(None, wanted_artist, candidate_artist).ratio()
        score = 0.72 * title_score + 0.28 * artist_score
        if title_score >= 0.88 and artist_score >= 0.72 and score > best_score:
            best_score = score
            best_url = item.get("artworkUrl100", "").replace("100x100bb", "600x600bb")

    return best_url or None


@st.cache_resource(show_spinner="Loading the recommendation engine...")
def get_artifacts():
    return load_artifacts()


def show_track_art(track_name: str, artist_name: str, width: int = 150) -> None:
    """Render matched cover art or a consistent branded fallback."""
    image_url = find_album_artwork(track_name, artist_name)
    if image_url:
        st.image(image_url, width=width)
    else:
        st.markdown(
            """
            <div style="
              width:100%; aspect-ratio:1/1; border-radius:14px;
              display:flex; align-items:center; justify-content:center;
              background:linear-gradient(135deg,#123c43,#172638 60%,#26344b);
              border:1px solid rgba(107,229,200,.25); color:#6be5c8;
              font-size:3rem;">♫</div>
            """,
            unsafe_allow_html=True,
        )


st.markdown('<div class="eyebrow">CONTENT-BASED MUSIC DISCOVERY</div>', unsafe_allow_html=True)
st.title("🎧 Music Discovery")
st.markdown(
    '<p class="hero-note">Discover tracks with similar audio characteristics — '
    'powered by a machine-learning nearest-neighbor model.</p>',
    unsafe_allow_html=True,
)

try:
    artifacts = get_artifacts()
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(str(exc))
    st.info("The model files are not available. The dataset and trained artifacts must be included in the deployment environment.")
    st.stop()

songs = artifacts["songs"]
scaled = artifacts["scaled_features"]
neighbor_model = artifacts["neighbor_model"]

with st.container(border=True):
    st.subheader("Find your next track")
    query_col, count_col = st.columns([3, 1])
    with query_col:
        query = st.text_input(
            "Search by song title or artist",
            placeholder="Try a song title or artist name…",
            label_visibility="visible",
        )
    with count_col:
        st.metric("Tracks in this model", f"{len(songs):,}")

    if query.strip():
        matches = search_songs(songs, query, limit=50)
        if matches.empty:
            st.warning("No matching track was found. Try a different title or artist.")
        else:
            option_pairs = []
            for idx, row in matches.iterrows():
                label = f"{row['name']} — {row['artists']}  |  result {len(option_pairs) + 1}"
                option_pairs.append((label, int(idx)))
            labels = [pair[0] for pair in option_pairs]
            label_to_index = {pair[0]: pair[1] for pair in option_pairs}
            selected_label = st.selectbox("Choose the exact track", labels)
            selected_index = label_to_index[selected_label]

            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                top_n = st.slider("Number of recommendations", 5, 20, 10)
            with col_b:
                exclude_same_artist = st.checkbox("Exclude the same artist", value=False)
            with col_c:
                st.write("")
                run_search = st.button(
                    "Find similar tracks",
                    type="primary",
                    use_container_width=True,
                )

            if run_search:
                target = songs.iloc[selected_index]
                st.markdown("## Your selected track")
                target_left, target_right = st.columns([1, 4])
                with target_left:
                    show_track_art(str(target["name"]), str(target["artists"]), width=170)
                with target_right:
                    st.markdown(f"### {target['name']}")
                    st.caption(str(target["artists"]))
                    spotify_id = target.get("spotify_id")
                    if pd.notna(spotify_id):
                        st.link_button(
                            "Open original track in Spotify",
                            f"https://open.spotify.com/track/{spotify_id}",
                        )

                results = recommend_by_index(
                    track_index=selected_index,
                    songs=songs,
                    scaled_features=scaled,
                    neighbor_model=neighbor_model,
                    top_n=top_n,
                    exclude_same_artist=exclude_same_artist,
                )

                st.markdown("## Recommended for you")
                st.caption(
                    "Ranked by Euclidean distance between standardized audio-feature vectors. "
                    "A smaller distance means closer numeric features — not a probability, "
                    "accuracy score, or guarantee of personal preference."
                )

                if results.empty:
                    st.info("No recommendations are available for this selection.")
                else:
                    for rank, row in results.reset_index(drop=True).iterrows():
                        with st.container(border=True):
                            art_col, info_col, metric_col = st.columns([1, 4, 1.25])
                            with art_col:
                                show_track_art(str(row["name"]), str(row["artists"]), width=112)
                            with info_col:
                                st.markdown(f"**{rank + 1}. {row['name']}**")
                                st.caption(row["artists"])

                                candidate_idx = []
                                result_id = row.get("spotify_id")
                                if pd.notna(result_id) and "spotify_id" in songs.columns:
                                    candidate_idx = songs.index[
                                        songs["spotify_id"].astype(str) == str(result_id)
                                    ].tolist()
                                if not candidate_idx:
                                    candidate_idx = songs.index[
                                        (songs["name"] == row["name"])
                                        & (songs["artists"] == row["artists"])
                                    ].tolist()

                                if candidate_idx:
                                    dimensions = closest_feature_dimensions(
                                        scaled[selected_index],
                                        scaled[candidate_idx[0]],
                                        top_k=3,
                                    )
                                    if dimensions:
                                        st.caption("Closest audio features: " + " · ".join(dimensions))

                                if pd.notna(row.get("spotify_id")):
                                    st.link_button(
                                        "Listen on Spotify",
                                        f"https://open.spotify.com/track/{row['spotify_id']}",
                                    )
                            with metric_col:
                                st.metric("Feature distance", f"{row['feature_distance']:.3f}")

                    with st.expander("View recommendations as a table"):
                        st.dataframe(
                            results[["name", "artists", "feature_distance"]],
                            use_container_width=True,
                            hide_index=True,
                        )

                with st.expander("Explore the selected track's audio features"):
                    available = [
                        feature for feature in AUDIO_FEATURES
                        if feature in target.index and pd.notna(target[feature])
                    ]
                    features_df = pd.DataFrame(
                        {
                            "Audio feature": available,
                            "Value": [float(target[feature]) for feature in available],
                        }
                    )
                    st.bar_chart(features_df.set_index("Audio feature"))
                    st.dataframe(features_df, use_container_width=True, hide_index=True)
    else:
        st.info("Enter a song title or artist above to start discovering similar music.")
        st.markdown(
            """
            **How it works**
            1. Search for a track and select the exact match.
            2. The model compares its nine standardized audio features.
            3. Explore the nearest tracks, their cover artwork, and the closest feature dimensions.
            """
        )

st.divider()
st.caption(
    "Independent portfolio project. Album artwork is retrieved from the Apple iTunes Search API when a confident title-and-artist match is available. "
    "Not affiliated with or endorsed by Spotify or Apple."
)
