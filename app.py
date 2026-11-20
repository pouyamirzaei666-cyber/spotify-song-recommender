"""Streamlit interface for exploring similar music tracks."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src.recommender import AUDIO_FEATURES, closest_feature_dimensions, load_artifacts, recommend_by_index, search_songs

st.set_page_config(page_title="Music Discovery | Song Recommender", page_icon="🎧", layout="wide")
st.markdown("""
<style>
:root { --accent: #19c5a2; }
.block-container { padding-top: 4.5rem !important; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stAppViewContainer"] { background: linear-gradient(145deg, #08131d 0%, #0c1e28 48%, #101d2b 100%); color: #f1f5f9; }
[data-testid="stSidebar"] { background: #0a1721; }
[data-testid="stMetric"] { background: rgba(25,197,162,.08); border: 1px solid rgba(25,197,162,.23); padding: 12px; border-radius: 14px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], label { color: #e2e8f0 !important; }
div.stButton > button[kind="primary"] { background: #16b998; border: 0; color: #041a1b; font-weight: 700; border-radius: 10px; }
div.stButton > button[kind="primary"]:hover { background: #39d5b5; color: #041a1b; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: rgba(90,180,171,.25); background: rgba(255,255,255,.025); border-radius: 14px; }
.small-muted { color: #9fb2c2; font-size: .88rem; }
.pill { color: #6be5c8; font-size: .78rem; letter-spacing: .08em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the recommendation engine...")
def get_artifacts():
    return load_artifacts()

st.markdown('<div class="pill">CONTENT-BASED MUSIC DISCOVERY</div>', unsafe_allow_html=True)
st.title("🎧 Music Discovery")
st.caption("Find tracks with similar audio characteristics using a nearest-neighbor recommendation model.")

try:
    artifacts = get_artifacts()
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(str(exc))
    st.info("Place the source CSV in `data/`, install requirements, then run `python train_model.py`.")
    st.stop()

songs = artifacts["songs"]
scaled = artifacts["scaled_features"]
neighbor_model = artifacts["neighbor_model"]

with st.container(border=True):
    st.subheader("Find a track")
    query_col, help_col = st.columns([3, 1])
    with query_col:
        query = st.text_input("Search by title or artist", placeholder="e.g. Ordinary or Ariana Grande")
    with help_col:
        st.metric("Available tracks", f"{len(songs):,}")

    if query.strip():
        matches = search_songs(songs, query, limit=50)
        if matches.empty:
            st.warning("No matching track was found. Try another title or artist.")
        else:
            option_pairs = []
            for idx, row in matches.iterrows():
                label = f"{row['name']} — {row['artists']}  |  row {idx + 1}"
                option_pairs.append((label, int(idx)))
            labels = [p[0] for p in option_pairs]
            label_to_index = {p[0]: p[1] for p in option_pairs}
            selected_label = st.selectbox("Choose the exact track", labels)
            selected_index = label_to_index[selected_label]
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                top_n = st.slider("Recommendations", min_value=5, max_value=20, value=10)
            with col_b:
                exclude_same_artist = st.checkbox("Exclude same artist", value=False)
            with col_c:
                st.write("")
                run_search = st.button("Find similar tracks", type="primary", use_container_width=True)

            if run_search:
                target = songs.iloc[selected_index]
                st.markdown("### Selected track")
                st.markdown(f"**{target['name']}**")
                st.caption(str(target['artists']))
                spotify_id = target.get("spotify_id")
                if pd.notna(spotify_id):
                    st.link_button("Open track in Spotify", f"https://open.spotify.com/track/{spotify_id}")

                results = recommend_by_index(
                    track_index=selected_index,
                    songs=songs,
                    scaled_features=scaled,
                    neighbor_model=neighbor_model,
                    top_n=top_n,
                    exclude_same_artist=exclude_same_artist,
                )
                st.markdown("### Recommended tracks")
                st.caption("Ranked by Euclidean distance between standardized audio-feature vectors. Lower distance means closer numeric features—not a probability, accuracy score, or guarantee of musical preference.")

                if results.empty:
                    st.info("No recommendations are available for this selection.")
                else:
                    for rank, row in results.reset_index(drop=True).iterrows():
                        with st.container(border=True):
                            left, middle, right = st.columns([4, 1, 1])
                            with left:
                                st.markdown(f"**{rank + 1}. {row['name']}**")
                                st.caption(row["artists"])
                                # Explain the closest dimensions, not an unvalidated subjective match.
                                candidate_idx = []
                                result_id = row.get("spotify_id")
                                if pd.notna(result_id) and "spotify_id" in songs.columns:
                                    candidate_idx = songs.index[songs["spotify_id"].astype(str) == str(result_id)].tolist()
                                if not candidate_idx:
                                    candidate_idx = songs.index[(songs["name"] == row["name"]) & (songs["artists"] == row["artists"])].tolist()
                                if candidate_idx:
                                    dims = closest_feature_dimensions(scaled[selected_index], scaled[candidate_idx[0]], top_k=3)
                                    st.caption("Closest feature dimensions: " + " · ".join(dims))
                            with middle:
                                st.metric("Feature distance", f"{row['feature_distance']:.3f}")
                            with right:
                                if pd.notna(row.get("spotify_id")):
                                    st.link_button("Open", f"https://open.spotify.com/track/{row['spotify_id']}")
                    with st.expander("View results as a table"):
                        st.dataframe(results[["name", "artists", "feature_distance"]], use_container_width=True, hide_index=True)

                with st.expander("Inspect selected audio features"):
                    available = [f for f in AUDIO_FEATURES if f in target.index and pd.notna(target[f])]
                    features_df = pd.DataFrame({"Feature": available, "Value": [float(target[f]) for f in available]})
                    st.dataframe(features_df, use_container_width=True, hide_index=True)
    else:
        st.info("Enter a track title or artist to start discovering similar tracks.")
        st.markdown("""**How it works**
1. Search for a track.
2. Select the exact title and artist.
3. Retrieve nearest tracks by standardized audio-feature distance.""")

st.divider()
st.caption("Independent portfolio project. Not affiliated with or endorsed by Spotify.")
