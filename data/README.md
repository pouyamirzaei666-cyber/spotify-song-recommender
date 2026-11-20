# Dataset

The training script expects `universal_top_spotify_songs.csv` in this folder.

The user-supplied file inspected for this portfolio build contains 2,110,316 rows and 25 columns, including nine audio features. After removing incomplete required fields and duplicate Spotify track IDs (falling back to `(name, artists)` only when an ID is missing), 24,975 unique track records remain. The reproducible modeling sample contains 12,000 tracks (`random_state=42`).

**The dataset is not included.** Its source and redistribution license have not been verified. Do not commit or redistribute the CSV until its license and attribution requirements are confirmed.
