"""
run_pipeline.py
---------------
Runs the full data preprocessing pipeline in order:

    Step 1: parse_tweets       → user_sequences.pkl, rt_sequences.pkl
    Step 2: ideology_scorer    → ideology_map.pkl, scored_rt_sequences.pkl,
                                  user_ideology_states.pkl
    Step 3: parse_graph        → graph_data.pkl

Usage:
    python run_pipeline.py \
        --tweets   data/raw/USER_TWEETS.txt.gz \
        --barbera  data/raw/USER_POLARITY_BARBERA.txt \
        --follower data/raw/FULL_FOLLOWER_NETWORK.txt \
        --friend   data/raw/FULL_FRIEND_NETWORK.txt \
        --output   data/processed
"""

import argparse
import pickle
from pathlib import Path

from data.parse_tweets    import parse_tweets
from data.ideology_scorer import run as score_run
from data.parse_graph     import build_graph


def run_pipeline(
    tweets_path:   str,
    barbera_path:  str,
    follower_path: str,
    friend_path:   str,
    output_dir:    str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Parsing tweets")
    print("=" * 60)
    parse_tweets(tweets_path, output_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Scoring ideology")
    print("=" * 60)
    rt_seq_path = output_dir / "rt_sequences.pkl"
    score_run(barbera_path, rt_seq_path, output_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Building graph")
    print("=" * 60)
    ideology_map_path = output_dir / "ideology_map.pkl"
    with open(ideology_map_path, "rb") as f:
        ideology_map = pickle.load(f)
    build_graph(follower_path, friend_path, ideology_map, output_dir)

    print("\n" + "=" * 60)
    print("Pipeline complete. Processed files in:", output_dir)
    print("=" * 60)
    for p in sorted(output_dir.glob("*.pkl")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<40} {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweets",   required=True)
    parser.add_argument("--barbera",  required=True)
    parser.add_argument("--follower", required=True)
    parser.add_argument("--friend",   required=True)
    parser.add_argument("--output",   default="data/processed")
    args = parser.parse_args()

    run_pipeline(
        args.tweets,
        args.barbera,
        args.follower,
        args.friend,
        args.output,
    )
