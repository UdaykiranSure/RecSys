"""
parse_tweets.py
---------------
Parses USER_TWEETS.txt.gz into per-user RT interaction sequences.

Expected columns (tab-separated, no header):
    tweet_id | timestamp | tweet_text | retweeted_user_id | mentions | URLs

NOTE ON user_id:
    user_id is not present as a direct column. Two strategies:
    (A) Set HAS_USER_ID_COLUMN = True if it's actually the first column.
    (B) Provide an external tweet_id -> user_id mapping file (TSV).
        Pass --user_map <path> at the command line.

OUTPUT (written to processed_dir):
    user_sequences.pkl   {user_id: [tweet_record, ...]}  sorted by timestamp
    rt_sequences.pkl     {user_id: [retweeted_user_id, ...]}  RT-only, sorted
"""

import gzip
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Column layout (0-based, before any user_id offset) ────────────────────────
COL_TWEET_ID          = 0
COL_TIMESTAMP         = 1
COL_TEXT              = 2
COL_RETWEETED_USER_ID = 3
COL_MENTIONS          = 4
COL_URLS              = 5

# Set True if user_id is an extra first column in your file
HAS_USER_ID_COLUMN    = False

# Timestamp format: "unix" | "twitter" | "iso"
TIMESTAMP_FORMAT      = "unix"

# Minimum RT events per user to keep
MIN_RT_SEQUENCE_LEN   = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(ts_str: str) -> float:
    ts_str = ts_str.strip()
    if TIMESTAMP_FORMAT == "unix":
        return float(ts_str)
    elif TIMESTAMP_FORMAT == "twitter":
        return datetime.strptime(ts_str, "%a %b %d %H:%M:%S +0000 %Y").timestamp()
    elif TIMESTAMP_FORMAT == "iso":
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
    raise ValueError(f"Unknown TIMESTAMP_FORMAT: {TIMESTAMP_FORMAT}")


def is_retweet(text: str, rt_uid: str) -> bool:
    has_field = rt_uid and rt_uid.strip().lower() not in ("", "null", "none", "nan")
    has_text  = text.strip().startswith("RT @")
    return bool(has_field or has_text)


def rt_user_from_text(text: str) -> str | None:
    m = re.match(r"RT @(\w+)", text.strip())
    return m.group(1) if m else None


def load_user_map(path: str | Path) -> dict[str, str]:
    """Load optional tweet_id -> user_id TSV map."""
    user_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                user_map[parts[0].strip()] = parts[1].strip()
    print(f"Loaded user map: {len(user_map):,} tweet->user entries")
    return user_map


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_tweets(
    tweets_path: str | Path,
    output_dir: str | Path,
    user_map_path: str | Path | None = None,
) -> tuple[dict, dict]:

    tweets_path = Path(tweets_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_map: dict[str, str] = {}
    if user_map_path:
        user_map = load_user_map(user_map_path)

    offset = 1 if HAS_USER_ID_COLUMN else 0
    c_tid  = COL_TWEET_ID          + offset
    c_ts   = COL_TIMESTAMP         + offset
    c_txt  = COL_TEXT              + offset
    c_ruid = COL_RETWEETED_USER_ID + offset
    c_men  = COL_MENTIONS          + offset
    c_url  = COL_URLS              + offset
    min_c  = c_url + 1

    user_sequences: dict[str, list[dict]] = defaultdict(list)
    total = skipped = rt_count = 0

    open_fn = gzip.open if str(tweets_path).endswith(".gz") else open
    with open_fn(tweets_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < min_c:
                skipped += 1
                continue

            try:
                tweet_id = parts[c_tid].strip()
                ts       = parse_timestamp(parts[c_ts])
                text     = parts[c_txt].strip()
                rt_uid   = parts[c_ruid].strip()
                mentions = parts[c_men].strip()
                urls     = parts[c_url].strip()
            except (ValueError, IndexError):
                skipped += 1
                continue

            # Resolve user_id
            if HAS_USER_ID_COLUMN:
                user_id = parts[0].strip()
            elif tweet_id in user_map:
                user_id = user_map[tweet_id]
            else:
                # Fallback placeholder — replace with real resolution
                user_id = f"UNKNOWN_{tweet_id}"

            rt_flag = is_retweet(text, rt_uid)
            if rt_flag and not rt_uid:
                rt_uid = rt_user_from_text(text) or ""

            user_sequences[user_id].append({
                "tweet_id"          : tweet_id,
                "timestamp"         : ts,
                "text"              : text,
                "retweeted_user_id" : rt_uid,
                "mentions"          : mentions,
                "urls"              : urls,
                "is_retweet"        : rt_flag,
            })
            if rt_flag:
                rt_count += 1

    # Sort each user's timeline
    for uid in user_sequences:
        user_sequences[uid].sort(key=lambda x: x["timestamp"])

    # RT-only sequences with minimum length filter
    rt_sequences: dict[str, list[str]] = {
        uid: [r["retweeted_user_id"] for r in recs
              if r["is_retweet"] and r["retweeted_user_id"]]
        for uid, recs in user_sequences.items()
    }
    rt_sequences = {u: s for u, s in rt_sequences.items()
                    if len(s) >= MIN_RT_SEQUENCE_LEN}

    # Stats
    print(f"Total lines    : {total:,}")
    print(f"Skipped        : {skipped:,}")
    print(f"Unique users   : {len(user_sequences):,}")
    print(f"Total RTs      : {rt_count:,}")
    print(f"Users w/ RT≥{MIN_RT_SEQUENCE_LEN} : {len(rt_sequences):,}")
    lengths = [len(v) for v in rt_sequences.values()]
    if lengths:
        print(f"RT seq lengths : min={min(lengths)} max={max(lengths)} "
              f"mean={sum(lengths)/len(lengths):.1f}")

    # Save
    with open(output_dir / "user_sequences.pkl", "wb") as f:
        pickle.dump(dict(user_sequences), f)
    with open(output_dir / "rt_sequences.pkl", "wb") as f:
        pickle.dump(rt_sequences, f)

    print(f"\nSaved → {output_dir}/user_sequences.pkl")
    print(f"Saved → {output_dir}/rt_sequences.pkl")
    return dict(user_sequences), rt_sequences


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tweets",      required=True)
    p.add_argument("--output_dir",  default="data/processed")
    p.add_argument("--user_map",    default=None,
                   help="Optional TSV: tweet_id<tab>user_id")
    args = p.parse_args()
    parse_tweets(args.tweets, args.output_dir, args.user_map)
