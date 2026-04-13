"""
parse_tweets.py
---------------
Parses USER_TWEETS.txt(.gz) into per-user RT interaction sequences.

Actual column layout (tab-separated, no header):
    0: user_id
    1: tweet_id
    2: tweet_text
    3: timestamp          ("Mon Oct 09 04:37:29 +0000 2017" format)
    4: source             (HTML anchor, ignored)
    5: retweet_count      (ignored)
    6: unknown_flag       (ignored)
    7: unknown_int        (ignored)
    8: (empty / ignored)
    9: urls               (space/pipe-separated, may be empty)
   10+: (ignored)

Retweet detection:
    A row is a retweet iff tweet_text starts with "RT @username".
    The retweeted screen name is always parsed from that text prefix —
    column 10 and beyond are never used.
    To resolve screen name → user_id, supply --username_map <path>:
        a TSV with  screen_name<TAB>user_id  (case-insensitive match).

OUTPUT (written to output_dir):
    user_sequences.pkl   {user_id: [tweet_record, ...]}  sorted by timestamp
    rt_sequences.pkl     {user_id: [retweeted_user_id, ...]}  RT-only, sorted
                         retweeted_user_id is the resolved user_id when the map
                         is available, otherwise the raw screen name.
"""

import gzip
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Column indices (0-based) ──────────────────────────────────────────────────
COL_USER_ID             = 0
COL_TWEET_ID            = 1
COL_TEXT                = 2
COL_TIMESTAMP           = 3
COL_URLS                = 9

# Minimum RT events per user to keep in rt_sequences
MIN_RT_SEQUENCE_LEN     = 5

# Minimum columns a valid row must have
MIN_COLUMNS             = COL_URLS + 1   # 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(ts_str: str) -> float:
    """Parse Twitter-format timestamp → Unix float."""
    return datetime.strptime(ts_str.strip(), "%a %b %d %H:%M:%S +0000 %Y").timestamp()


def rt_screen_name_from_text(text: str) -> str | None:
    """Return the screen name from 'RT @screen_name: ...' or None."""
    m = re.match(r"RT @(\w+)", text.strip())
    return m.group(1) if m else None


def load_username_map(path: str | Path) -> dict[str, str]:
    """
    Load a TSV mapping  screen_name -> user_id.
    Keys are stored lower-cased for case-insensitive lookup.
    """
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0].strip().lower()] = parts[1].strip()
    print(f"Loaded username map: {len(mapping):,} screen_name->user_id entries")
    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_tweets(
    tweets_path: str | Path,
    output_dir: str | Path,
    username_map_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """
    Parse USER_TWEETS.txt(.gz) and produce per-user tweet / RT sequences.

    Parameters
    ----------
    tweets_path       : path to the raw tweet file (.txt or .txt.gz)
    output_dir        : directory where .pkl outputs are written
    username_map_path : optional TSV  screen_name<TAB>user_id
                        used to resolve retweeted screen names to user IDs
    """
    tweets_path = Path(tweets_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    username_map: dict[str, str] = {}
    if username_map_path:
        username_map = load_username_map(username_map_path)

    user_sequences: dict[str, list[dict]] = defaultdict(list)
    total = skipped = rt_count = 0

    open_fn = gzip.open if str(tweets_path).endswith(".gz") else open
    with open_fn(tweets_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            parts = line.rstrip("\n").split("\t")

            if len(parts) < MIN_COLUMNS:
                skipped += 1
                continue

            try:
                user_id  = parts[COL_USER_ID].strip()
                tweet_id = parts[COL_TWEET_ID].strip()
                text     = parts[COL_TEXT].strip()
                ts       = parse_timestamp(parts[COL_TIMESTAMP])
                urls     = parts[COL_URLS].strip()
            except (ValueError, IndexError):
                skipped += 1
                continue

            # ── Retweet detection: parse screen name from text only ────────
            rt_screen_name = rt_screen_name_from_text(text)
            is_rt = rt_screen_name is not None

            # ── Resolve retweeted screen name → user_id ────────────────────
            retweeted_user_id = ""
            if is_rt and rt_screen_name:
                retweeted_user_id = username_map.get(
                    rt_screen_name.lower(),
                    rt_screen_name,   # fall back to screen name if not in map
                )

            user_sequences[user_id].append({
                "tweet_id"             : tweet_id,
                "timestamp"            : ts,
                "text"                 : text,
                "urls"                 : urls,
                "is_retweet"           : is_rt,
                "retweeted_screen_name": rt_screen_name or "",
                "retweeted_user_id"    : retweeted_user_id,
            })
            if is_rt:
                rt_count += 1

    # Sort each user's timeline chronologically
    for uid in user_sequences:
        user_sequences[uid].sort(key=lambda x: x["timestamp"])

    # RT-only sequences: list of retweeted user IDs, filtered by min length
    rt_sequences: dict[str, list[str]] = {
        uid: [r["retweeted_user_id"] for r in recs
              if r["is_retweet"] and r["retweeted_user_id"]]
        for uid, recs in user_sequences.items()
    }
    rt_sequences = {u: s for u, s in rt_sequences.items()
                    if len(s) >= MIN_RT_SEQUENCE_LEN}

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"Total lines     : {total:,}")
    print(f"Skipped         : {skipped:,}")
    print(f"Unique users    : {len(user_sequences):,}")
    print(f"Total RTs       : {rt_count:,}")
    print(f"Users w/ RT≥{MIN_RT_SEQUENCE_LEN}  : {len(rt_sequences):,}")
    lengths = [len(v) for v in rt_sequences.values()]
    if lengths:
        print(f"RT seq lengths  : min={min(lengths)}  max={max(lengths)}  "
              f"mean={sum(lengths)/len(lengths):.1f}")

    # ── Save ───────────────────────────────────────────────────────────────
    with open(output_dir / "user_sequences.pkl", "wb") as fh:
        pickle.dump(dict(user_sequences), fh)
    with open(output_dir / "rt_sequences.pkl", "wb") as fh:
        pickle.dump(rt_sequences, fh)

    print(f"\nSaved → {output_dir}/user_sequences.pkl")
    print(f"Saved → {output_dir}/rt_sequences.pkl")
    return dict(user_sequences), rt_sequences


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Parse USER_TWEETS.txt into RT sequences.")
    p.add_argument("--tweets",        required=True,
                   help="Path to USER_TWEETS.txt or USER_TWEETS.txt.gz")
    p.add_argument("--output_dir",    default="data/processed",
                   help="Directory for output .pkl files (default: data/processed)")
    p.add_argument("--username_map",  default=None,
                   help="Optional TSV: screen_name<TAB>user_id  "
                        "(used to resolve retweeted usernames to user IDs)")
    args = p.parse_args()
    parse_tweets(args.tweets, args.output_dir, args.username_map)