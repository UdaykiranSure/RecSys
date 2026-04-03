"""
dataset.py
----------
PyTorch Dataset for the sequential ideology recommender.

Each training sample is:
    (user_id, seq_of_item_ids, seq_ideology_states,
     target_item_id, target_ideology,
     ideo_current, direction, delta)

Temporal split (per user, never shuffle):
    train : items [0 .. n-test_holdout-val_holdout-1]
    val   : item  [n-test_holdout-val_holdout]
    test  : item  [n-test_holdout]

Item vocabulary:
    Items are retweeted_user_ids (the "tweet authors" being recommended).
    Each unique retweeted_user_id is mapped to an integer item index.
    PAD_IDX = 0  (reserved for padding shorter sequences)
"""

import pickle
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_IDX = 0


# ── Item vocabulary ───────────────────────────────────────────────────────────

def build_item_vocab(
    scored_rt_sequences: dict[str, list[tuple[str, float]]],
    min_freq: int = 5,
) -> tuple[dict[str, int], dict[int, str], dict[int, float]]:
    """
    Build item vocabulary from scored RT sequences.

    Returns
    -------
    item2idx   : {retweeted_user_id: int}   (1-indexed; 0 = PAD)
    idx2item   : {int: retweeted_user_id}
    item_ideo  : {item_idx: ideology_score}
    """
    from collections import Counter
    freq: Counter = Counter()
    item_scores: dict[str, list[float]] = {}

    for seq in scored_rt_sequences.values():
        for uid, score in seq:
            freq[uid] += 1
            item_scores.setdefault(uid, []).append(score)

    # Filter by minimum frequency
    valid = {uid for uid, cnt in freq.items() if cnt >= min_freq}
    print(f"Item vocab: {len(valid):,} items  "
          f"(min_freq={min_freq}, dropped {len(freq)-len(valid):,})")

    item2idx: dict[str, int] = {uid: i+1 for i, uid in enumerate(sorted(valid))}
    idx2item: dict[int, str] = {i: uid for uid, i in item2idx.items()}

    # Item ideology = mean Barberá score across all observations
    item_ideo: dict[int, float] = {
        item2idx[uid]: float(np.mean(scores))
        for uid, scores in item_scores.items()
        if uid in item2idx
    }

    return item2idx, idx2item, item_ideo


# ── Dataset ───────────────────────────────────────────────────────────────────

class IdeologySeqDataset(Dataset):
    """
    Parameters
    ----------
    scored_rt_sequences  : {user_id: [(item_id_str, ideo_score), ...]}
    user_ideology_states : {user_id: [ideo_current_at_t, ...]}
    item2idx             : {item_str: int}
    item_ideo            : {item_idx: float}
    user2idx             : {user_id: int}  (for graph lookup)
    split                : "train" | "val" | "test"
    max_seq_len          : truncate/pad history to this length
    val_holdout          : N items held for val from end
    test_holdout         : N items held for test from end
    delta                : fixed ideology step size
    """

    def __init__(
        self,
        scored_rt_sequences:  dict[str, list[tuple[str, float]]],
        user_ideology_states: dict[str, list[float]],
        item2idx:             dict[str, int],
        item_ideo:            dict[int, float],
        user2idx:             dict[str, int],
        split:                Literal["train", "val", "test"] = "train",
        max_seq_len:          int   = 50,
        val_holdout:          int   = 1,
        test_holdout:         int   = 1,
        delta:                float = 0.2,
    ):
        self.item2idx   = item2idx
        self.item_ideo  = item_ideo
        self.user2idx   = user2idx
        self.max_seq_len = max_seq_len
        self.delta       = delta

        self.samples: list[dict] = []
        self._build_samples(
            scored_rt_sequences,
            user_ideology_states,
            split, val_holdout, test_holdout,
        )

    def _build_samples(
        self,
        scored_rt_sequences,
        user_ideology_states,
        split, val_holdout, test_holdout,
    ):
        total_holdout = val_holdout + test_holdout

        for user_id, seq in scored_rt_sequences.items():
            if user_id not in user_ideology_states:
                continue

            states = user_ideology_states[user_id]

            # Filter sequence to known items only
            filtered = [
                (uid, sc, st)
                for (uid, sc), st in zip(seq, states)
                if uid in self.item2idx
            ]
            if len(filtered) < total_holdout + 1:
                continue

            n = len(filtered)

            # Determine target index for this split
            if split == "train":
                target_indices = range(1, n - total_holdout)
            elif split == "val":
                target_indices = [n - total_holdout]
            else:  # test
                target_indices = [n - test_holdout]

            for t in target_indices:
                # History: items before t, truncated to max_seq_len
                history_raw = filtered[max(0, t - self.max_seq_len): t]
                history_items  = [self.item2idx[uid] for uid, _, _ in history_raw]
                history_states = [st for _, _, st in history_raw]

                target_uid, target_score, ideo_current = filtered[t]
                target_idx = self.item2idx[target_uid]

                # Nudge direction: toward center (sign of mean ideology)
                # This is a placeholder — bandit module will supply this later
                mean_ideo = float(np.mean(history_states)) if history_states else 0.0
                direction = -1.0 if mean_ideo > 0 else 1.0

                self.samples.append({
                    "user_id"       : user_id,
                    "user_idx"      : self.user2idx.get(user_id, 0),
                    "history_items" : history_items,
                    "history_states": history_states,
                    "target_item"   : target_idx,
                    "target_ideo"   : target_score,
                    "ideo_current"  : ideo_current,
                    "direction"     : direction,
                    "delta"         : self.delta,
                })

        print(f"[{split}] {len(self.samples):,} samples built")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Pad / truncate history
        hist = s["history_items"]
        hist_states = s["history_states"]
        L = self.max_seq_len

        if len(hist) < L:
            pad_len = L - len(hist)
            hist        = [PAD_IDX] * pad_len + hist
            hist_states = [0.0]    * pad_len + hist_states
        else:
            hist        = hist[-L:]
            hist_states = hist_states[-L:]

        return {
            "user_idx"      : torch.tensor(s["user_idx"],    dtype=torch.long),
            "history_items" : torch.tensor(hist,             dtype=torch.long),
            "history_states": torch.tensor(hist_states,      dtype=torch.float32),
            "target_item"   : torch.tensor(s["target_item"], dtype=torch.long),
            "target_ideo"   : torch.tensor(s["target_ideo"], dtype=torch.float32),
            "ideo_current"  : torch.tensor(s["ideo_current"],dtype=torch.float32),
            "direction"     : torch.tensor(s["direction"],   dtype=torch.float32),
            "delta"         : torch.tensor(s["delta"],       dtype=torch.float32),
        }


# ── Negative sampler ──────────────────────────────────────────────────────────

class NegativeSampler:
    """
    Samples negative items for BPR loss.

    Strategy "hard": negatives are sampled from the same ideology band
    as the positive item [ideo_pos ± band]. Forces the model to
    discriminate beyond raw ideology.

    Strategy "random": uniform sample from full item pool.
    """

    def __init__(
        self,
        item_ideo: dict[int, float],
        strategy:  str   = "hard",
        band:      float = 0.5,
    ):
        self.item_ideo = item_ideo
        self.strategy  = strategy
        self.band      = band
        self.all_items = list(item_ideo.keys())

        # Pre-bucket items by ideology for fast hard-negative lookup
        # Buckets of width 0.5 from -3 to +3
        self._buckets: dict[int, list[int]] = {}
        for item, sc in item_ideo.items():
            bucket = int((sc + 3) / 0.5)  # 0..11
            self._buckets.setdefault(bucket, []).append(item)

    def sample(self, pos_item: int, k: int = 1) -> list[int]:
        if self.strategy == "random":
            return random.choices(self.all_items, k=k)

        # Hard: sample from same ideology band
        pos_score = self.item_ideo.get(pos_item, 0.0)
        bucket    = int((pos_score + 3) / 0.5)
        pool = []
        for b in [bucket - 1, bucket, bucket + 1]:
            pool.extend(self._buckets.get(b, []))
        pool = [x for x in pool if x != pos_item]

        if not pool:
            return random.choices(self.all_items, k=k)
        return random.choices(pool, k=k)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack a list of sample dicts into batched tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ── Factory ───────────────────────────────────────────────────────────────────

def make_datasets(
    processed_dir: str | Path,
    max_seq_len:   int   = 50,
    val_holdout:   int   = 1,
    test_holdout:  int   = 1,
    min_item_freq: int   = 5,
    delta:         float = 0.2,
) -> tuple[IdeologySeqDataset, IdeologySeqDataset, IdeologySeqDataset,
           dict, dict, dict, dict, NegativeSampler]:

    d = Path(processed_dir)

    with open(d / "scored_rt_sequences.pkl",  "rb") as f:
        scored = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f:
        states = pickle.load(f)
    with open(d / "user2idx.pkl",             "rb") as f:
        user2idx = pickle.load(f)

    item2idx, idx2item, item_ideo = build_item_vocab(scored, min_item_freq)

    kwargs = dict(
        scored_rt_sequences  = scored,
        user_ideology_states = states,
        item2idx             = item2idx,
        item_ideo            = item_ideo,
        user2idx             = user2idx,
        max_seq_len          = max_seq_len,
        val_holdout          = val_holdout,
        test_holdout         = test_holdout,
        delta                = delta,
    )

    train_ds = IdeologySeqDataset(**kwargs, split="train")
    val_ds   = IdeologySeqDataset(**kwargs, split="val")
    test_ds  = IdeologySeqDataset(**kwargs, split="test")

    neg_sampler = NegativeSampler(item_ideo, strategy="hard", band=0.5)

    return train_ds, val_ds, test_ds, item2idx, idx2item, item_ideo, user2idx, neg_sampler
