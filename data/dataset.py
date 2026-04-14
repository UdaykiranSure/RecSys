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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PAD_IDX = 0


@dataclass
class ItemCatalog:
    """Catalog metadata used by training/evaluation."""

    item2idx: dict[str, int]
    idx2item: dict[int, str]
    item_ideo: np.ndarray

    @property
    def num_items(self) -> int:
        return len(self.item_ideo)


class CollateWithNegatives:
    """Picklable collate callable that injects negative samples."""

    def __init__(self, neg_sampler: "NegativeSampler", item_ideo_arr: np.ndarray, num_negatives: int):
        self.neg_sampler = neg_sampler
        self.item_ideo_arr = item_ideo_arr
        self.num_negatives = max(1, num_negatives)

    def __call__(self, batch: list[dict]) -> dict:
        out = collate_fn(batch)

        pos_items         = out["target_item"].tolist()
        ideo_current_list = out["ideo_current"].tolist()
        direction_list    = out["direction"].tolist()
        delta_list        = out["delta"].tolist()

        # Standard BPR negatives (same ideology band)
        neg_items    = [self.neg_sampler.sample(int(pos), k=self.num_negatives)[0] for pos in pos_items]
        neg_item_idx = torch.tensor(neg_items, dtype=torch.long)
        neg_ideo     = torch.tensor(self.item_ideo_arr[neg_item_idx.numpy()], dtype=torch.float32)
        out["neg_item_idx"] = neg_item_idx
        out["neg_ideo"]     = neg_ideo

        # Ideology-contrastive items
        aligned_items = [
            self.neg_sampler.sample_aligned(ic, di, de, exclude_item=int(p))
            for ic, di, de, p in zip(ideo_current_list, direction_list, delta_list, pos_items)
        ]
        outside_items = [
            self.neg_sampler.sample_outside(ic, di, de, exclude_item=int(p))
            for ic, di, de, p in zip(ideo_current_list, direction_list, delta_list, pos_items)
        ]

        aligned_idx = torch.tensor(aligned_items, dtype=torch.long)
        outside_idx = torch.tensor(outside_items, dtype=torch.long)

        out["ideo_aligned_item_idx"] = aligned_idx
        out["ideo_aligned_ideo"]     = torch.tensor(
            self.item_ideo_arr[aligned_idx.numpy()], dtype=torch.float32
        )
        out["ideo_outside_item_idx"] = outside_idx
        out["ideo_outside_ideo"]     = torch.tensor(
            self.item_ideo_arr[outside_idx.numpy()], dtype=torch.float32
        )
        return out


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

            # Filter sequence to known items only; replace NaN ideology with 0
            filtered = [
                (uid,
                 float(sc) if np.isfinite(sc) else 0.0,
                 float(st) if np.isfinite(st) else 0.0)
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

    def sample_aligned(
        self,
        ideo_current: float,
        direction: float,
        delta: float,
        exclude_item: int | None = None,
    ) -> int:
        """
        Sample an item within the ideology window:
            [ideo_current, ideo_current + direction*delta]
        i.e. items that move in the correct direction by at most delta.
        Falls back to a random item if the window contains nothing.
        """
        d = direction if direction != 0.0 else 1.0
        lo = ideo_current + min(0.0, d * delta)
        hi = ideo_current + max(0.0, d * delta)

        b_lo = max(0, int((lo + 3) / 0.5))
        b_hi = min(11, int((hi + 3) / 0.5))

        pool: list[int] = []
        for b in range(b_lo, b_hi + 1):
            pool.extend(self._buckets.get(b, []))

        # Exact continuous-window filter
        pool = [
            x for x in pool
            if lo <= self.item_ideo.get(x, 0.0) <= hi
            and x != exclude_item
        ]

        if not pool:
            pool = [x for x in self.all_items if x != exclude_item]
        return random.choice(pool) if pool else random.choice(self.all_items)

    def sample_outside(
        self,
        ideo_current: float,
        direction: float,
        delta: float,
        exclude_item: int | None = None,
    ) -> int:
        """
        Sample an item OUTSIDE the ideology window — either in the wrong
        direction or overshooting beyond delta.
        Falls back to a random item in the degenerate case.
        """
        d = direction if direction != 0.0 else 1.0
        pool = [
            x for x in self.all_items
            if x != exclude_item
            and not (0.0 <= d * (self.item_ideo.get(x, 0.0) - ideo_current) <= delta)
        ]
        if not pool:
            pool = [x for x in self.all_items if x != exclude_item]
        return random.choice(pool) if pool else random.choice(self.all_items)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack a list of sample dicts into batched tensors."""
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}

    # Backward-compatible aliases expected by train/evaluate modules.
    out["hist_item_idx"] = out["history_items"]
    out["hist_ideo"] = out["history_states"]
    out["padding_mask"] = out["history_items"].eq(PAD_IDX)
    out["target_item_idx"] = out["target_item"]
    return out


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
    with open(d / "user2idx-3.pkl",             "rb") as f:
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


def build_dataloaders(
    processed_dir: str | Path,
    max_seq_len: int = 50,
    batch_size: int = 256,
    num_workers: int = 0,
    delta: float = 0.2,
    val_holdout: int = 1,
    test_holdout: int = 1,
    hard_neg_band: float = 0.5,
    num_negatives: int = 1,
    min_item_freq: int = 5,
) -> tuple[DataLoader, DataLoader, DataLoader, ItemCatalog]:
    """Build train/val/test dataloaders with compatibility batch fields."""

    train_ds, val_ds, test_ds, item2idx, idx2item, item_ideo_dict, _, neg_sampler = make_datasets(
        processed_dir=processed_dir,
        max_seq_len=max_seq_len,
        val_holdout=val_holdout,
        test_holdout=test_holdout,
        min_item_freq=min_item_freq,
        delta=delta,
    )
    neg_sampler.band = hard_neg_band

    num_items = (max(item2idx.values()) + 1) if item2idx else 1
    item_ideo_arr = np.zeros(num_items, dtype=np.float32)
    for idx, score in item_ideo_dict.items():
        if 0 <= idx < num_items:
            item_ideo_arr[idx] = float(score) if np.isfinite(score) else 0.0

    # Replace any residual NaN/Inf with 0 (neutral ideology)
    item_ideo_arr = np.nan_to_num(item_ideo_arr, nan=0.0, posinf=0.0, neginf=0.0)

    nan_count = int(np.isnan(item_ideo_arr).sum())
    print(f"item_ideo_arr: {num_items} items, NaN cleaned={nan_count}")

    item_catalog = ItemCatalog(item2idx=item2idx, idx2item=idx2item, item_ideo=item_ideo_arr)

    collate_with_negs = CollateWithNegatives(
        neg_sampler=neg_sampler,
        item_ideo_arr=item_ideo_arr,
        num_negatives=num_negatives,
    )

    common = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_with_negs)
    train_dl = DataLoader(train_ds, shuffle=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    test_dl = DataLoader(test_ds, shuffle=False, **common)

    return train_dl, val_dl, test_dl, item_catalog
