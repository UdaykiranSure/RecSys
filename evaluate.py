"""
evaluate.py
-----------
Evaluation for the Ideological Sequential Recommender.

Metrics:
    Standard retrieval:
        Hit@K     : fraction of users where ground truth is in top-K
        NDCG@K    : normalized discounted cumulative gain at K

    Ideology-specific:
        ideo_drift@K  : mean ideology of top-K recommendations minus
                        user's current ideology (positive = moved correctly)
        ideo_in_window: fraction of top-K recs within [current, current+δ]
        direction_acc : fraction of top-K recs moving in correct direction
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import ItemCatalog
from models.recommender import IdeologyRecommender


# ── Metric helpers ────────────────────────────────────────────────────────────

def hit_at_k(ranked_list: list[int], target: int, k: int) -> float:
    return float(target in ranked_list[:k])


def ndcg_at_k(ranked_list: list[int], target: int, k: int) -> float:
    if target not in ranked_list[:k]:
        return 0.0
    rank = ranked_list.index(target)
    return 1.0 / np.log2(rank + 2)   # +2 because rank is 0-indexed


def ideology_drift(
    rec_ideologies:  np.ndarray,    # [K] ideology of top-K recs
    ideo_current:    float,
    direction:       float,
) -> float:
    """
    Mean signed ideology shift of top-K recs relative to user's current state.
    Positive = moved in correct direction.
    """
    mean_rec_ideo = rec_ideologies.mean()
    return float(direction * (mean_rec_ideo - ideo_current))


def fraction_in_window(
    rec_ideologies: np.ndarray,
    ideo_current:   float,
    direction:      float,
    delta:          float,
) -> float:
    """Fraction of top-K recs within the ideology window [current, current+δ]."""
    if direction >= 0:
        in_window = (rec_ideologies >= ideo_current) & \
                    (rec_ideologies <= ideo_current + delta)
    else:
        in_window = (rec_ideologies <= ideo_current) & \
                    (rec_ideologies >= ideo_current + direction * delta)
    return float(in_window.mean())


# ── Main evaluation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model:         IdeologyRecommender,
    dataloader:    DataLoader,
    item_catalog:  ItemCatalog,
    graph_x:       torch.Tensor,
    graph_edge_index: torch.Tensor,
    device:        str,
    k_values:      list[int] = [5, 10, 20],
    split:         str = "val",
) -> dict[str, float]:
    """
    Evaluate model on a val/test dataloader.

    For each sample:
    1. Encode the user from their history + graph embedding
    2. Score all items in the catalog
    3. Apply ideology window filter
    4. Rank and compute metrics against ground truth

    Returns
    -------
    metrics dict: {hit@5, hit@10, ndcg@10, ideo_drift@10, ...}
    """
    model.eval()

    # Pre-encode all catalog items (batch to avoid OOM)
    print(f"  [{split}] Encoding {item_catalog.num_items:,} catalog items...")
    all_item_idx  = torch.arange(item_catalog.num_items, dtype=torch.long).to(device)
    all_item_ideo = torch.tensor(item_catalog.item_ideo, dtype=torch.float32).to(device)

    ITEM_BATCH = 4096
    all_item_embs_list = []
    for start in range(0, item_catalog.num_items, ITEM_BATCH):
        end   = min(start + ITEM_BATCH, item_catalog.num_items)
        embs  = model.tweet_encoder(
            all_item_idx[start:end],
            all_item_ideo[start:end]
        )
        all_item_embs_list.append(embs)
    all_item_embs = torch.cat(all_item_embs_list, dim=0)   # [M, D]

    # Accumulators
    accum = {f"hit@{k}": [] for k in k_values}
    accum.update({f"ndcg@{k}": [] for k in k_values})
    accum["ideo_drift@10"]    = []
    accum["ideo_in_window"]   = []
    accum["direction_acc"]    = []

    for batch in dataloader:
        user_idx        = batch["user_idx"].to(device)            # [B]
        hist_item_idx   = batch["history_items"].to(device)       # [B, L]
        hist_ideo       = batch["history_states"].to(device)      # [B, L]
        target_item_idx = batch["target_item"].to(device)         # [B]
        ideo_current    = batch["ideo_current"].to(device)        # [B]
        direction       = batch["direction"].to(device)           # [B]
        delta           = batch["delta"].to(device)               # [B]

        # Encode users
        u_final = model.encode_user(
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            user_graph_idx=user_idx,
            seq_item_ids=hist_item_idx,
            seq_ideo_scores=hist_ideo,
        )

        # Score all items: [B, M]
        scores = u_final @ all_item_embs.T                          # [B, M]
        B,M = scores.shape
        # Per-sample evaluation
        for i in range(B):
            target_idx   = target_item_idx[i].item()
            ideo_cur     = ideo_current[i].item()
            dir_i        = direction[i].item()
            delta_i      = delta[i].item()
            scores_i     = scores[i].cpu().numpy()                  # [M]

            # ── Ideology window mask ──────────────────────────────────
            ideo_np = item_catalog.item_ideo                         # [M]
            if dir_i >= 0:
                mask = (ideo_np >= ideo_cur) & (ideo_np <= ideo_cur + delta_i)
            else:
                mask = (ideo_np <= ideo_cur) & (ideo_np >= ideo_cur + dir_i * delta_i)

            scores_masked = scores_i.copy()
            scores_masked[~mask] = -np.inf

            # Rank items (descending score)
            ranked = np.argsort(-scores_masked).tolist()

            # ── Standard metrics ──────────────────────────────────────
            for k in k_values:
                accum[f"hit@{k}"].append(hit_at_k(ranked, target_idx, k))
                accum[f"ndcg@{k}"].append(ndcg_at_k(ranked, target_idx, k))

            # ── Ideology metrics ──────────────────────────────────────
            top10_idx   = [r for r in ranked[:10] if r < len(ideo_np)]
            if top10_idx:
                top10_ideo  = ideo_np[top10_idx]
                accum["ideo_drift@10"].append(
                    ideology_drift(top10_ideo, ideo_cur, dir_i)
                )
                accum["ideo_in_window"].append(
                    fraction_in_window(top10_ideo, ideo_cur, dir_i, delta_i)
                )
                accum["direction_acc"].append(
                    float(np.mean(dir_i * (top10_ideo - ideo_cur) > 0))
                )

    # Average all metrics
    metrics = {k: float(np.mean(v)) for k, v in accum.items() if v}
    return metrics
