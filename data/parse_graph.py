"""
parse_graph.py
--------------
Builds PyTorch Geometric Data objects from:
    FULL_FOLLOWER_NETWORK.txt   follower → user  (directed)
    FULL_FRIEND_NETWORK.txt     user → following (directed)

Node features per user:
    [barbera_score, log_degree]   shape (N, 2)

Edge index:
    Combined undirected view of follower + friend edges for GraphSAGE.
    (GraphSAGE aggregates from neighbors regardless of edge direction.)

OUTPUT (written to processed_dir):
    graph_data.pkl      PyG Data object (edge_index, x, user2idx, idx2user)
    user2idx.pkl        {user_id_str: int_index}
"""

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


def load_edges(path: str | Path, delimiter: str = "\t") -> list[tuple[str, str]]:
    """Load directed edges from a two-column file: src <delim> dst."""
    edges = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) < 2:
                skipped += 1
                continue
            edges.append((parts[0].strip(), parts[1].strip()))
    print(f"  Loaded {len(edges):,} edges  (skipped {skipped})")
    return edges


def build_graph(
    follower_path: str | Path,
    friend_path: str | Path,
    ideology_map: dict[str, float],
    output_dir: str | Path,
    delimiter: str = "\t",
):
    """
    Build a combined graph from follower + friend edges.

    Parameters
    ----------
    follower_path : FULL_FOLLOWER_NETWORK.txt  (follower -> user)
    friend_path   : FULL_FRIEND_NETWORK.txt    (user -> following)
    ideology_map  : {user_id: barbera_score}
    output_dir    : where to save graph_data.pkl

    Returns
    -------
    data      : torch_geometric.data.Data
    user2idx  : {user_id: int}
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "torch_geometric not found. Install with:\n"
            "  pip install torch_geometric"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading follower network...")
    follower_edges = load_edges(follower_path, delimiter)
    print("Loading friend network...")
    friend_edges   = load_edges(friend_path, delimiter)

    # ── Collect all user ids ──────────────────────────────────────────────
    all_users: set[str] = set()
    for src, dst in follower_edges + friend_edges:
        all_users.add(src)
        all_users.add(dst)

    # Ensure all users with Barberá scores are included
    all_users.update(ideology_map.keys())

    # Build index mapping
    user2idx: dict[str, int] = {u: i for i, u in enumerate(sorted(all_users))}
    idx2user: dict[int, str] = {i: u for u, i in user2idx.items()}
    N = len(user2idx)
    print(f"\nTotal nodes : {N:,}")

    # ── Build edge list (undirected union) ────────────────────────────────
    # Follower: A->B means A follows B  →  edge (A,B) and (B,A)
    # Friend:   A->B means A follows B  →  same semantics, add both directions
    edge_set: set[tuple[int, int]] = set()
    for src, dst in follower_edges + friend_edges:
        u, v = user2idx[src], user2idx[dst]
        edge_set.add((u, v))
        edge_set.add((v, u))   # make undirected

    edge_list = list(edge_set)
    src_nodes = [e[0] for e in edge_list]
    dst_nodes = [e[1] for e in edge_list]
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    print(f"Total edges : {edge_index.shape[1]:,}  (undirected)")

    # ── Node features ─────────────────────────────────────────────────────
    # Feature 1: Barberá ideology score (0.0 for unknown users)
    barbera_vec = np.zeros(N, dtype=np.float32)
    known = 0
    for uid, score in ideology_map.items():
        if uid in user2idx:
            barbera_vec[user2idx[uid]] = score
            known += 1
    print(f"Nodes with Barberá score : {known:,} / {N:,}  "
          f"({100*known/N:.1f}%)")

    # Feature 2: log(1 + degree) — measures influence/centrality
    degree = np.zeros(N, dtype=np.float32)
    for u, v in edge_list:
        degree[u] += 1
    log_degree = np.log1p(degree)

    # Normalize both features to [0, 1] for stable training
    def minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    node_feats = np.stack([
        barbera_vec,          # raw score kept as-is for interpretability
        minmax(log_degree),   # normalized degree
    ], axis=1)                # shape (N, 2)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # ── Build PyG Data object ─────────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = N

    # Attach metadata as plain attributes (not tracked by PyG)
    data.user2idx = user2idx
    data.idx2user = idx2user

    # ── Save ──────────────────────────────────────────────────────────────
    with open(output_dir / "graph_data.pkl", "wb") as f:
        pickle.dump(data, f)
    with open(output_dir / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)

    print(f"\nSaved → {output_dir}/graph_data.pkl")
    print(f"Saved → {output_dir}/user2idx.pkl")
    print(f"Node feature shape : {x.shape}")
    print(f"Edge index shape   : {edge_index.shape}")

    return data, user2idx


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--follower",     required=True)
    p.add_argument("--friend",       required=True)
    p.add_argument("--ideology_map", required=True,
                   help="ideology_map.pkl from ideology_scorer.py")
    p.add_argument("--output_dir",   default="data/processed")
    p.add_argument("--delimiter",    default="\t")
    args = p.parse_args()

    with open(args.ideology_map, "rb") as f:
        ideology_map = pickle.load(f)

    build_graph(
        args.follower, args.friend, ideology_map,
        args.output_dir, args.delimiter,
    )
