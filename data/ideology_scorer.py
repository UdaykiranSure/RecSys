"""
ideology_scorer.py
------------------
Assigns ideology scores to tweets via USER_POLARITY_BARBERA.txt.

A tweet's ideology = Barberá score of the retweeted user.
Produces per-step recency-weighted ideology state for every user.

OUTPUT (written to processed_dir):
    ideology_map.pkl          {user_id: float}
    scored_rt_sequences.pkl   {user_id: [(retweeted_user_id, ideo_score), ...]}
    user_ideology_states.pkl  {user_id: [ideo_current_at_t, ...]}
"""

import pickle
from pathlib import Path

import numpy as np

RECENCY_LAMBDA            = 0.1    # decay rate for weighted mean
MISSING_IDEOLOGY_FALLBACK = None   # None=drop item, 0.0=centrist
MIN_SCORED_LEN            = 5      # drop users with fewer scored RTs


def load_barbera_scores(path: str | Path) -> dict[str, float]:
    """Load user_id <TAB> ideology_score, no header assumed."""
    scores = {}
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                skipped += 1
                continue
            try:
                scores[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                skipped += 1
    vals = list(scores.values())
    print(f"Barberá scores loaded : {len(scores):,} users  "
          f"(skipped {skipped})")
    print(f"Score range           : [{min(vals):.3f}, {max(vals):.3f}]  "
          f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")
    return scores


def recency_weighted_ideology(
    ideo_seq: list[float],
    lam: float = RECENCY_LAMBDA,
) -> list[float]:
    """
    Compute ideo_current(u, t) for each position t.

    At t=0 no prior history exists, so we return the first item's score
    as the cold-start prior.
    For t>0, we weight history[0..t-1] by exp(-λ*(t-1-i)).
    """
    states = []
    for t in range(len(ideo_seq)):
        if t == 0:
            states.append(ideo_seq[0])
        else:
            history = np.array(ideo_seq[:t])
            weights = np.array([np.exp(-lam * (t - 1 - i)) for i in range(t)])
            states.append(float(np.dot(weights, history) / weights.sum()))
    return states


def score_rt_sequences(
    rt_sequences: dict[str, list[str]],
    ideology_map: dict[str, float],
) -> tuple[dict, dict, dict]:

    scored: dict[str, list[tuple[str, float]]] = {}
    states: dict[str, list[float]]             = {}
    total = hit = dropped = 0

    for uid, seq in rt_sequences.items():
        scored_seq = []
        for rt_uid in seq:
            total += 1
            if rt_uid in ideology_map:
                scored_seq.append((rt_uid, ideology_map[rt_uid]))
                hit += 1
            elif MISSING_IDEOLOGY_FALLBACK is not None:
                scored_seq.append((rt_uid, float(MISSING_IDEOLOGY_FALLBACK)))
                hit += 1

        if len(scored_seq) < MIN_SCORED_LEN:
            dropped += 1
            continue

        scored[uid] = scored_seq
        states[uid] = recency_weighted_ideology([s for _, s in scored_seq])

    cov = hit / total if total else 0
    print(f"\nScoring complete:")
    print(f"  Coverage  : {hit:,}/{total:,}  ({cov*100:.1f}%)")
    print(f"  Kept users: {len(scored):,}   dropped: {dropped:,}")
    return scored, states, {"coverage": cov, "kept": len(scored)}


def analyze_distribution(scored: dict) -> None:
    all_s = [s for seq in scored.values() for _, s in seq]
    finite_scores = np.asarray(all_s, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]

    if finite_scores.size == 0:
        print("\nIdeology distribution of RT items:")
        print("  No finite ideology scores available to summarize.")
        return

    bins  = np.linspace(-3, 3, 13)
    counts, edges = np.histogram(finite_scores, bins=bins)
    max_count = int(counts.max()) if counts.size else 0
    print("\nIdeology distribution of RT items:")
    for i, c in enumerate(counts):
        bar_len = int(40 * c / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  [{edges[i]:+.1f},{edges[i+1]:+.1f})  {c:>8,}  {bar}")
    print(f"\n  N={finite_scores.size:,}  mean={np.mean(finite_scores):.3f}  "
          f"std={np.std(finite_scores):.3f}  median={np.median(finite_scores):.3f}")


def run(barbera_path, rt_seq_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ideology_map = load_barbera_scores(barbera_path)

    with open(rt_seq_path, "rb") as f:
        rt_sequences = pickle.load(f)
    print(f"RT sequences loaded   : {len(rt_sequences):,} users")

    scored, states, stats = score_rt_sequences(rt_sequences, ideology_map)
    analyze_distribution(scored)

    with open(output_dir / "ideology_map.pkl", "wb") as f:
        pickle.dump(ideology_map, f)
    with open(output_dir / "scored_rt_sequences.pkl", "wb") as f:
        pickle.dump(scored, f)
    with open(output_dir / "user_ideology_states.pkl", "wb") as f:
        pickle.dump(states, f)

    print(f"\nSaved → {output_dir}/ideology_map.pkl")
    print(f"Saved → {output_dir}/scored_rt_sequences.pkl")
    print(f"Saved → {output_dir}/user_ideology_states.pkl")
    return ideology_map, scored, states


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--barbera",    required=True)
    p.add_argument("--rt_seq",     required=True)
    p.add_argument("--output_dir", default="data/processed")
    args = p.parse_args()
    run(args.barbera, args.rt_seq, args.output_dir)
