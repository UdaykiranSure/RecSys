# Ideological Sequential Recommender — Obamacare Twitter Dataset

A joint Graph + Sequential recommender that nudges users gradually across the
political spectrum, enforcing controlled ideological drift via a constrained
loss function. No ground-truth drift sequences needed.

---

## Architecture

```
Follower/Friend Graph → GraphSAGE → u_graph [B, D]
                                          ↓
RT Sequence → TweetEncoder → SASRec → u_seq [B, D]
                                          ↓
                               FusionMLP (gated concat)
                                          ↓
                                   u_final [B, D]
                                          ↓
                              dot product with tweet_emb
                                          ↓
                               BPR + Direction + Smoothness Loss
```

---

## Loss Function

```
L_total = L_bpr + α * L_direction + β * L_smoothness

L_bpr        = -log σ(score_pos - score_neg)
L_direction  = max(0, δ - direction * Δideo)²
L_smoothness = max(0, |Δideo| - δ)²

where Δideo = ideo(recommended) - ideo_current(user)
      direction = +1 (push right) or -1 (push left), given externally
      δ = fixed budget (default 0.2), later replaced by bandit module
```

Key insight: **no drift ground truth needed**. The model trains on real RT
sequences for engagement, while the loss terms enforce ideological progression.

---

## Codebase Structure

```
obamacare_rec/
├── config.py                   # all hyperparameters
├── run_pipeline.py             # full data preprocessing
├── train.py                    # training loop
├── evaluate.py                 # Hit@K, NDCG@K, ideology drift metrics
├── requirements.txt
│
├── data/
│   ├── parse_tweets.py         # parse USER_TWEETS.txt.gz → RT sequences
│   ├── ideology_scorer.py      # score tweets via Barberá + recency weighting
│   ├── parse_graph.py          # build follower/friend graph
│   └── dataset.py              # PyTorch Dataset + DataLoader factory
│
├── models/
│   ├── tweet_encoder.py        # item embedding (id + ideology)
│   ├── graph_module.py         # GraphSAGE + GraphData loader
│   ├── sasrec.py               # SASRec transformer encoder
│   ├── fusion.py               # gated MLP fusion
│   └── recommender.py          # full model + inference
│
└── loss/
    └── ideology_loss.py        # BPR + direction + smoothness
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run data pipeline
```bash
python run_pipeline.py \
    --tweets   data/raw/USER_TWEETS.txt \
    --barbera  data/raw/USER_POLARITY_BARBERA.txt \
    --follower data/raw/FULL_FOLLOWER_NETWORK.txt \
    --friend   data/raw/FULL_FRIEND_NETWORK.txt \
    --output   data/processed
```

### 3. Train
```bash
python train.py --epochs 30 --delta 0.2 --alpha 0.5 --beta 0.3
```

### 4. Evaluate
Evaluation runs automatically at the end of training.
Best model is saved to `checkpoints/best_model.pt`.

---

## Key Configuration (config.py)

| Parameter | Default | Description |
|---|---|---|
| `loss.delta` | 0.2 | Max ideology step per recommendation |
| `loss.alpha` | 0.5 | Direction loss weight |
| `loss.beta` | 0.3 | Smoothness loss weight |
| `data.max_seq_len` | 50 | Max RT history length |
| `data.recency_lambda` | 0.1 | Decay for ideology state weighting |
| `model.embed_dim` | 64 | Unified embedding dimension |

---

## Important Notes

### user_id is missing from tweet fields
Your `USER_TWEETS.txt.gz` does not contain a `user_id` column.
In `parse_tweets.py`, set `HAS_USER_ID_COLUMN = True` if it is actually
present as the first column, or implement a `tweet_id → user_id` mapping
in the `TODO` block in `parse_tweets.py`.

### Nudge direction
`direction` (+1 / -1) is passed in externally per user. This module is a
**pure executor** — it does not decide the direction or magnitude of drift.
The contextual bandit module (future work) will supply these values.

### Delta is fixed at 0.2
During training, δ=0.2 for all users. When the bandit module is ready,
it replaces this with per-user, per-step delta values. The interface
is already designed to accept delta as a tensor.

---

## Extending to the Bandit Module

The recommender's `forward()` and `recommend()` methods accept `delta` and
`direction` as inputs. To plug in the bandit module:

```python
# Bandit module proposes (direction, delta) per user
direction, delta = bandit.propose(user_id, user_history, current_ideology)

# Recommender executes within that constraint
top_items, scores = recommender.recommend(
    ...,
    direction=direction,
    delta=delta,
)
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `hit@K` | Ground truth RT in top-K recommendations |
| `ndcg@K` | Ranking quality of ground truth in top-K |
| `ideo_drift@10` | Mean ideology shift of top-10 (positive = correct direction) |
| `ideo_in_window` | Fraction of top-10 within ideology window |
| `direction_acc` | Fraction of top-10 moving in correct direction |
