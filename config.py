"""
config.py
---------
Central configuration for all hyperparameters, paths, and settings.
Edit this file before running any pipeline step.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    pwd                   : str = field(default_factory=lambda: str(Path.cwd()))

    # Raw data
    tweets_gz             : str = "data/raw/USER_TWEETS.txt.gz"
    barbera               : str = "data/raw/USER_POLARITY_BARBERA.txt"
    follower_net          : str = "data/raw/FULL_FOLLOWER_NETWORK.txt"
    friend_net            : str = "data/raw/FULL_FRIEND_NETWORK.txt"

    # Processed outputs
    processed_dir         : str = "data/processed"

    # Model checkpoints
    checkpoint_dir        : str = "checkpoints"

    # Evaluation outputs
    eval_dir              : str = "eval"

    def __post_init__(self):
        base = Path(self.pwd).resolve()
        self.pwd = str(base)
        for attr in [
            "tweets_gz",
            "barbera",
            "follower_net",
            "friend_net",
            "processed_dir",
            "checkpoint_dir",
            "eval_dir",
        ]:
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                setattr(self, attr, str(base / p))


# ── Data Pipeline ─────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    # parse_tweets.py
    has_user_id_column        : bool  = False     # True if user_id is col 0
    timestamp_format          : str   = "unix"    # "unix" | "twitter" | "iso"
    min_rt_sequence_len       : int   = 5

    # ideology_scorer.py
    recency_lambda            : float = 0.1
    missing_ideology_fallback : float = None      # None=drop, 0.0=centrist

    # dataset.py
    max_seq_len               : int   = 50
    val_holdout               : int   = 1
    test_holdout              : int   = 1
    min_item_freq             : int   = 5

    # graph
    graph_delimiter           : str   = "\t"
    graph_max_neighbors       : int   = 10
    graph_num_hops            : int   = 2


# ── Tweet Encoder ─────────────────────────────────────────────────────────────

@dataclass
class TweetEncoderConfig:
    text_embed_dim  : int   = 64
    ideology_dim    : int   = 1
    engagement_dim  : int   = 2
    hidden_dim      : int   = 128
    output_dim      : int   = 64
    dropout         : float = 0.1


# ── GraphSAGE ─────────────────────────────────────────────────────────────────

@dataclass
class GraphConfig:
    node_feat_dim   : int   = 2
    hidden_dim      : int   = 128
    output_dim      : int   = 64
    num_layers      : int   = 2
    aggregator      : str   = "mean"
    dropout         : float = 0.1
    num_neighbors   : int   = 10


# ── SASRec ────────────────────────────────────────────────────────────────────

@dataclass
class SASRecConfig:
    hidden_dim      : int   = 64
    num_heads       : int   = 2
    num_layers      : int   = 2
    dropout         : float = 0.1
    max_seq_len     : int   = 50


# ── Fusion ────────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    graph_dim       : int   = 64
    seq_dim         : int   = 64
    hidden_dim      : int   = 128
    output_dim      : int   = 64
    dropout         : float = 0.1


# ── Loss ──────────────────────────────────────────────────────────────────────

@dataclass
class LossConfig:
    alpha_bpr           : float = 1.0
    alpha_ideology      : float = 0.5
    alpha_smoothness    : float = 0.3
    delta               : float = 0.2      # fixed for now; bandit module later
    num_negatives       : int   = 1
    negative_strategy   : str   = "hard"   # "random" | "hard"
    hard_neg_band       : float = 0.5


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    batch_size      : int   = 256
    num_epochs      : int   = 50
    learning_rate   : float = 1e-3
    weight_decay    : float = 1e-4
    grad_clip       : float = 1.0
    early_stopping  : int   = 5
    eval_every      : int   = 1
    seed            : int   = 42
    device          : str   = "auto"       # "auto" | "cpu" | "cuda" | "mps"
    num_workers     : int   = 4
    pin_memory      : bool  = True


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    k_values            : list = field(default_factory=lambda: [5, 10, 20])
    ideology_drift_bins : int  = 12


# ── Master Config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths     : PathConfig         = field(default_factory=PathConfig)
    data      : DataConfig         = field(default_factory=DataConfig)
    tweet_enc : TweetEncoderConfig = field(default_factory=TweetEncoderConfig)
    graph     : GraphConfig        = field(default_factory=GraphConfig)
    sasrec    : SASRecConfig       = field(default_factory=SASRecConfig)
    fusion    : FusionConfig       = field(default_factory=FusionConfig)
    loss      : LossConfig         = field(default_factory=LossConfig)
    train     : TrainConfig        = field(default_factory=TrainConfig)
    eval      : EvalConfig         = field(default_factory=EvalConfig)

    def __post_init__(self):
        # Enforce dimensional consistency
        d = self.tweet_enc.output_dim
        self.graph.output_dim   = d
        self.sasrec.hidden_dim  = d
        self.fusion.graph_dim   = d
        self.fusion.seq_dim     = d
        self.sasrec.max_seq_len = self.data.max_seq_len
        for attr in ["processed_dir", "checkpoint_dir", "eval_dir"]:
            Path(getattr(self.paths, attr)).mkdir(parents=True, exist_ok=True)


cfg = Config()
