"""
models/tweet_encoder.py
-----------------------
Encodes a tweet (item) into a dense embedding.

Input features per tweet:
    - item_idx       : integer item id  → learnable embedding
    - ideology_score : float scalar     → normalized and projected
    - engagement     : [retweet_count, mention_count] (optional, zeros if unavailable)

Architecture:
    embedding(item_idx)  → R^d
    MLP([embed || ideo || engagement]) → R^output_dim
"""

import torch
import torch.nn as nn


class TweetEncoder(nn.Module):
    """
    Parameters
    ----------
    num_items       : vocabulary size (including PAD at index 0)
    embed_dim       : learnable item embedding dimension
    ideology_dim    : 1  (scalar ideology score)
    engagement_dim  : 2  (retweet_count, mention_count)
    hidden_dim      : MLP hidden layer width
    output_dim      : final embedding dimension
    dropout         : dropout probability
    """

    def __init__(
        self,
        num_items:      int,
        embed_dim:      int   = 64,
        ideology_dim:   int   = 1,
        engagement_dim: int   = 2,
        hidden_dim:     int   = 128,
        output_dim:     int   = 64,
        dropout:        float = 0.1,
        padding_idx:    int   = 0,
    ):
        super().__init__()
        self.embed_dim     = embed_dim
        self.output_dim    = output_dim

        # Learnable item embedding
        self.item_embedding = nn.Embedding(
            num_items, embed_dim, padding_idx=padding_idx
        )

        # Project ideology scalar to a small vector
        self.ideo_proj = nn.Linear(ideology_dim, 8)

        # Project engagement features
        self.eng_proj = nn.Linear(engagement_dim, 8)

        # Fusion MLP: embed_dim + 8 + 8 → hidden → output_dim
        in_dim = embed_dim + 8 + 8
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        item_ids:    torch.Tensor,          # (...,) long
        ideo_scores: torch.Tensor,          # (...,) float  ideology score
        engagement:  torch.Tensor | None = None,  # (..., 2) float
    ) -> torch.Tensor:                      # (..., output_dim)
        """
        item_ids and ideo_scores must have the same leading shape.
        engagement is optional; zeros are used if not provided.
        """
        shape = item_ids.shape

        # Item embedding
        emb = self.item_embedding(item_ids)             # (..., embed_dim)

        # Ideology projection
        ideo = ideo_scores.unsqueeze(-1)                # (..., 1)
        ideo_feat = self.ideo_proj(ideo)                # (..., 8)

        # Engagement projection
        if engagement is None:
            engagement = torch.zeros(*shape, 2, device=item_ids.device)
        eng_feat = self.eng_proj(engagement)            # (..., 8)

        # Concatenate and project
        cat = torch.cat([emb, ideo_feat, eng_feat], dim=-1)  # (..., in_dim)
        out = self.mlp(cat)                             # (..., output_dim)
        return out

    def encode_sequence(
        self,
        seq_item_ids:    torch.Tensor,      # (B, L) long
        seq_ideo_scores: torch.Tensor,      # (B, L) float
        seq_engagement:  torch.Tensor | None = None,  # (B, L, 2)
    ) -> torch.Tensor:                      # (B, L, output_dim)
        """Convenience wrapper for batched sequences."""
        B, L = seq_item_ids.shape
        if seq_engagement is not None:
            eng = seq_engagement
        else:
            eng = torch.zeros(B, L, 2, device=seq_item_ids.device)
        return self.forward(seq_item_ids, seq_ideo_scores, eng)
