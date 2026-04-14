"""
models/sasrec.py
----------------
SASRec: Self-Attentive Sequential Recommendation.
(Kang & McAuley, 2018 — adapted for ideology-aware items)

Input  : sequence of tweet embeddings  (B, L, d)
         attention mask for padding    (B, L)
Output : sequence representation       (B, d)
         (the hidden state at the last real position)

Architecture:
    Positional encoding
    → N × (Causal Multi-Head Self-Attention + FFN + LayerNorm + Dropout)
    → Output at last non-padding position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    """One Transformer block with causal self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff     = FeedForward(d_model, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        x:               torch.Tensor,              # (B, L, d)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True=pad
        causal_mask:      torch.Tensor | None = None,  # (L, L) bool — passed in from SASRec
    ) -> torch.Tensor:

        # Self-attention with residual
        attn_out, _ = self.attn(
            x, x, x,
            attn_mask         = causal_mask,
            key_padding_mask  = key_padding_mask,
        )
        # Padding queries that have ALL keys masked produce softmax([-inf,...]) = NaN.
        # Replace NaN with 0 so the residual x + 0 = x (identity for those positions).
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = self.norm1(x + self.drop(attn_out))

        # FFN with residual
        x = self.norm2(x + self.ff(x))
        return x


class SASRec(nn.Module):
    """
    Parameters
    ----------
    d_model     : embedding / hidden dimension
    num_heads   : attention heads
    num_layers  : number of transformer blocks
    max_seq_len : maximum sequence length (for positional encoding)
    dropout     : dropout probability
    """

    def __init__(
        self,
        d_model:     int   = 64,
        num_heads:   int   = 2,
        num_layers:  int   = 2,
        max_seq_len: int   = 50,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len + 1, d_model)

        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Pre-build causal mask once; avoids re-allocating on every forward call
        causal = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal)   # (max_seq_len, max_seq_len)

        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        item_embs:    torch.Tensor,              # (B, L, d)  from TweetEncoder
        padding_mask: torch.Tensor | None = None,# (B, L) bool True=pad
    ) -> torch.Tensor:                           # (B, d)
        """
        Returns the hidden state at the last non-padding position for each
        sequence in the batch.
        """
        B, L, d = item_embs.shape

        # Positional ids: 1, 2, ..., L
        positions = torch.arange(1, L + 1, device=item_embs.device)  # (L,)
        positions = positions.unsqueeze(0).expand(B, -1)              # (B, L)
        pos_emb   = self.pos_embedding(positions)                      # (B, L, d)

        x = self.drop(item_embs + pos_emb)

        # Slice the pre-built mask to the actual sequence length
        causal = self.causal_mask[:L, :L]   # (L, L) — no allocation

        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask, causal_mask=causal)

        x = self.norm(x)    # (B, L, d)

        # Extract representation at last real (non-padding) position
        if padding_mask is None:
            out = x[:, -1, :]   # (B, d)
        else:
            # Last non-padded position per sample
            # padding_mask: True = pad position
            # seq_len[i] = number of non-padded tokens in sample i
            seq_lens = (~padding_mask).sum(dim=1).clamp(min=1) - 1  # (B,)
            out = x[torch.arange(B, device=x.device), seq_lens]      # (B, d)

        return out    # (B, d)


def make_padding_mask(item_ids: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Build padding mask from item id tensor.
    Returns bool tensor (B, L), True where item_id == pad_idx.
    """
    return item_ids == pad_idx
