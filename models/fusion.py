"""
models/fusion.py
----------------
Fuses graph embedding (u_graph) and sequential embedding (u_seq)
into a single user representation u_final.

    u_final = MLP( LayerNorm([u_graph || u_seq]) )

score(u, tweet) = u_final · tweet_emb
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(
        self,
        graph_dim  : int   = 64,
        seq_dim    : int   = 64,
        hidden_dim : int   = 128,
        output_dim : int   = 64,
        dropout    : float = 0.1,
    ):
        super().__init__()
        in_dim = graph_dim + seq_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u_graph: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        # u_graph: (B, graph_dim)   u_seq: (B, seq_dim)
        cat    = torch.cat([u_graph, u_seq], dim=-1)
        return self.mlp(self.norm(cat))              # (B, output_dim)
