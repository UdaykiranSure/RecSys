"""
models/graph_module.py
----------------------
GraphSAGE encoder over the follower/friend social graph.

Input  : node feature matrix x  (N, node_feat_dim)
         edge_index              (2, E)
Output : node embeddings         (N, output_dim)

At inference, we look up user_idx to get their graph embedding u_graph.

Architecture: 2-layer GraphSAGE with mean aggregation + skip connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    """
    Single GraphSAGE layer.

    h_v' = W_self * h_v  +  W_neigh * MEAN(h_neighbors)
    followed by LayerNorm + GELU.

    This is a from-scratch implementation so torch_geometric is
    only required for the Data object, not for message passing.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_self  = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim,  out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)  [src, dst]
    ) -> torch.Tensor:              # (N, out_dim)

        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbor features for each node
        # neigh_agg[v] = MEAN of x[u] for all u->v edges
        neigh_sum   = torch.zeros(N, x.size(1), device=x.device)
        neigh_count = torch.zeros(N, 1,          device=x.device)

        neigh_sum.index_add_(0, dst, x[src])
        neigh_count.index_add_(
            0, dst,
            torch.ones(src.size(0), 1, device=x.device)
        )
        neigh_count = neigh_count.clamp(min=1)   # avoid div by zero
        neigh_agg   = neigh_sum / neigh_count     # (N, in_dim)

        out = self.W_self(x) + self.W_neigh(neigh_agg)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class GraphSAGEEncoder(nn.Module):
    """
    2-layer GraphSAGE encoder.

    Parameters
    ----------
    node_feat_dim : input feature dimension (2: barbera + log_degree)
    hidden_dim    : intermediate layer width
    output_dim    : final user embedding dimension
    num_layers    : number of SAGE convolution layers
    dropout       : dropout probability
    """

    def __init__(
        self,
        node_feat_dim : int   = 2,
        hidden_dim    : int   = 128,
        output_dim    : int   = 64,
        num_layers    : int   = 2,
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers

        dims = [node_feat_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            SAGEConv(dims[i], dims[i+1], dropout)
            for i in range(num_layers)
        ])

        # Project node features to hidden_dim before first layer
        self.input_proj = nn.Linear(node_feat_dim, node_feat_dim)

    def forward(
        self,
        x:          torch.Tensor,    # (N, node_feat_dim)
        edge_index: torch.Tensor,    # (2, E)
    ) -> torch.Tensor:               # (N, output_dim)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h                     # (N, output_dim)

    def get_user_embeddings(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        user_indices: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:               # (B, output_dim)
        """
        Full graph forward pass, then index by user_indices.
        For large graphs use mini-batch neighbor sampling instead.
        """
        all_embs = self.forward(x, edge_index)   # (N, output_dim)
        return all_embs[user_indices]             # (B, output_dim)
