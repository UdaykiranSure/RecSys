"""
models/recommender.py
---------------------
Full end-to-end recommender: GraphSAGE + SASRec + Fusion.

Forward pass produces u_final ∈ R^d for each user in the batch.
Scoring is done externally (dot product with tweet embeddings) to
keep the model flexible for both training (BPR) and inference (ranking).

                     ┌─────────────────┐
  Social Graph ───► │  GraphSAGE      │ ──► u_graph (B,d)
                     └─────────────────┘          │
                                                  ▼
                     ┌─────────────────┐     ┌──────────┐
  RT Sequence ─────► │  TweetEncoder   │     │  Fusion  │ ──► u_final (B,d)
                ─────► │  SASRec        │ ───► │  MLP     │
                     └─────────────────┘     └──────────┘
"""

import torch
import torch.nn as nn

from models.tweet_encoder import TweetEncoder
from models.graph_module  import GraphSAGEEncoder
from models.sasrec        import SASRec, make_padding_mask
from models.fusion        import FusionModule


class IdeologyRecommender(nn.Module):
    """
    Parameters
    ----------
    num_items       : item vocabulary size
    num_graph_nodes : number of nodes in the social graph
    embed_dim       : tweet embedding dimension (d)
    graph_node_feat : node feature dim (2: barbera + log_degree)
    hidden_dim      : hidden dim for all MLPs
    num_sage_layers : GraphSAGE depth
    num_sasrec_layers : SASRec transformer blocks
    num_heads       : attention heads in SASRec
    max_seq_len     : maximum RT sequence length
    dropout         : dropout probability
    """

    def __init__(
        self,
        num_items         : int,
        num_graph_nodes   : int,
        embed_dim         : int   = 64,
        graph_node_feat   : int   = 2,
        hidden_dim        : int   = 128,
        num_sage_layers   : int   = 2,
        num_sasrec_layers : int   = 2,
        num_heads         : int   = 2,
        max_seq_len       : int   = 50,
        dropout           : float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Tweet encoder (shared between sequence and candidate) ─────────
        self.tweet_encoder = TweetEncoder(
            num_items    = num_items,
            embed_dim    = embed_dim,
            hidden_dim   = hidden_dim,
            output_dim   = embed_dim,
            dropout      = dropout,
        )

        # ── Social graph encoder ──────────────────────────────────────────
        self.graph_encoder = GraphSAGEEncoder(
            node_feat_dim = graph_node_feat,
            hidden_dim    = hidden_dim,
            output_dim    = embed_dim,
            num_layers    = num_sage_layers,
            dropout       = dropout,
        )

        # ── Sequential encoder ────────────────────────────────────────────
        self.sasrec = SASRec(
            d_model     = embed_dim,
            num_heads   = num_heads,
            num_layers  = num_sasrec_layers,
            max_seq_len = max_seq_len,
            dropout     = dropout,
        )

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = FusionModule(
            graph_dim  = embed_dim,
            seq_dim    = embed_dim,
            hidden_dim = hidden_dim,
            output_dim = embed_dim,
            dropout    = dropout,
        )

    def encode_user(
        self,
        # Graph inputs
        graph_x:           torch.Tensor,   # (N, node_feat_dim) full graph features
        graph_edge_index:  torch.Tensor,   # (2, E) full graph edge index
        user_graph_idx:    torch.Tensor,   # (B,) index into graph nodes
        # Sequence inputs
        seq_item_ids:      torch.Tensor,   # (B, L) padded item ids
        seq_ideo_scores:   torch.Tensor,   # (B, L) ideology scores
    ) -> torch.Tensor:                     # (B, embed_dim)
        """
        Encode a batch of users into dense representations.
        """
        # 1. Graph embedding
        all_graph_embs = self.graph_encoder(graph_x, graph_edge_index)  # (N, d)
        u_graph        = all_graph_embs[user_graph_idx]                 # (B, d)

        # 2. Sequence embedding
        padding_mask  = make_padding_mask(seq_item_ids)                 # (B, L)
        seq_embs      = self.tweet_encoder.encode_sequence(
            seq_item_ids, seq_ideo_scores
        )                                                               # (B, L, d)
        u_seq         = self.sasrec(seq_embs, padding_mask)            # (B, d)

        # 3. Fuse
        u_final = self.fusion(u_graph, u_seq)                          # (B, d)
        return u_final

    def encode_items(
        self,
        item_ids:    torch.Tensor,   # (B,) or (B, K)
        ideo_scores: torch.Tensor,   # same shape as item_ids
    ) -> torch.Tensor:               # same shape + embed_dim
        """Encode candidate tweet items."""
        return self.tweet_encoder(item_ids, ideo_scores)

    def score(
        self,
        u_final:    torch.Tensor,    # (B, d)
        item_embs:  torch.Tensor,    # (B, d) or (B, K, d)
    ) -> torch.Tensor:               # (B,) or (B, K)
        """
        Dot-product score between user and item embeddings.
        Supports single item (B, d) or K candidates (B, K, d).
        """
        if item_embs.dim() == 2:
            # (B, d) · (B, d) → (B,)
            return (u_final * item_embs).sum(dim=-1)
        else:
            # (B, 1, d) · (B, K, d)^T → (B, K)
            return torch.bmm(
                u_final.unsqueeze(1),       # (B, 1, d)
                item_embs.transpose(1, 2),  # (B, d, K)
            ).squeeze(1)                    # (B, K)

    def forward(
        self,
        # Graph
        graph_x:          torch.Tensor,
        graph_edge_index: torch.Tensor,
        user_graph_idx:   torch.Tensor,
        # Sequence
        seq_item_ids:     torch.Tensor,
        seq_ideo_scores:  torch.Tensor,
        # Candidate items
        pos_item_ids:     torch.Tensor,    # (B,)
        pos_ideo_scores:  torch.Tensor,    # (B,)
        neg_item_ids:     torch.Tensor,    # (B,)
        neg_ideo_scores:  torch.Tensor,    # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (u_final, pos_scores, neg_scores) for loss computation.
        """
        u_final    = self.encode_user(
            graph_x, graph_edge_index, user_graph_idx,
            seq_item_ids, seq_ideo_scores,
        )
        pos_embs   = self.encode_items(pos_item_ids, pos_ideo_scores)  # (B, d)
        neg_embs   = self.encode_items(neg_item_ids, neg_ideo_scores)  # (B, d)

        pos_scores = self.score(u_final, pos_embs)   # (B,)
        neg_scores = self.score(u_final, neg_embs)   # (B,)

        return u_final, pos_scores, neg_scores
