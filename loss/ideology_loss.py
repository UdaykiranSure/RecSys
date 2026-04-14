"""
loss/ideology_loss.py
---------------------
Combined training loss:

    L_total = α_bpr         * L_bpr
            + α_contrastive * L_contrastive

L_bpr:
    Standard Bayesian Personalised Ranking:
    -log σ(score_pos - score_neg)

L_contrastive (ideology-contrastive):
    For each sample the dataloader provides two ideology-typed items:
      i_aligned : an item within the ideology window
                  [ideo_current, ideo_current + direction*delta]
      i_outside : an item OUTSIDE that window (wrong direction or overshoot)

    Loss: -log σ(score_aligned - score_outside)

    This is the *only* loss component that carries ideology and smoothness
    signal. Unlike the old ideology/smoothness losses that operated on
    fixed data scalars (∂L/∂θ ≡ 0), this loss flows gradients through the
    model's dot-product scores:

        ∂L_contrastive/∂θ ≠ 0
          via score_aligned = u_final · aligned_emb   (TweetEncoder + Fusion)
          via score_outside = u_final · outside_emb   (TweetEncoder + Fusion)

    A single α_contrastive weight replaces the old α_ideology + α_smoothness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdeologyLoss(nn.Module):
    def __init__(
        self,
        alpha_bpr:         float = 1.0,
        alpha_contrastive: float = 0.5,
    ):
        super().__init__()
        self.alpha_bpr         = alpha_bpr
        self.alpha_contrastive = alpha_contrastive

    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Standard BPR: -log σ(s_pos - s_neg)."""
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def ideology_contrastive_loss(
        self,
        aligned_scores: torch.Tensor,   # (B,)  scores for in-window items
        outside_scores: torch.Tensor,   # (B,)  scores for out-of-window items
    ) -> torch.Tensor:
        """
        Pairwise contrastive: prefer ideology-aligned items over outside items.
        -log σ(score_aligned - score_outside)
        Handles direction correctness AND smoothness in one term.
        """
        return -F.logsigmoid(aligned_scores - outside_scores).mean()

    def forward(
        self,
        pos_scores:     torch.Tensor,   # (B,)
        neg_scores:     torch.Tensor,   # (B,)
        aligned_scores: torch.Tensor,   # (B,)
        outside_scores: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, dict]:

        l_bpr         = self.bpr_loss(pos_scores, neg_scores)
        l_contrastive = self.ideology_contrastive_loss(aligned_scores, outside_scores)

        total = self.alpha_bpr * l_bpr + self.alpha_contrastive * l_contrastive

        return total, {
            "loss_total":       total.item(),
            "loss_bpr":         l_bpr.item(),
            "loss_contrastive": l_contrastive.item(),
        }
