"""
loss/ideology_loss.py
---------------------
Combined training loss:

    L_total = α_bpr        * L_bpr
            + α_ideology   * L_ideology
            + α_smoothness * L_smoothness

L_bpr:
    -log σ(score_pos - score_neg)

L_ideology (directional progression):
    Penalizes when the recommended item does NOT move in target direction.
    L_ideo = mean( max(0, δ - direction*(ideo_pos - ideo_current))² )

    direction*(ideo_pos - ideo_current) >= δ  → moved correctly  → 0 penalty
    direction*(ideo_pos - ideo_current) in (0,δ) → timid move   → small penalty
    direction*(ideo_pos - ideo_current) <= 0  → wrong direction  → large penalty

L_smoothness (anti-overshoot):
    Penalizes jumps larger than δ.
    L_smooth = mean( max(0, |ideo_pos - ideo_current| - δ)² )

direction and delta are passed per-sample from the Dataset,
making this module ready to accept values from the bandit module later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdeologyLoss(nn.Module):
    def __init__(
        self,
        alpha_bpr:        float = 1.0,
        alpha_ideology:   float = 0.5,
        alpha_smoothness: float = 0.3,
    ):
        super().__init__()
        self.alpha_bpr        = alpha_bpr
        self.alpha_ideology   = alpha_ideology
        self.alpha_smoothness = alpha_smoothness

    def bpr_loss(self, pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def ideology_loss(self, ideo_pos, ideo_current, direction, delta):
        delta_ideo = direction * (ideo_pos - ideo_current)
        return (F.relu(delta - delta_ideo) ** 2).mean()

    def smoothness_loss(self, ideo_pos, ideo_current, delta):
        abs_shift = (ideo_pos - ideo_current).abs()
        return (F.relu(abs_shift - delta) ** 2).mean()

    def forward(
        self,
        pos_scores:   torch.Tensor,   # (B,)
        neg_scores:   torch.Tensor,   # (B,)
        ideo_pos:     torch.Tensor,   # (B,)  ideology of positive item
        ideo_current: torch.Tensor,   # (B,)  user's current ideology state
        direction:    torch.Tensor,   # (B,)  +1 or -1
        delta:        torch.Tensor,   # (B,)  max allowed step
    ) -> tuple[torch.Tensor, dict]:

        l_bpr    = self.bpr_loss(pos_scores, neg_scores)
        l_ideo   = self.ideology_loss(ideo_pos, ideo_current, direction, delta)
        l_smooth = self.smoothness_loss(ideo_pos, ideo_current, delta)

        total = (
            self.alpha_bpr        * l_bpr
            + self.alpha_ideology   * l_ideo
            + self.alpha_smoothness * l_smooth
        )

        return total, {
            "loss_total"      : total.item(),
            "loss_bpr"        : l_bpr.item(),
            "loss_ideology"   : l_ideo.item(),
            "loss_smoothness" : l_smooth.item(),
        }
