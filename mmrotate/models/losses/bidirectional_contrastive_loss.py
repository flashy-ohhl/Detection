# Copyright (c) OpenMMLab. All rights reserved.
"""Bidirectional supervised-contrastive loss for disentangled embeddings.

Two complementary supervised-contrastive (multi-positive InfoNCE) objectives:

* ``F_disc`` (discriminative): grouped by *content identity* (object id at
  proposal level, or image id at image level). Different corruptions of the
  same object are positives -> invariant to corruption, discriminative across
  objects.
* ``F_nuis`` (nuisance): grouped by *corruption identity* (aug id). Different
  objects under the same corruption are positives -> invariant to content,
  sensitive to corruption.

The denominator is the SupCon "out" form (positives + negatives, all samples
except the anchor), which is the numerically standard, stable choice.
"""
import torch
import torch.nn as nn

try:
    from ..builder import ROTATED_LOSSES
except (ImportError, ValueError):
    # Allow importing this module standalone (e.g. CPU-only unit tests) without
    # the full mmrotate/mmdet registry stack.
    class _DummyRegistry:

        @staticmethod
        def register_module(*args, **kwargs):

            def _decorator(cls):
                return cls

            return _decorator

    ROTATED_LOSSES = _DummyRegistry()


def _info_nce(feats, pos_mask, neg_mask, temperature, eps=1e-8):
    """Multi-positive InfoNCE given explicit positive / negative masks.

    Diagonals of both masks must be False. Averages over valid anchors (those
    with >=1 positive and >=1 negative). Returns 0 (with grad) if none.
    """
    sim = torch.matmul(feats, feats.t()) / temperature
    consider = pos_mask | neg_mask
    sim_masked = sim.masked_fill(~consider, float('-inf'))
    row_max = sim_masked.max(dim=1, keepdim=True).values.detach()
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    logits = sim - row_max

    exp_logits = torch.exp(logits) * consider.float()
    denom = exp_logits.sum(dim=1)
    log_prob = logits - torch.log(denom + eps).unsqueeze(1)

    pos_count = pos_mask.float().sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1)
    pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1)

    valid = (pos_count > 0) & (neg_count > 0)
    if valid.sum() == 0:
        return feats.sum() * 0.0
    return -pos_log_prob[valid].mean()


def supervised_contrastive(feats, labels, temperature):
    """Supervised contrastive loss: same-label = positive, diff-label = negative."""
    labels = labels.view(-1, 1)
    same = labels == labels.t()
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=feats.device)
    pos_mask = same & (~eye)
    neg_mask = ~same  # different label (self is same-label, so excluded)
    return _info_nce(feats, pos_mask, neg_mask, temperature)


@ROTATED_LOSSES.register_module()
class BidirectionalContrastiveLoss(nn.Module):
    """Bidirectional (disc / nuis) supervised contrastive loss.

    Args:
        temperature (float): InfoNCE temperature. Default: 0.07.
        loss_weight_disc (float): weight of the disc term. Default: 0.5.
        loss_weight_nuis (float): weight of the nuis term. Default: 0.5.
    """

    def __init__(self,
                 temperature=0.07,
                 loss_weight_disc=0.5,
                 loss_weight_nuis=0.5):
        super(BidirectionalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight_disc = loss_weight_disc
        self.loss_weight_nuis = loss_weight_nuis

    def forward(self, disc_feats, nuis_feats, disc_labels, nuis_labels):
        """Compute the bidirectional loss.

        Args:
            disc_feats (Tensor): (M, d) L2-normalized disc embeddings.
            nuis_feats (Tensor): (M, d) L2-normalized nuis embeddings.
            disc_labels (Tensor): (M,) content identity (object/image id).
            nuis_labels (Tensor): (M,) corruption identity (aug id).
        """
        disc_labels = disc_labels.to(disc_feats.device)
        nuis_labels = nuis_labels.to(nuis_feats.device)
        loss_disc = supervised_contrastive(disc_feats, disc_labels,
                                           self.temperature)
        loss_nuis = supervised_contrastive(nuis_feats, nuis_labels,
                                           self.temperature)
        return dict(
            loss_disc_bicon=self.loss_weight_disc * loss_disc,
            loss_nuis_bicon=self.loss_weight_nuis * loss_nuis)
