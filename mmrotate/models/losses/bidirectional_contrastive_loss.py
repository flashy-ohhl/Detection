# Copyright (c) OpenMMLab. All rights reserved.
"""Bidirectional contrastive loss over a B x K (image x augmentation) batch.

Given image-level embeddings tagged with ``(original_id, aug_id)``, this loss
imposes two symmetric InfoNCE objectives:

* ``F_disc`` (discriminative): pull together different augmentations of the
  *same* image, push apart *different* images.  => invariant to corruption,
  sensitive to content.
* ``F_nuis`` (nuisance): pull together the *same* augmentation across
  *different* images, push apart different augmentations of the same image.
  => invariant to content, sensitive to corruption.

The InfoNCE uses the supervised-contrastive ("multi-positive") form where the
denominator is the sum over positives *and* negatives (all samples except the
anchor itself), which is the numerically standard and stable choice.
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
    """Multi-positive InfoNCE.

    Args:
        feats (Tensor): L2-normalized embeddings, shape (M, d).
        pos_mask (Tensor[bool]): (M, M), True where j is a positive of anchor
            i. Diagonal must be False.
        neg_mask (Tensor[bool]): (M, M), True where j is a negative of anchor
            i. Diagonal must be False.
        temperature (float): InfoNCE temperature tau.

    Returns:
        Tensor: scalar loss averaged over valid anchors (anchors having at
        least one positive and one negative). Returns 0 (with grad) if none.
    """
    sim = torch.matmul(feats, feats.t()) / temperature  # (M, M)
    # numerical stability: subtract per-row max over the considered set
    consider = pos_mask | neg_mask
    sim_masked = sim.masked_fill(~consider, float('-inf'))
    row_max = sim_masked.max(dim=1, keepdim=True).values.detach()
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    logits = sim - row_max

    exp_logits = torch.exp(logits) * consider.float()
    denom = exp_logits.sum(dim=1)  # (M,)
    log_prob = logits - torch.log(denom + eps).unsqueeze(1)  # (M, M)

    pos_count = pos_mask.float().sum(dim=1)  # (M,)
    neg_count = neg_mask.float().sum(dim=1)
    # mean log-likelihood over the positives of each anchor
    pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1)

    valid = (pos_count > 0) & (neg_count > 0)
    if valid.sum() == 0:
        return feats.sum() * 0.0
    return -pos_log_prob[valid].mean()


@ROTATED_LOSSES.register_module()
class BidirectionalContrastiveLoss(nn.Module):
    """Bidirectional (disc/nuis) contrastive loss.

    Args:
        temperature (float): InfoNCE temperature. Default: 0.07.
        loss_weight_disc (float): Weight for the disc term. Default: 0.5.
        loss_weight_nuis (float): Weight for the nuis term. Default: 0.5.
    """

    def __init__(self,
                 temperature=0.07,
                 loss_weight_disc=0.5,
                 loss_weight_nuis=0.5):
        super(BidirectionalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight_disc = loss_weight_disc
        self.loss_weight_nuis = loss_weight_nuis

    @staticmethod
    def _build_masks(original_ids, aug_ids):
        """Build the four boolean masks. Diagonals are all False."""
        oid = original_ids.view(-1, 1)
        aid = aug_ids.view(-1, 1)
        same_orig = oid == oid.t()
        same_aug = aid == aid.t()

        disc_pos = same_orig & (~same_aug)   # same image, different aug
        disc_neg = ~same_orig                # different image (any aug)
        nuis_pos = same_aug & (~same_orig)   # same aug, different image
        nuis_neg = same_orig & (~same_aug)   # same image, different aug
        return disc_pos, disc_neg, nuis_pos, nuis_neg

    def forward(self, disc_feats, nuis_feats, original_ids, aug_ids):
        """Compute the bidirectional loss.

        Args:
            disc_feats (Tensor): (M, d) L2-normalized disc embeddings.
            nuis_feats (Tensor): (M, d) L2-normalized nuis embeddings.
            original_ids (Tensor): (M,) image id per embedding.
            aug_ids (Tensor): (M,) augmentation id per embedding.

        Returns:
            dict[str, Tensor]: ``loss_disc_bicon`` and ``loss_nuis_bicon``.
        """
        original_ids = original_ids.to(disc_feats.device)
        aug_ids = aug_ids.to(disc_feats.device)
        disc_pos, disc_neg, nuis_pos, nuis_neg = self._build_masks(
            original_ids, aug_ids)

        loss_disc = _info_nce(disc_feats, disc_pos, disc_neg, self.temperature)
        loss_nuis = _info_nce(nuis_feats, nuis_pos, nuis_neg, self.temperature)
        return dict(
            loss_disc_bicon=self.loss_weight_disc * loss_disc,
            loss_nuis_bicon=self.loss_weight_nuis * loss_nuis)
