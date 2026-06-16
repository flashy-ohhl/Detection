# Copyright (c) OpenMMLab. All rights reserved.
"""PETDet RoI head with disentangle + bidirectional contrastive + GRL.

Extends :class:`OrientedStandardRoIHead` to learn corruption-disentangled
proposal features on top of the B x K (image x augmentation) batch.

Contrast level:
* ``proposal`` (default): every positive proposal is a contrastive sample.
  - F_disc grouped by *object id* = (original_id, matched_gt_index): different
    corruptions of the same physical object are positives. The GT boxes are
    shared across the K augmentations of an image (corruption does not move
    boxes), so the object id is matched across augmentations.
  - F_nuis grouped by *aug id*: same corruption across objects are positives.
  This gives hundreds of contrastive samples per step even with B=2.
* ``image``: mean-pool positive proposals per image (the original MVE).

Disentanglement (optional, ``disentangle_adv.enable``): two GRL adversaries
remove cross-factor leakage --
* predict aug_id from F_disc through a GRL -> F_disc loses corruption info;
* predict class   from F_nuis through a GRL -> F_nuis loses content info.
The adversary accuracy is itself a disentanglement diagnostic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS, build_loss
from .grl import dann_alpha, grad_reverse
from .oriented_standard_roi_head import OrientedStandardRoIHead

_GT_STRIDE = 100000  # assumes < 1e5 gt boxes per image (object-id packing)


@ROTATED_HEADS.register_module()
class PETDetRoIHead(OrientedStandardRoIHead):
    """Oriented RoI head with disentangle + bidirectional contrastive + GRL.

    Args:
        contrastive_loss (dict): config of the contrastive loss module.
        contrast_level (str): ``'proposal'`` or ``'image'``. Default 'proposal'.
        num_aug (int): K, number of augmentation slots (adversary output size).
        max_pos_per_img (int): cap positives per image for the contrastive
            matrix (proposal level), to bound memory. Default 256.
        disentangle_adv (dict, optional): GRL adversary config with keys
            ``enable``, ``hidden_channels``, ``grl_alpha``, ``grl_warmup_iters``,
            ``loss_weight_disc`` (aug-from-disc), ``loss_weight_nuis``
            (class-from-nuis).
    """

    def __init__(self,
                 contrastive_loss=dict(
                     type='BidirectionalContrastiveLoss',
                     temperature=0.07,
                     loss_weight_disc=0.5,
                     loss_weight_nuis=0.5),
                 contrast_level='proposal',
                 num_aug=3,
                 max_pos_per_img=256,
                 disentangle_adv=None,
                 *args,
                 **kwargs):
        super(PETDetRoIHead, self).__init__(*args, **kwargs)
        assert contrast_level in ('proposal', 'image')
        self.contrast_level = contrast_level
        self.num_aug = num_aug
        self.max_pos_per_img = max_pos_per_img
        self.contrastive_loss = build_loss(contrastive_loss)

        adv = disentangle_adv or {}
        self.adv_enable = adv.get('enable', False)
        if self.adv_enable:
            disc_c = self.bbox_head.disentangle.disc_channels
            nuis_c = self.bbox_head.disentangle.nuis_channels
            h = adv.get('hidden_channels', 128)
            self.aug_adversary = nn.Sequential(
                nn.Linear(disc_c, h), nn.ReLU(inplace=True),
                nn.Linear(h, num_aug))
            self.cls_adversary = nn.Sequential(
                nn.Linear(nuis_c, h), nn.ReLU(inplace=True),
                nn.Linear(h, self.bbox_head.num_classes))
            self.grl_alpha = adv.get('grl_alpha', 1.0)
            self.grl_warmup_iters = adv.get('grl_warmup_iters', 2000)
            self.adv_w_disc = adv.get('loss_weight_disc', 0.1)
            self.adv_w_nuis = adv.get('loss_weight_nuis', 0.1)
            self.register_buffer('_grl_step', torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # forward / loss plumbing
    # ------------------------------------------------------------------ #
    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[self.start_level:self.start_level +
              self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        out = self.bbox_head(bbox_feats)
        if isinstance(out, tuple) and len(out) == 4:
            cls_score, bbox_pred, disc, nuis = out
        else:
            cls_score, bbox_pred = out
            disc = nuis = None
        return dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            disc=disc, nuis=nuis)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        if bbox_results['disc'] is not None:
            if self.contrast_level == 'proposal':
                extra = self._disentangle_loss_proposal(
                    bbox_results['disc'], bbox_results['nuis'], rois,
                    sampling_results, bbox_targets, img_metas)
            else:
                extra = self._contrastive_loss_image(
                    bbox_results['disc'], bbox_results['nuis'], rois,
                    bbox_targets, img_metas)
            loss_bbox.update(extra)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    # ------------------------------------------------------------------ #
    # proposal-level: object-anchored contrast + GRL adversaries
    # ------------------------------------------------------------------ #
    def _gather_positives(self, labels, sampling_results, img_metas):
        """Collect positive-proposal indices and their (object, aug, class)."""
        num_classes = self.bbox_head.num_classes
        sel_idx, obj_ids, aug_ids, cls_ids = [], [], [], []
        start = 0
        for i, res in enumerate(sampling_results):
            n_i = res.bboxes.shape[0]
            block_labels = labels[start:start + n_i]
            pos_in_block = torch.nonzero(
                block_labels < num_classes, as_tuple=False).flatten()
            n_pos = pos_in_block.numel()
            if n_pos > 0:
                gt_inds = res.pos_assigned_gt_inds[:n_pos]
                oid = int(img_metas[i]['original_id'])
                aid = int(img_metas[i]['aug_id'])
                global_pos = pos_in_block + start
                if self.max_pos_per_img and n_pos > self.max_pos_per_img:
                    perm = torch.randperm(
                        n_pos, device=labels.device)[:self.max_pos_per_img]
                    global_pos = global_pos[perm]
                    gt_inds = gt_inds[perm]
                sel_idx.append(global_pos)
                obj_ids.append(oid * _GT_STRIDE + gt_inds.to(labels.dtype))
                aug_ids.append(
                    torch.full((global_pos.numel(), ), aid,
                               dtype=labels.dtype, device=labels.device))
                cls_ids.append(labels[global_pos])
            start += n_i
        if len(sel_idx) == 0:
            return None
        return (torch.cat(sel_idx), torch.cat(obj_ids), torch.cat(aug_ids),
                torch.cat(cls_ids))

    def _disentangle_loss_proposal(self, disc, nuis, rois, sampling_results,
                                   bbox_targets, img_metas):
        assert 'original_id' in img_metas[0] and 'aug_id' in img_metas[0], (
            "PETDetRoIHead needs 'original_id' and 'aug_id' in img_metas "
            '(StructuredFAIR1MDataset + Collect meta_keys).')
        labels = bbox_targets[0]
        gathered = self._gather_positives(labels, sampling_results, img_metas)
        if gathered is None or gathered[0].numel() < 2:
            z = disc.sum() * 0.0
            losses = dict(loss_disc_bicon=z, loss_nuis_bicon=z)
            if self.adv_enable:
                losses.update(loss_adv_disc=z, loss_adv_nuis=z)
            return losses

        sel_idx, obj_ids, aug_ids, cls_ids = gathered
        disc_pos = disc[sel_idx]
        nuis_pos = nuis[sel_idx]

        losses = self.contrastive_loss(disc_pos, nuis_pos, obj_ids, aug_ids)

        if self.adv_enable:
            alpha = dann_alpha(
                int(self._grl_step.item()), self.grl_warmup_iters,
                self.grl_alpha)
            self._grl_step += 1
            aug_logits = self.aug_adversary(grad_reverse(disc_pos, alpha))
            cls_logits = self.cls_adversary(grad_reverse(nuis_pos, alpha))
            loss_adv_disc = F.cross_entropy(aug_logits, aug_ids.long())
            loss_adv_nuis = F.cross_entropy(cls_logits, cls_ids.long())
            losses.update(
                loss_adv_disc=self.adv_w_disc * loss_adv_disc,
                loss_adv_nuis=self.adv_w_nuis * loss_adv_nuis)
        return losses

    # ------------------------------------------------------------------ #
    # image-level: mean-pool positive proposals (original MVE)
    # ------------------------------------------------------------------ #
    def _contrastive_loss_image(self, disc, nuis, rois, bbox_targets,
                                img_metas):
        assert 'original_id' in img_metas[0] and 'aug_id' in img_metas[0], (
            "PETDetRoIHead needs 'original_id' and 'aug_id' in img_metas.")
        labels = bbox_targets[0]
        img_inds = rois[:, 0].long()
        num_classes = self.bbox_head.num_classes
        pos_mask = labels < num_classes

        disc_vecs, nuis_vecs, orig_ids, aug_ids = [], [], [], []
        for i in range(len(img_metas)):
            sel = (img_inds == i) & pos_mask
            if sel.sum() == 0:
                continue
            disc_vecs.append(disc[sel].mean(dim=0))
            nuis_vecs.append(nuis[sel].mean(dim=0))
            orig_ids.append(img_metas[i]['original_id'])
            aug_ids.append(img_metas[i]['aug_id'])

        if len(disc_vecs) < 2:
            z = disc.sum() * 0.0
            return dict(loss_disc_bicon=z, loss_nuis_bicon=z)

        disc_vecs = F.normalize(torch.stack(disc_vecs), dim=1)
        nuis_vecs = F.normalize(torch.stack(nuis_vecs), dim=1)
        orig_ids = disc.new_tensor(orig_ids)
        aug_ids = disc.new_tensor(aug_ids)
        return self.contrastive_loss(disc_vecs, nuis_vecs, orig_ids, aug_ids)
