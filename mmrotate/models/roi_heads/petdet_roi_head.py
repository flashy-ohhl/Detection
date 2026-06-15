# Copyright (c) OpenMMLab. All rights reserved.
"""PETDet RoI head with image-level bidirectional contrastive loss.

Extends :class:`OrientedStandardRoIHead` to:

1. unpack the 4-tuple ``(cls_score, bbox_pred, F_disc, F_nuis)`` produced by
   :class:`RotatedShared2FCBBoxDisentangleARLHead`;
2. for each image in the B x K batch, mean-pool the *positive* proposals'
   ``F_disc`` / ``F_nuis`` into a single image-level vector (MVE = image-level
   contrast, decision 1 of module 4);
3. compute the bidirectional contrastive loss over the (<= B*K) image vectors
   using the ``(original_id, aug_id)`` tags carried in ``img_metas``.

Regression and the ARL classification loss are unchanged from the base head.
At inference the contrastive branch is skipped automatically (no loss is
computed in the test path).
"""
import torch
import torch.nn.functional as F

from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS, build_loss
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class PETDetRoIHead(OrientedStandardRoIHead):
    """Oriented RoI head with disentangle + bidirectional contrastive loss.

    Args:
        contrastive_loss (dict): config of the contrastive loss module.
    """

    def __init__(self,
                 contrastive_loss=dict(
                     type='BidirectionalContrastiveLoss',
                     temperature=0.07,
                     loss_weight_disc=0.5,
                     loss_weight_nuis=0.5),
                 *args,
                 **kwargs):
        super(PETDetRoIHead, self).__init__(*args, **kwargs)
        self.contrastive_loss = build_loss(contrastive_loss)

    def _bbox_forward(self, x, rois):
        """Box head forward. Also returns disc/nuis embeddings when available."""
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
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            disc=disc,
            nuis=nuis)

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
            con_losses = self._contrastive_loss_train(
                bbox_results['disc'], bbox_results['nuis'], rois, bbox_targets,
                img_metas)
            loss_bbox.update(con_losses)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _contrastive_loss_train(self, disc, nuis, rois, bbox_targets,
                                img_metas):
        """Image-level pooling of positive proposals + bidirectional loss."""
        labels = bbox_targets[0]
        img_inds = rois[:, 0].long()
        num_classes = self.bbox_head.num_classes
        pos_mask = labels < num_classes

        assert 'original_id' in img_metas[0] and 'aug_id' in img_metas[0], (
            "PETDetRoIHead needs 'original_id' and 'aug_id' in img_metas. Add "
            "them to the Collect meta_keys and use StructuredFAIR1MDataset + "
            'StructuredBatchSampler.')

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
            zero = disc.sum() * 0.0
            return dict(loss_disc_bicon=zero, loss_nuis_bicon=zero)

        disc_vecs = F.normalize(torch.stack(disc_vecs), dim=1)
        nuis_vecs = F.normalize(torch.stack(nuis_vecs), dim=1)
        orig_ids = disc.new_tensor(orig_ids)
        aug_ids = disc.new_tensor(aug_ids)
        return self.contrastive_loss(disc_vecs, nuis_vecs, orig_ids, aug_ids)
