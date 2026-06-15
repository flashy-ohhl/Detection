# Copyright (c) OpenMMLab. All rights reserved.
"""ARL bbox head with a disentangle projection (module 5 of the MVE plan).

Inserts a :class:`DisentangleHead` after the shared FCs (``F_feat``):

    F_feat(1024) --> MLP_disc --> F_disc(128) --> Linear(128->1024) --> fc_cls
                 --> MLP_nuis --> F_nuis(128)
                 --> F_feat(1024) ----------------------------------> fc_reg

Classification is driven by the (unnormalized) ``F_disc`` projected back to the
shared dimension; regression keeps using ``F_feat``.  The L2-normalized
``F_disc`` / ``F_nuis`` embeddings are returned for the contrastive loss.

Only the Shared2FC layout (no extra cls/reg convs or fcs) is supported.
"""
import torch.nn as nn

from ...builder import ROTATED_HEADS
from ..disentangle_head import DisentangleHead
from .convfc_rbbox_arl_head import RotatedConvFCBBoxARLHead


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxDisentangleARLHead(RotatedConvFCBBoxARLHead):
    """Shared2FC ARL head with disentangle projection.

    Args:
        disentangle (dict): kwargs for :class:`DisentangleHead`
            (``hidden_channels``, ``disc_channels``, ``nuis_channels``).
        fc_out_channels (int): Shared FC width (``F_feat`` dim). Default: 1024.
    """

    def __init__(self,
                 disentangle=dict(
                     hidden_channels=512,
                     disc_channels=128,
                     nuis_channels=128),
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super(RotatedShared2FCBBoxDisentangleARLHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        # this head assumes the plain Shared2FC routing
        assert self.num_cls_fcs == 0 and self.num_reg_fcs == 0
        assert self.num_cls_convs == 0 and self.num_reg_convs == 0

        self.disentangle = DisentangleHead(
            in_channels=self.shared_out_channels, **disentangle)
        # map F_disc back to the classifier input dimension
        self.cls_proj = nn.Linear(self.disentangle.disc_channels,
                                  self.cls_last_dim)
        self._init_disentangle_weights()

    def _init_disentangle_weights(self):
        self.disentangle.init_weights()
        nn.init.xavier_uniform_(self.cls_proj.weight)
        nn.init.constant_(self.cls_proj.bias, 0)

    def forward(self, x):
        """Forward.

        Returns:
            tuple: ``(cls_score, bbox_pred, disc, nuis)`` where ``disc`` and
            ``nuis`` are the L2-normalized embeddings (N, d_proj) for the
            contrastive loss.
        """
        # shared FCs -> F_feat
        if self.num_shared_fcs > 0:
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        feat = x  # F_feat (N, fc_out_channels)

        dis = self.disentangle(feat)

        # classification: F_disc -> proj -> fc_cls
        x_cls = self.relu(self.cls_proj(dis['disc_raw']))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # regression: F_feat -> fc_reg
        bbox_pred = self.fc_reg(feat) if self.with_reg else None

        return cls_score, bbox_pred, dis['disc'], dis['nuis']
