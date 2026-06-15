# Copyright (c) OpenMMLab. All rights reserved.
"""Disentangle projection head.

Splits the per-proposal ``F_feat`` produced by the (ARL) bbox head shared FCs
into a discriminative embedding ``F_disc`` and a nuisance embedding
``F_nuis`` via two parallel MLPs.

Each branch returns both an *unnormalized* vector (used by the classifier,
which needs magnitude information) and an *L2-normalized* vector (used by the
contrastive loss, which lives on the unit sphere).  See module 3 of the MVE
plan for the rationale of the two-version output.
"""
import torch.nn as nn
import torch.nn.functional as F


class DisentangleHead(nn.Module):
    """Two parallel projection MLPs that disentangle ``F_feat``.

    Args:
        in_channels (int): Dimension of the input ``F_feat`` (BCFN/shared-FC
            output). Default: 1024.
        hidden_channels (int): Width of the MLP hidden layer. Default: 512.
        disc_channels (int): Dimension of ``F_disc``. Default: 128.
        nuis_channels (int): Dimension of ``F_nuis``. Default: 128.
    """

    def __init__(self,
                 in_channels=1024,
                 hidden_channels=512,
                 disc_channels=128,
                 nuis_channels=128):
        super(DisentangleHead, self).__init__()
        self.in_channels = in_channels
        self.disc_channels = disc_channels
        self.nuis_channels = nuis_channels

        self.mlp_disc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, disc_channels))
        self.mlp_nuis = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, nuis_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward.

        Args:
            x (Tensor): ``F_feat`` of shape (N, in_channels).

        Returns:
            dict[str, Tensor]: with keys
                - ``disc`` / ``nuis``: L2-normalized embeddings (for the
                  contrastive loss).
                - ``disc_raw`` / ``nuis_raw``: unnormalized embeddings (for
                  the classifier head).
        """
        disc_raw = self.mlp_disc(x)
        nuis_raw = self.mlp_nuis(x)
        return dict(
            disc=F.normalize(disc_raw, dim=1),
            nuis=F.normalize(nuis_raw, dim=1),
            disc_raw=disc_raw,
            nuis_raw=nuis_raw)
