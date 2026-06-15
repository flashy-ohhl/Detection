# Copyright (c) OpenMMLab. All rights reserved.
"""Structured B x K batch sampler.

Organizes each batch as ``B`` base images x ``K`` augmentation slots so that the
"image x augmentation" matrix is complete (every base image appears exactly K
times, every aug slot appears exactly B times).  This is what the bidirectional
contrastive loss needs.

It is a *batch* sampler: ``__iter__`` yields lists of ``B * K`` flat indices.
The flat index encodes ``(base_index, aug_id)`` as ``base_index * K + aug_id``;
``StructuredKAugDataset`` decodes it back.

Distributed training: pass ``num_replicas`` / ``rank`` to shard *base images*
across ranks (each rank still produces complete B x K batches).
"""
import math

import numpy as np
from torch.utils.data import Sampler


class StructuredBatchSampler(Sampler):
    """Yield lists of ``B * K`` indices forming a complete B x K matrix.

    Args:
        num_base (int): Number of base (clean) images in the dataset.
        num_aug (int): K, augmentation versions per image.
        batch_base (int): B, number of base images per batch.
        shuffle (bool): Shuffle base images each epoch. Default: True.
        drop_last (bool): Drop the last incomplete group of B. Default: True.
        seed (int): Base random seed. Default: 0.
        num_replicas (int, optional): Number of distributed processes.
        rank (int, optional): Rank of the current process.
    """

    def __init__(self,
                 num_base,
                 num_aug,
                 batch_base,
                 shuffle=True,
                 drop_last=True,
                 seed=0,
                 num_replicas=None,
                 rank=None):
        self.num_base = num_base
        self.num_aug = num_aug
        self.batch_base = batch_base
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        assert 0 <= self.rank < self.num_replicas

        # number of base images handled by each rank
        self.base_per_rank = int(
            math.ceil(self.num_base / self.num_replicas))
        # number of complete B-groups per rank
        if self.drop_last:
            self.num_groups = self.base_per_rank // self.batch_base
        else:
            self.num_groups = int(
                math.ceil(self.base_per_rank / self.batch_base))

    def __iter__(self):
        if self.shuffle:
            g = np.random.default_rng(self.seed + self.epoch)
            base_order = g.permutation(self.num_base)
        else:
            base_order = np.arange(self.num_base)

        # pad so every rank gets the same number of base images
        total = self.base_per_rank * self.num_replicas
        if total > self.num_base:
            pad = base_order[:total - self.num_base]
            base_order = np.concatenate([base_order, pad])
        # shard base images across ranks (strided)
        rank_bases = base_order[self.rank:total:self.num_replicas]

        for grp in range(self.num_groups):
            grp_bases = rank_bases[grp * self.batch_base:
                                   (grp + 1) * self.batch_base]
            if len(grp_bases) == 0:
                continue
            if self.drop_last and len(grp_bases) < self.batch_base:
                continue
            batch = []
            for base in grp_bases:
                for aug in range(self.num_aug):
                    batch.append(int(base) * self.num_aug + aug)
            yield batch

        # auto-advance so the next epoch reshuffles without needing an explicit
        # set_epoch hook (mmcv's non-distributed runner does not call one).
        self.epoch += 1

    def __len__(self):
        return self.num_groups

    def set_epoch(self, epoch):
        self.epoch = epoch
