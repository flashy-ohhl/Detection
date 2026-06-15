# Copyright (c) OpenMMLab. All rights reserved.
"""Dataset that exposes the B x K structure to the data loader.

It wraps FAIR1M by presenting ``num_base * K`` logical samples: logical index
``idx`` maps to base image ``idx // K`` with augmentation slot ``idx % K``.
The augmentation slot is written into ``results['aug_id']`` (consumed by
:class:`MultiCorruptionAugment`) and the base image id into
``results['original_id']`` (consumed by the contrastive loss via ``img_metas``).

Pair this dataset with :class:`StructuredBatchSampler` and remember to add
``'original_id'`` and ``'aug_id'`` to the ``meta_keys`` of the ``Collect``
transform so they reach ``img_metas``.
"""
import numpy as np

from .builder import ROTATED_DATASETS
from .fair1m import FAIR1MDataset


@ROTATED_DATASETS.register_module()
class StructuredFAIR1MDataset(FAIR1MDataset):
    """FAIR1M dataset with a B x K (image x augmentation) logical layout.

    Args:
        num_aug (int): K, number of augmentation slots per base image.
        All other args are forwarded to :class:`FAIR1MDataset`.
    """

    def __init__(self, *args, num_aug=3, **kwargs):
        self.num_aug = num_aug
        super(StructuredFAIR1MDataset, self).__init__(*args, **kwargs)

    @property
    def num_base(self):
        return len(self.data_infos)

    def __len__(self):
        return len(self.data_infos) * self.num_aug

    def _set_group_flag(self):
        """All images are treated as a single aspect-ratio group."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_train_img(self, idx):
        """Decode (base, aug) from the flat index and run the pipeline."""
        base = idx // self.num_aug
        aug = idx % self.num_aug
        img_info = self.data_infos[base]
        ann_info = self.get_ann_info(base)
        results = dict(img_info=img_info, ann_info=ann_info)
        results['original_id'] = int(base)
        results['aug_id'] = int(aug)
        if self.proposals is not None:
            results['proposals'] = self.proposals[base]
        self.pre_pipeline(results)
        return self.pipeline(results)
