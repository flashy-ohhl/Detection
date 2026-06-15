# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS, PIPELINES
from mmdet.datasets.builder import _concat_dataset, worker_init_fn
from torch.utils.data import DataLoader

from .samplers import StructuredBatchSampler

ROTATED_DATASETS = DATASETS
ROTATED_PIPELINES = PIPELINES

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset,
                                                 MultiImageMixDataset,
                                                 RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, ROTATED_DATASETS, default_args)

    return dataset


def build_structured_dataloader(dataset,
                                sampler_cfg,
                                workers_per_gpu,
                                dist=False,
                                seed=None,
                                pin_memory=False,
                                persistent_workers=False,
                                **kwargs):
    """Build a DataLoader that yields complete B x K batches.

    Uses :class:`StructuredBatchSampler` as the ``batch_sampler`` so each batch
    contains ``B`` base images x ``K`` augmentation slots. The dataset must
    expose ``num_base`` (e.g. :class:`StructuredFAIR1MDataset`).

    Args:
        dataset (Dataset): a B x K structured dataset (has ``num_base``).
        sampler_cfg (dict): ``num_aug`` (K), ``batch_base`` (B), and optional
            ``shuffle`` / ``drop_last``.
        workers_per_gpu (int): dataloader workers per process.
        dist (bool): distributed training; shards base images across ranks.
        seed (int, optional): random seed.
    """
    rank, world_size = get_dist_info()
    num_replicas = world_size if dist else 1
    rk = rank if dist else 0

    num_aug = sampler_cfg['num_aug']
    batch_base = sampler_cfg['batch_base']

    batch_sampler = StructuredBatchSampler(
        num_base=dataset.num_base,
        num_aug=num_aug,
        batch_base=batch_base,
        shuffle=sampler_cfg.get('shuffle', True),
        drop_last=sampler_cfg.get('drop_last', True),
        seed=seed if seed is not None else 0,
        num_replicas=num_replicas,
        rank=rk)

    samples_per_gpu = batch_base * num_aug
    init_fn = partial(
        worker_init_fn, num_workers=workers_per_gpu, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers_per_gpu,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=persistent_workers,
        **kwargs)
    return data_loader
