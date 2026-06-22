# Copyright (c) OpenMMLab. All rights reserved.
"""Corruption augmentation transforms.

Two transforms share one size-safe corruption backend:

* :class:`MultiCorruptionAugment` -- picks the corruption by ``results['aug_id']``
  (used by the structured B x K loader / CADR).
* :class:`RandomCorruptionAugment` -- with probability ``prob`` applies a random
  corruption (random severity); otherwise leaves the image clean. Used for the
  "corruption-augmented" baseline (Baseline B), and as a plain single-image aug.

Both run **before Normalize** (expect HxWx3 uint8) and leave bboxes untouched.

Backends:
* ``builtin`` (default): size-agnostic numpy/cv2 (works on 1024x1024 tiles).
* ``imagecorruptions``: upstream package (its fog breaks above 256px).

IMPORTANT: training and evaluation must use the SAME corruption code/backend so
the degradation distributions agree.
"""
import numpy as np

from ..builder import ROTATED_PIPELINES
from .corruptions_builtin import CORRUPTIONS as _BUILTIN

try:
    from imagecorruptions import corrupt as _ic_corrupt
    HAS_IMAGECORRUPTIONS = True
except ImportError:  # pragma: no cover
    _ic_corrupt = None
    HAS_IMAGECORRUPTIONS = False


def apply_corruption(img, corruption, severity, backend='builtin'):
    """Apply one corruption to an HxWx3 uint8 image. ``corruption=None`` -> clean."""
    if corruption is None:
        return img
    if isinstance(severity, (list, tuple)):
        severity = int(np.random.choice(severity))
    severity = int(severity)
    img_u8 = np.ascontiguousarray(img.astype(np.uint8))

    if backend == 'builtin':
        if corruption not in _BUILTIN:
            raise KeyError(
                f"builtin backend has no '{corruption}'. Available: "
                f'{sorted(_BUILTIN)}.')
        out = _BUILTIN[corruption](img_u8, severity)
    elif backend == 'imagecorruptions':
        if not HAS_IMAGECORRUPTIONS:
            raise ImportError(
                'backend="imagecorruptions" needs `pip install imagecorruptions`.')
        # mmdet loads BGR; imagecorruptions (and the FAIR1M-C generator) expect
        # RGB. Convert in/out so training matches the benchmark exactly.
        rgb = np.ascontiguousarray(img_u8[..., ::-1])
        out = _ic_corrupt(rgb, corruption_name=corruption, severity=severity)
        out = np.ascontiguousarray(out[..., ::-1])
    else:
        raise ValueError(f'unknown backend {backend}')
    return out.astype(img.dtype)


# Default K=3 MVE pool (clean / fog / gaussian_noise).
DEFAULT_AUG_POOL = [
    dict(corruption=None, severity=None),
    dict(corruption='fog', severity=[1, 2]),
    dict(corruption='gaussian_noise', severity=[1, 2]),
]


@ROTATED_PIPELINES.register_module()
class MultiCorruptionAugment(object):
    """Apply the corruption selected by ``results['aug_id']`` (structured loader).

    Args:
        aug_pool (list[dict], optional): one entry per ``aug_id`` with
            ``corruption`` (str/None) and ``severity`` (int/list/None).
        backend (str): ``'builtin'`` (default) or ``'imagecorruptions'``.
        skip_if_no_aug_id (bool): pass through when ``aug_id`` is absent.
    """

    def __init__(self, aug_pool=None, backend='builtin', skip_if_no_aug_id=True):
        assert backend in ('builtin', 'imagecorruptions')
        self.aug_pool = aug_pool if aug_pool is not None else DEFAULT_AUG_POOL
        self.backend = backend
        self.skip_if_no_aug_id = skip_if_no_aug_id

    def __call__(self, results):
        aug_id = results.get('aug_id', None)
        if aug_id is None:
            if self.skip_if_no_aug_id:
                return results
            raise KeyError("MultiCorruptionAugment expected 'aug_id'.")
        spec = self.aug_pool[int(aug_id)]
        results['img'] = apply_corruption(
            results['img'], spec['corruption'], spec['severity'], self.backend)
        results['corruption_name'] = spec['corruption']
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(backend={self.backend}, '
                f'aug_pool={self.aug_pool})')


@ROTATED_PIPELINES.register_module()
class RandomCorruptionAugment(object):
    """With prob ``prob`` apply a random corruption (random severity); else clean.

    Used for the corruption-augmented baseline (Baseline B). Match the
    ``corruptions`` / ``severities`` / ``prob`` to the structured-loader pool so
    the comparison isolates the method, not the augmentation.

    Args:
        corruptions (list[str]): corruption pool to sample from.
        severities (list[int]): severity pool to sample from.
        prob (float): probability of applying a corruption (else clean).
        backend (str): ``'builtin'`` or ``'imagecorruptions'``.
    """

    def __init__(self,
                 corruptions=('gaussian_noise', 'defocus_blur', 'brightness',
                              'fog', 'spatter'),
                 severities=(1, 2, 3),
                 prob=0.5,
                 backend='builtin'):
        assert backend in ('builtin', 'imagecorruptions')
        self.corruptions = list(corruptions)
        self.severities = list(severities)
        self.prob = prob
        self.backend = backend

    def __call__(self, results):
        if np.random.rand() < self.prob:
            name = str(np.random.choice(self.corruptions))
            sev = int(np.random.choice(self.severities))
            results['img'] = apply_corruption(
                results['img'], name, sev, self.backend)
            results['corruption_name'] = name
        else:
            results['corruption_name'] = None
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'corruptions={self.corruptions}, severities={self.severities})')
