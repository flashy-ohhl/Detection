# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-corruption augmentation transform (module 2 of the MVE plan).

Turns a clean image into one of ``K`` degraded versions, selected by the
``aug_id`` field injected by :class:`StructuredFAIR1MDataset`.  The degradation
is applied in-place on ``results['img']`` and **must run before Normalize**
(it expects an HxWx3 uint8 image).  Bounding boxes are image-level invariant
and are left untouched.

Two backends:

* ``builtin`` (default): size-agnostic numpy re-implementation
  (:mod:`corruptions_builtin`). Works on 1024x1024 FAIR1M tiles.
* ``imagecorruptions``: the upstream Hendrycks package. NOTE: its ``fog`` /
  ``frost`` use a fixed 256x256 plasma map and break on images larger than
  256px.  Use only if your tiles are <=256 or you have patched the package.

IMPORTANT: the corruptions used here for *training* should match whatever
generated your pre-generated *evaluation* corruption sets, so the train and
test degradation distributions agree.

MVE pool (K=3): ``aug_id`` 0 -> clean, 1 -> fog, 2 -> gaussian_noise, with the
severity drawn at random from the configured range each call.
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


# Default K=3 MVE pool. Each entry: corruption name (or None == clean) and the
# severity (int, or list of ints to sample uniformly).
DEFAULT_AUG_POOL = [
    dict(corruption=None, severity=None),                 # 0: clean
    dict(corruption='fog', severity=[1, 2]),              # 1: fog
    dict(corruption='gaussian_noise', severity=[1, 2]),   # 2: gaussian noise
]


@ROTATED_PIPELINES.register_module()
class MultiCorruptionAugment(object):
    """Apply a corruption selected by ``results['aug_id']``.

    Args:
        aug_pool (list[dict], optional): One entry per ``aug_id``, each with
            ``corruption`` (str or None) and ``severity`` (int / list / None).
            Defaults to the K=3 MVE pool (clean / fog / gaussian_noise).
        backend (str): ``'builtin'`` (default) or ``'imagecorruptions'``.
        skip_if_no_aug_id (bool): pass the image through unchanged when
            ``aug_id`` is absent (e.g. plain val/test). Default: True.
    """

    def __init__(self,
                 aug_pool=None,
                 backend='builtin',
                 skip_if_no_aug_id=True):
        assert backend in ('builtin', 'imagecorruptions')
        self.aug_pool = aug_pool if aug_pool is not None else DEFAULT_AUG_POOL
        self.backend = backend
        self.skip_if_no_aug_id = skip_if_no_aug_id

    def _apply(self, img, corruption, severity):
        if corruption is None:
            return img
        if isinstance(severity, (list, tuple)):
            severity = int(np.random.choice(severity))
        severity = int(severity)
        img_u8 = np.ascontiguousarray(img.astype(np.uint8))

        if self.backend == 'builtin':
            if corruption not in _BUILTIN:
                raise KeyError(
                    f"builtin backend has no '{corruption}'. Available: "
                    f'{sorted(_BUILTIN)}. Use backend="imagecorruptions" or '
                    'add it to corruptions_builtin.py.')
            out = _BUILTIN[corruption](img_u8, severity)
        else:
            if not HAS_IMAGECORRUPTIONS:
                raise ImportError(
                    'backend="imagecorruptions" needs the `imagecorruptions` '
                    'package (`pip install imagecorruptions`).')
            out = _ic_corrupt(
                img_u8, corruption_name=corruption, severity=severity)
        return out.astype(img.dtype)

    def __call__(self, results):
        aug_id = results.get('aug_id', None)
        if aug_id is None:
            if self.skip_if_no_aug_id:
                return results
            raise KeyError(
                "MultiCorruptionAugment expected 'aug_id' in results.")
        spec = self.aug_pool[int(aug_id)]
        results['img'] = self._apply(results['img'], spec['corruption'],
                                     spec['severity'])
        results['corruption_name'] = spec['corruption']
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(backend={self.backend}, '
                f'aug_pool={self.aug_pool})')
