"""Standalone (CPU-only) unit tests for the MVE building blocks.

These cover the pure-torch / pure-numpy pieces that don't need the full
mmrotate + mmdet stack:

* StructuredBatchSampler  -- B x K matrix completeness
* BidirectionalContrastiveLoss -- mask construction + InfoNCE behaviour
* DisentangleHead         -- shapes + L2 normalization

Run (e.g. in the yolov8 conda env which has torch + numpy)::

    python -m pytest tests/test_mve_components.py -v
"""
import importlib.util
import os.path as osp

import numpy as np
import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, osp.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sampler = _load('mmrotate/datasets/samplers/structured_batch_sampler.py',
                 'structured_batch_sampler')
_loss = _load('mmrotate/models/losses/bidirectional_contrastive_loss.py',
              'bidirectional_contrastive_loss')
_dis = _load('mmrotate/models/roi_heads/disentangle_head.py',
             'disentangle_head')

StructuredBatchSampler = _sampler.StructuredBatchSampler
BidirectionalContrastiveLoss = _loss.BidirectionalContrastiveLoss
DisentangleHead = _dis.DisentangleHead


# --------------------------------------------------------------------------- #
# StructuredBatchSampler
# --------------------------------------------------------------------------- #
def test_sampler_matrix_complete():
    num_base, K, B = 10, 3, 2
    s = StructuredBatchSampler(num_base, K, B, shuffle=True, seed=0)
    batches = list(s)
    assert len(batches) == num_base // B
    for batch in batches:
        assert len(batch) == B * K
        bases = [idx // K for idx in batch]
        augs = [idx % K for idx in batch]
        # exactly B distinct base images, each appearing K times
        uniq = set(bases)
        assert len(uniq) == B
        for ub in uniq:
            assert bases.count(ub) == K
        # every aug slot appears exactly B times
        for a in range(K):
            assert augs.count(a) == B


def test_sampler_reshuffles_each_epoch():
    s = StructuredBatchSampler(8, 3, 2, shuffle=True, seed=1)
    e0 = list(s)
    e1 = list(s)
    assert e0 != e1  # auto-advancing epoch changes the order


def test_sampler_distributed_partition():
    num_base, K, B = 12, 3, 2
    seen = []
    for rank in range(2):
        s = StructuredBatchSampler(
            num_base, K, B, shuffle=True, seed=5, num_replicas=2, rank=rank)
        for batch in s:
            seen.extend(idx // K for idx in batch)
    # both ranks together cover all base images, no overlap within an epoch
    assert sorted(set(seen)) == list(range(num_base))


# --------------------------------------------------------------------------- #
# BidirectionalContrastiveLoss
# --------------------------------------------------------------------------- #
def _ids_for(B, K):
    orig = torch.tensor([b for b in range(B) for _ in range(K)])
    aug = torch.tensor([a for _ in range(B) for a in range(K)])
    return orig, aug


def test_mask_construction():
    orig, aug = _ids_for(2, 3)  # 6 samples
    dp, dn, npos, nn_ = BidirectionalContrastiveLoss._build_masks(orig, aug)
    # no self-positives / self-negatives
    assert not torch.diag(dp).any()
    assert not torch.diag(dn).any()
    assert not torch.diag(npos).any()
    assert not torch.diag(nn_).any()
    # disc positives: same image, different aug -> K-1 per row
    assert dp.sum(1).unique().tolist() == [2]
    # disc negatives: different image -> (B-1)*K per row
    assert dn.sum(1).unique().tolist() == [3]
    # nuis positives: same aug, different image -> B-1 per row
    assert npos.sum(1).unique().tolist() == [1]


def test_loss_positive_and_finite():
    torch.manual_seed(0)
    B, K, d = 3, 3, 16
    orig, aug = _ids_for(B, K)
    feats = torch.randn(B * K, d)
    feats = torch.nn.functional.normalize(feats, dim=1)
    crit = BidirectionalContrastiveLoss(temperature=0.07)
    out = crit(feats.clone(), feats.clone(), orig, aug)
    assert torch.isfinite(out['loss_disc_bicon'])
    assert torch.isfinite(out['loss_nuis_bicon'])
    assert out['loss_disc_bicon'] > 0


def test_loss_perfect_disc_is_low():
    """If same-image embeddings are identical and different images are
    orthogonal, the disc loss should be much smaller than for random feats."""
    B, K, d = 3, 3, 8
    orig, aug = _ids_for(B, K)
    # one orthonormal-ish vector per image, shared across its K augs
    base = torch.eye(B, d)
    feats = torch.stack([base[b] for b in range(B) for _ in range(K)])
    feats = torch.nn.functional.normalize(feats, dim=1)
    crit = BidirectionalContrastiveLoss(temperature=0.07)
    good = crit(feats, feats, orig, aug)['loss_disc_bicon']

    torch.manual_seed(1)
    rand = torch.nn.functional.normalize(torch.randn(B * K, d), dim=1)
    bad = crit(rand, rand, orig, aug)['loss_disc_bicon']
    assert good < bad


def test_loss_backward():
    B, K, d = 2, 3, 8
    orig, aug = _ids_for(B, K)
    feats = torch.nn.functional.normalize(
        torch.randn(B * K, d, requires_grad=True), dim=1)
    crit = BidirectionalContrastiveLoss()
    out = crit(feats, feats, orig, aug)
    (out['loss_disc_bicon'] + out['loss_nuis_bicon']).backward()


# --------------------------------------------------------------------------- #
# DisentangleHead
# --------------------------------------------------------------------------- #
def test_disentangle_shapes_and_norm():
    head = DisentangleHead(
        in_channels=1024, hidden_channels=512, disc_channels=128,
        nuis_channels=128)
    head.init_weights()
    x = torch.randn(7, 1024)
    out = head(x)
    assert out['disc'].shape == (7, 128)
    assert out['nuis'].shape == (7, 128)
    assert out['disc_raw'].shape == (7, 128)
    # L2-normalized outputs have unit norm
    assert torch.allclose(
        out['disc'].norm(dim=1), torch.ones(7), atol=1e-5)
    assert torch.allclose(
        out['nuis'].norm(dim=1), torch.ones(7), atol=1e-5)


if __name__ == '__main__':
    import sys
    sys.exit(__import__('pytest').main([__file__, '-v']))
