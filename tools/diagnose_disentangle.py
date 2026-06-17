# Copyright (c) OpenMMLab. All rights reserved.
"""Disentanglement diagnostic via frozen-feature linear probes.

Resolves the "is F_disc really corruption-free, or did the GRL adversary just
collapse?" ambiguity that the in-training adversary loss cannot answer.

It freezes a trained model, extracts per-positive-proposal F_disc / F_nuis over
a B x K structured loader (so every proposal has a class label and an aug id),
then trains four independent linear probes on the frozen embeddings:

                       predict CLASS            predict AUG
    from F_disc   high  (content kept) v    low  (corruption removed) v
    from F_nuis   low   (content removed) v  high (corruption kept)  v

The diagonal (disc->class, nuis->aug) should be HIGH; the off-diagonal
(disc->aug, nuis->class) should be LOW (near chance). Low off-diagonal == good
disentanglement; high off-diagonal == leakage (and a collapsed adversary).

Example:
    python tools/diagnose_disentangle.py \
        configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py \
        work_dirs/petdet_mve_b2k3_fair1m_le90/epoch_20.pth \
        --num-batches 150
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint

from mmrotate.core import rbbox2roi
from mmrotate.datasets import build_dataset
from mmrotate.datasets.builder import build_structured_dataloader
from mmrotate.models import build_detector


def parse_args():
    p = argparse.ArgumentParser(description='Disentanglement linear-probe diagnostic')
    p.add_argument('config', help='model config (the MVE config)')
    p.add_argument('checkpoint', help='trained checkpoint')
    p.add_argument('--num-batches', type=int, default=150,
                   help='B x K batches to collect features from')
    p.add_argument('--max-samples', type=int, default=40000,
                   help='cap on collected positive proposals')
    p.add_argument('--probe-epochs', type=int, default=200)
    p.add_argument('--test-frac', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _unwrap(x):
    return x.data[0] if isinstance(x, DataContainer) else x


@torch.no_grad()
def collect_features(model, loader, num_batches, max_samples, device):
    """Return frozen disc/nuis embeddings + class/aug labels for positives."""
    roi = model.roi_head
    disc_bank, nuis_bank, cls_bank, aug_bank = [], [], [], []
    total = 0
    for bi, data in enumerate(loader):
        if bi >= num_batches or total >= max_samples:
            break
        img = _unwrap(data['img']).to(device)
        img_metas = _unwrap(data['img_metas'])
        gt_bboxes = [b.to(device) for b in _unwrap(data['gt_bboxes'])]
        gt_labels = [l.to(device) for l in _unwrap(data['gt_labels'])]

        x = model.extract_feat(img)
        # RPN runs on the pre-fusion FPN features; BCFN fusion is applied
        # afterwards, only for the RoI head (matches PETDet.simple_test).
        proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
        if model.with_fusion:
            x = model.fusion(x)

        sampling_results = []
        for i in range(len(img_metas)):
            assign_result = roi.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], None, gt_labels[i])
            sr = roi.bbox_sampler.sample(
                assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i],
                feats=[lvl[i][None] for lvl in x])
            if gt_bboxes[i].numel() == 0:
                sr.pos_gt_bboxes = gt_bboxes[i].new((0, gt_bboxes[i].size(-1))).zero_()
            else:
                sr.pos_gt_bboxes = gt_bboxes[i][sr.pos_assigned_gt_inds, :]
            sampling_results.append(sr)

        rois = rbbox2roi([res.bboxes for res in sampling_results])
        if rois.shape[0] == 0:
            continue
        br = roi._bbox_forward(x, rois)
        if br['disc'] is None:
            raise RuntimeError('Model has no disentangle head (disc is None).')
        bbox_targets = roi.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, roi.train_cfg)
        labels = bbox_targets[0]
        gathered = roi._gather_positives(labels, sampling_results, img_metas)
        if gathered is None:
            continue
        sel_idx, _obj, aug_ids, cls_ids = gathered

        disc_bank.append(br['disc'][sel_idx].cpu())
        nuis_bank.append(br['nuis'][sel_idx].cpu())
        cls_bank.append(cls_ids.cpu())
        aug_bank.append(aug_ids.cpu())
        total += sel_idx.numel()
        if bi % 10 == 0:
            print(f'  batch {bi}/{num_batches}  collected {total} positives')

    return (torch.cat(disc_bank), torch.cat(nuis_bank),
            torch.cat(cls_bank).long(), torch.cat(aug_bank).long())


def linear_probe(feats, targets, num_classes, device, epochs=200,
                 test_frac=0.3, seed=0):
    """Train a linear classifier on frozen feats; return (test_acc, majority)."""
    g = torch.Generator().manual_seed(seed)
    n = feats.size(0)
    perm = torch.randperm(n, generator=g)
    n_test = int(n * test_frac)
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, ytr = feats[tr].to(device), targets[tr].to(device)
    Xte, yte = feats[te].to(device), targets[te].to(device)

    clf = nn.Linear(feats.size(1), num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    # majority-class baseline on the test split
    maj = torch.bincount(yte, minlength=num_classes).max().item() / max(1, yte.numel())
    return acc, maj


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(device).eval()

    num_aug = cfg.data.structured_sampler['num_aug']
    num_classes = model.roi_head.bbox_head.num_classes

    dataset = build_dataset(cfg.data.train)
    loader = build_structured_dataloader(
        dataset, cfg.data.structured_sampler,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False, seed=args.seed)

    print('Collecting frozen features ...')
    disc, nuis, cls_ids, aug_ids = collect_features(
        model, loader, args.num_batches, args.max_samples, device)
    print(f'Collected {disc.size(0)} positive proposals '
          f'({num_classes} classes, {num_aug} augs).\n')

    print('Training linear probes ...')
    dc_acc, dc_maj = linear_probe(disc, cls_ids, num_classes, device,
                                  args.probe_epochs, args.test_frac, args.seed)
    da_acc, da_maj = linear_probe(disc, aug_ids, num_aug, device,
                                  args.probe_epochs, args.test_frac, args.seed)
    na_acc, na_maj = linear_probe(nuis, aug_ids, num_aug, device,
                                  args.probe_epochs, args.test_frac, args.seed)
    nc_acc, nc_maj = linear_probe(nuis, cls_ids, num_classes, device,
                                  args.probe_epochs, args.test_frac, args.seed)

    print('\n' + '=' * 64)
    print('Disentanglement probe matrix (linear, frozen features)')
    print('-' * 64)
    print(f'{"":14}{"-> CLASS acc":>22}{"-> AUG acc":>22}')
    print(f'{"from F_disc":14}{dc_acc * 100:>17.2f} (hi){da_acc * 100:>17.2f} (lo)')
    print(f'{"from F_nuis":14}{nc_acc * 100:>17.2f} (lo){na_acc * 100:>17.2f} (hi)')
    print('-' * 64)
    print(f'chance: class~{100.0 / num_classes:.2f}% (majority '
          f'{max(dc_maj, nc_maj) * 100:.1f}%), aug~{100.0 / num_aug:.2f}%')
    print('=' * 64)
    print('\nReading:')
    print(f'  disc->class {dc_acc*100:.1f}% should be HIGH  (content kept)')
    print(f'  disc->aug   {da_acc*100:.1f}% should be ~chance ({100.0/num_aug:.1f}%) '
          f'-> low = corruption removed from F_disc')
    print(f'  nuis->aug   {na_acc*100:.1f}% should be HIGH  (corruption kept)')
    print(f'  nuis->class {nc_acc*100:.1f}% should be ~majority '
          f'({max(dc_maj, nc_maj)*100:.1f}%) -> low = content removed from F_nuis')


if __name__ == '__main__':
    main()
