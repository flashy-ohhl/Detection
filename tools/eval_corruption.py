# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-corruption evaluation (module 6 of the MVE plan).

Evaluates one checkpoint over a set of pre-generated corruption x severity
image folders and prints an AP50 / mAP table with the drop (in points) vs the
clean variant.

Assumes the corrupted images live under a common root, one folder per variant,
and that every variant shares the *same* annotation files (image-level
corruptions do not move boxes):

    <corruption-root>/
        clean/images/*.png
        fog_2/images/*.png
        gaussian_noise_1/images/*.png
        ...
    <ann-file>/                 # shared annfiles for the val subset

Example:
    python tools/eval_corruption.py \
        configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py \
        work_dirs/petdet_mve_b2k3_fair1m_le90/best_mAP.pth \
        --corruption-root data/fair1m_val2k_corrupted \
        --ann-file data/fair1m_val2k_corrupted/clean/annfiles \
        --variants clean fog_2 gaussian_noise_1 brightness_2 defocus_blur_1
"""
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a checkpoint across image corruptions')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--corruption-root',
        required=True,
        help='root dir containing one folder per corruption variant')
    parser.add_argument(
        '--ann-file',
        required=True,
        help='shared annotation folder for the val subset')
    parser.add_argument(
        '--variants',
        nargs='+',
        default=[
            'clean', 'brightness_2', 'fog_2', 'gaussian_noise_1',
            'defocus_blur_1'
        ],
        help='variant folder names (first one is the clean reference)')
    parser.add_argument(
        '--images-subdir',
        default='images',
        help="image subfolder name inside each variant folder ('' for none)")
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for the reported AP (AP50 by default)')
    parser.add_argument(
        '--out', default=None, help='optional path to dump a json summary')
    return parser.parse_args()


def build_model(cfg, checkpoint):
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = ckpt.get('meta', {}).get('CLASSES', None)
    return MMDataParallel(model.cuda(), device_ids=[0]).eval()


def eval_one(model, cfg, ann_file, img_prefix, iou_thr):
    val_cfg = cfg.data.val.copy()
    val_cfg['ann_file'] = ann_file
    val_cfg['img_prefix'] = img_prefix
    val_cfg['test_mode'] = True
    if cfg.data.get('test_dataloader', {}).get('samples_per_gpu', 1) > 1:
        val_cfg['pipeline'] = replace_ImageToTensor(val_cfg['pipeline'])

    dataset = build_dataset(val_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False,
        shuffle=False)
    results = single_gpu_test(model, loader)
    metrics = dataset.evaluate(results, metric='mAP', iou_thr=iou_thr)
    ap_key = f'AP{int(iou_thr * 100):02d}'
    return float(metrics.get(ap_key, 0.0)), float(metrics.get('mAP', 0.0))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # the val set is a plain FAIR1M dataset (no structured sampler)
    cfg.data.pop('structured_sampler', None)
    if cfg.data.val.get('type', '').startswith('Structured'):
        cfg.data.val['type'] = 'FAIR1MDataset'

    model = build_model(cfg, args.checkpoint)

    table = OrderedDict()
    for variant in args.variants:
        img_prefix = osp.join(args.corruption_root, variant)
        if args.images_subdir:
            img_prefix = osp.join(img_prefix, args.images_subdir)
        ap50, mean_ap = eval_one(model, cfg, args.ann_file, img_prefix,
                                 args.iou_thr)
        table[variant] = dict(ap50=ap50, mAP=mean_ap)

    clean_ref = args.variants[0]
    clean_ap50 = table[clean_ref]['ap50']

    print('\n' + '=' * 60)
    print(f'{"variant":<22}{"AP50":>10}{"mAP":>10}{"dAP50(pp)":>14}')
    print('-' * 60)
    for variant, m in table.items():
        drop = (m['ap50'] - clean_ap50) * 100
        print(f'{variant:<22}{m["ap50"] * 100:>10.2f}'
              f'{m["mAP"] * 100:>10.2f}{drop:>14.2f}')
    print('=' * 60)

    if args.out:
        mmcv.dump(table, args.out)
        print(f'summary written to {args.out}')


if __name__ == '__main__':
    main()
