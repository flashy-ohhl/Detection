# Copyright (c) OpenMMLab. All rights reserved.
"""Generate FAIR1M-C: corrupted copies of a clean val set.

Reads clean images from ``--src-images`` and writes, for every
(corruption, severity) in the pool, a corrupted copy under
``--dst/{corruption}_{severity}/images/``. Annotations are NOT copied
(corruption does not move boxes) -- point the evaluator at the shared annfiles.

Uses the SAME size-safe builtin corruption code as training, so the train and
test degradation distributions agree. A fixed seed makes the benchmark
reproducible.

Layout produced::

    <dst>/
        clean/images/...            (symlinks to the source, reference)
        gaussian_noise_1/images/...
        gaussian_noise_2/images/...
        ...
        spatter_3/images/...

Evaluate with tools/eval_corruption.py (--corruption-root <dst>,
--ann-file <shared annfiles>).

Example:
    python tools/gen_fair1m_c.py \
        --src-images data/split_ss_fair1m2_0_eval2k/images \
        --dst        data/split_ss_fair1m2_0_C
"""
import argparse
import os
import os.path as osp

import mmcv

from mmrotate.datasets.pipelines.corruption import apply_corruption


def parse_args():
    p = argparse.ArgumentParser(description='Generate FAIR1M-C corrupted val set')
    p.add_argument('--src-images', required=True,
                   help='clean image folder (e.g. .../eval2k/images)')
    p.add_argument('--dst', required=True,
                   help='output root for the FAIR1M-C variants')
    p.add_argument(
        '--corruptions',
        nargs='+',
        default=['gaussian_noise', 'defocus_blur', 'brightness', 'fog',
                 'spatter'])
    p.add_argument('--severities', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--backend', default='builtin')
    p.add_argument('--ext', default='.png', help='image extension')
    p.add_argument('--seed', type=int, default=0, help='reproducibility seed')
    p.add_argument('--no-clean-link', action='store_true',
                   help='do not create the clean/ symlink variant')
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)

    files = sorted(mmcv.scandir(args.src_images, suffix=args.ext))
    assert len(files) > 0, f'no {args.ext} images in {args.src_images}'
    print(f'{len(files)} clean images in {args.src_images}')

    # clean reference variant (symlinks; falls back to copy if symlink fails)
    if not args.no_clean_link:
        clean_dir = osp.join(args.dst, 'clean', 'images')
        mmcv.mkdir_or_exist(clean_dir)
        for f in files:
            link = osp.join(clean_dir, f)
            src = osp.abspath(osp.join(args.src_images, f))
            if not osp.exists(link):
                try:
                    os.symlink(src, link)
                except OSError:
                    import shutil
                    shutil.copyfile(src, link)
        print(f'clean -> {clean_dir} ({len(files)} links)')

    for corruption in args.corruptions:
        for sev in args.severities:
            variant = f'{corruption}_{sev}'
            out_dir = osp.join(args.dst, variant, 'images')
            mmcv.mkdir_or_exist(out_dir)

            prog = mmcv.ProgressBar(len(files))
            for f in files:
                img = mmcv.imread(osp.join(args.src_images, f))  # BGR uint8
                out = apply_corruption(img, corruption, sev, args.backend)
                mmcv.imwrite(out, osp.join(out_dir, f))
                prog.update()
            print(f'\n{variant} -> {out_dir}')

    print('\nDONE. Evaluate with:')
    print(f'  python tools/eval_corruption.py <config> <ckpt> \\')
    print(f'    --corruption-root {args.dst} \\')
    print(f'    --ann-file <shared annfiles> \\')
    print('    --variants clean ' +
          ' '.join(f'{c}_{s}' for c in args.corruptions
                   for s in args.severities))


if __name__ == '__main__':
    main()
