"""Baseline B: PETDet + corruption augmentation (NO disentangle / restoration).

This is the REAL competitor for CADR-Lite v2 (not clean PETDet). It sees the
exact same corruption pool / severities / exposure as CADR, so any gain of CADR
over this baseline is attributable to the method, not to the augmentation.

- 50% clean / 50% corrupted per image (matches K=2: 1 clean + 1 corrupt per pair)
- pool = {gaussian_noise, defocus_blur, brightness, fog, spatter}, severity {1,2,3}
- standard single-image loader (no B x K), standard PETDet head
- 12 epochs, lr step [8,11] (PETDet FAIR1M protocol)

Run:
    python tools/train.py configs/petdet/cadr/petdet_baselineB_corruptaug_fair1m_le90.py \
        --work-dir work_dirs/petdet_baselineB_corruptaug
"""
_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Keep this pool / severities / prob IDENTICAL to the CADR structured loader.
corruption_pool = ['gaussian_noise', 'defocus_blur', 'brightness', 'fog',
                   'spatter']

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # 50% clean / 50% random corruption, before Normalize.
    dict(
        type='RandomCorruptionAugment',
        corruptions=corruption_pool,
        severities=[1, 2, 3],
        prob=0.5,
        backend='builtin'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(train=dict(pipeline=train_pipeline))

# 12 epochs + step [8,11] are inherited from the PETDet base config
# (schedule_1x + qopn lr_config). Stated here only for clarity:
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# lr_config step = [8, 11]
