"""CADR-Lite v2 -- Phase 1 (S + D): geometric-paired B x K + disentangle contrast.

Phase 1 validates the disentanglement half of CADR before adding severity (N)
and conditional restoration (R) in Phase 2.

Design (see the project notes):
* S: structured B x K loader, K=2 (clean + 1 random corruption from the 5-pool,
  imagecorruptions backend to match the FAIR1M-C benchmark).
* D: disentangle head (F_disc / F_nuis) + bidirectional proposal-level
  supervised contrast. GRL is OFF in Phase 1.

Batch: 4 GPU x (batch_base=1 x K=2) = total batch 8 == Baseline B (fair, no
retrain). With 1 base image per GPU the DISC contrast still works (objects of
that image, clean vs corrupt); the NUIS contrast needs different images per GPU
so it is down-weighted here (it gets properly shaped by L_sev / cross-GPU gather
in Phase 2).

Run:
    bash tools/dist_train.sh \
        configs/petdet/cadr/petdet_cadr_phase1_sd_fair1m_le90.py 4 \
        --work-dir work_dirs/petdet_cadr_phase1
"""
_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 5-corruption pool / severities -- identical to Baseline B.
corruption_pool = ['gaussian_noise', 'defocus_blur', 'brightness', 'fog',
                   'spatter']

# ----------------------------------------------------------------------------
# Model: disentangle ARL head + contrastive RoI head, GRL disabled.
# ----------------------------------------------------------------------------
model = dict(
    roi_head=dict(
        type='PETDetRoIHead',
        contrast_level='proposal',
        num_aug=2,
        max_pos_per_img=256,
        contrastive_loss=dict(
            type='BidirectionalContrastiveLoss',
            temperature=0.07,
            loss_weight_disc=0.5,
            loss_weight_nuis=0.1),   # weak in Phase 1 (1 image/GPU)
        disentangle_adv=dict(enable=False),   # GRL OFF in Phase 1
        bbox_head=dict(
            type='RotatedShared2FCBBoxDisentangleARLHead',
            disentangle=dict(
                hidden_channels=512, disc_channels=128, nuis_channels=128))))

# ----------------------------------------------------------------------------
# Data: K=2 structured loader, random corruption (imagecorruptions backend).
# ----------------------------------------------------------------------------
meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
             'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
             'original_id', 'aug_id', 'corruption_id', 'severity')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # aug_id 0 -> clean, aug_id 1 -> random corruption (matches Baseline B).
    dict(
        type='MultiCorruptionAugment',
        random_corruptions=corruption_pool,
        random_severities=[1, 2, 3],
        backend='imagecorruptions'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=meta_keys),
]

data = dict(
    samples_per_gpu=2,   # = batch_base(1) x K(2); ignored by structured loader
    workers_per_gpu=2,
    structured_sampler=dict(num_aug=2, batch_base=1, shuffle=True,
                            drop_last=True),
    train=dict(
        type='StructuredFAIR1MDataset',
        num_aug=2,
        version=angle_version,
        pipeline=train_pipeline),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

# NaN-safe optimizer: skip non-finite OR exploding (>max_loss) iterations.
optimizer_config = dict(
    type='NaNSafeOptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2),
    max_loss=1000.0)

# The nuisance branch / contrastive losses are only conditionally in the graph
# (e.g. when a batch has too few positives), so let DDP tolerate that.
find_unused_parameters = True

# 12 epochs / step [8,11] / lr 0.02 inherited (matches Baseline A/B).
