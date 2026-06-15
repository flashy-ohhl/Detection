"""MVE config: PETDet + disentangle head + bidirectional contrastive loss.

B=2 base images x K=3 augmentations (clean / fog / gaussian_noise), image-level
contrast, F_disc -> Linear(128->1024) -> ARL classifier, 20 epochs.

Run (single GPU) on the remote server:
    python tools/train.py configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py
"""
_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

angle_version = 'le90'

# ----------------------------------------------------------------------------
# Model: swap in the disentangle ARL head + contrastive RoI head.
# Everything else (BCFN, QualityOrientedRPN, ARL loss, coders) is inherited.
# ----------------------------------------------------------------------------
model = dict(
    roi_head=dict(
        type='PETDetRoIHead',
        contrastive_loss=dict(
            type='BidirectionalContrastiveLoss',
            temperature=0.07,
            loss_weight_disc=0.5,
            loss_weight_nuis=0.5),
        bbox_head=dict(
            type='RotatedShared2FCBBoxDisentangleARLHead',
            disentangle=dict(
                hidden_channels=512,
                disc_channels=128,
                nuis_channels=128))))

# ----------------------------------------------------------------------------
# Data: B x K structured sampler + multi-corruption augmentation.
# ----------------------------------------------------------------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# K=3 pool. aug_id 0=clean (anchor), 1=fog, 2=gaussian_noise. severity random.
aug_pool = [
    dict(corruption=None, severity=None),
    dict(corruption='fog', severity=[1, 2]),
    dict(corruption='gaussian_noise', severity=[1, 2]),
]

# meta_keys must carry original_id / aug_id through to img_metas.
meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
             'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
             'original_id', 'aug_id')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # corruption AFTER geometric aug, BEFORE Normalize (expects uint8 HWC).
    # backend='builtin' is size-safe on 1024x1024 tiles (imagecorruptions' fog
    # breaks above 256px). Match this to your eval-set corruption generator.
    dict(type='MultiCorruptionAugment', aug_pool=aug_pool, backend='builtin'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=meta_keys),
]

data = dict(
    # samples_per_gpu is ignored by the structured loader (batch = B*K = 6);
    # kept for the val loader.
    samples_per_gpu=2,
    workers_per_gpu=2,
    # presence of this key activates build_structured_dataloader in train.py.
    structured_sampler=dict(num_aug=3, batch_base=2, shuffle=True,
                            drop_last=True),
    train=dict(
        type='StructuredFAIR1MDataset',
        num_aug=3,
        version=angle_version,
        pipeline=train_pipeline),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

# ----------------------------------------------------------------------------
# Schedule: 20 epochs.
# ----------------------------------------------------------------------------
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 2000,
    step=[14, 18])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# evaluate every 5 epochs and keep the best mAP checkpoint
evaluation = dict(interval=5, metric='mAP', save_best='mAP')
checkpoint_config = dict(interval=5)

# NOTE: effective batch = B*K = 6 imgs/GPU (vs 2 for the baseline). If training
# is unstable, lower optimizer.lr below or warm up longer.
# To finetune from a trained PETDet baseline instead of training from scratch,
# set: load_from = '<path-to-petdet-baseline>.pth'
