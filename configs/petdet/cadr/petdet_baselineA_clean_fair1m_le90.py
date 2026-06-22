"""Baseline A: clean PETDet (NO corruption augmentation).

The "lower" reference: high clean accuracy but large drops under corruption.
Together with Baseline B (corruption-aug) it frames the story:
    A (clean)      -> high clean, big corruption drop
    B (corrupt-aug)-> slightly lower clean, smaller drop
    CADR           -> recover clean + smaller drop than B

Identical protocol to Baseline B (12 epochs, step [8,11], 4 GPU x 2 = batch 8,
lr 0.02) EXCEPT no corruption augmentation. NaN-safe hook kept for parity.

Run:
    bash tools/dist_train.sh \
        configs/petdet/cadr/petdet_baselineA_clean_fair1m_le90.py 4 \
        --work-dir work_dirs/petdet_baselineA_clean
Evaluate (clean + FAIR1M-C) like Baseline B with tools/eval_corruption.py.
"""
_base_ = ['../petdet_r50_fpn_1x_fair1m_le90.py']

# clean training = stock PETDet train pipeline (no RandomCorruptionAugment).
# Parity guard only (clean training shouldn't NaN, but keep the same hook).
optimizer_config = dict(
    type='NaNSafeOptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2))
