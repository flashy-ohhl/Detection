# MVE: Disentangle head + bidirectional contrastive loss

Proof-of-concept that adds corruption-disentangled representation learning on
top of PETDet (FAIR1M-v2.0). B=2 base images x K=3 augmentations
(clean / fog / gaussian_noise), image-level contrast, 20 epochs.

## What was added

| Module | File |
|---|---|
| StructuredBatchSampler (B x K matrix) | `mmrotate/datasets/samplers/structured_batch_sampler.py` |
| Structured dataset (injects `original_id` / `aug_id`) | `mmrotate/datasets/structured_dataset.py` |
| MultiCorruptionAugment + size-safe corruptions | `mmrotate/datasets/pipelines/corruption.py`, `corruptions_builtin.py` |
| DisentangleHead (F_disc / F_nuis) | `mmrotate/models/roi_heads/disentangle_head.py` |
| BidirectionalContrastiveLoss | `mmrotate/models/losses/bidirectional_contrastive_loss.py` |
| Disentangle ARL bbox head | `mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_disentangle_arl_head.py` |
| PETDetRoIHead (image-level pooling + contrastive loss) | `mmrotate/models/roi_heads/petdet_roi_head.py` |
| Structured dataloader hook | `mmrotate/datasets/builder.py`, `mmrotate/apis/train.py` |
| MVE config | `configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py` |
| Multi-corruption eval | `tools/eval_corruption.py` |
| CPU unit tests | `tests/test_mve_components.py` |

## Data flow

```
StructuredBatchSampler ── idx = base*K + aug ──> StructuredFAIR1MDataset
   (B x K complete matrix)        sets results['original_id'], ['aug_id']
        │
        ▼  MultiCorruptionAugment (reads aug_id, before Normalize)
   Collect(meta_keys += original_id, aug_id)  ──> img_metas
        │
backbone → FPN → BCFN → RPN → ROIAlign → shared_2fc (F_feat, 1024)
        ├─ MLP_disc → F_disc(128) ─┬─ Linear(128→1024) → ARL fc_cls
        │                          └─ image-level mean-pool(pos) → L_disc_bicon
        ├─ MLP_nuis → F_nuis(128) ──── image-level mean-pool(pos) → L_nuis_bicon
        └─ F_feat(1024) ───────────────────────────────────────→ fc_reg
```

`L_total = L_rpn + L_cls(ARL via F_disc) + L_bbox_reg + 0.5·L_disc + 0.5·L_nuis`

## Run (single GPU, remote server)

```bash
python tools/train.py configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py
```

Effective batch = B*K = 6 imgs/GPU (baseline was 2) — watch GPU memory; lower
`optimizer.lr` or extend warmup if unstable. To finetune from a trained PETDet
baseline, set `load_from` in the config.

## Evaluate across corruptions

```bash
python tools/eval_corruption.py \
    configs/petdet/mve/petdet_mve_b2k3_fair1m_le90.py \
    work_dirs/petdet_mve_b2k3_fair1m_le90/best_mAP.pth \
    --corruption-root <path-to-your-pregenerated-val-corruptions> \
    --ann-file       <path-to-shared-val-annfiles> \
    --variants clean fog_2 gaussian_noise_1 brightness_2 defocus_blur_1
```

Each variant folder holds the corrupted images (`<root>/<variant>/images/`);
annotations are shared (corruption does not move boxes).

## Verify on the server (could not be tested locally — no mmdet here)

1. **Config builds**: `python tools/train.py <cfg> --cfg-options runner.max_epochs=1`
   for a smoke run, or build the model/dataset in a python shell.
2. **`img_metas` carries ids**: confirm `original_id` / `aug_id` reach the RoI
   head (the head asserts their presence).
3. **Corruption backend matches your eval set**: the training pool uses
   `backend='builtin'` (size-safe). If your pre-generated eval corruptions came
   from the upstream `imagecorruptions` package, align them (the upstream `fog`
   differs from / crashes on 1024px tiles).
4. **Losses appear**: `loss_disc_bicon` / `loss_nuis_bicon` should show up in
   the log and be finite/decreasing.

## Tested locally (CPU, `tests/test_mve_components.py`, 8/8 pass)

B x K matrix completeness, distributed partition, per-epoch reshuffle,
contrastive mask construction, InfoNCE positivity / discriminativeness /
backward, DisentangleHead shapes + L2 norm, builtin fog/noise at 1024x1024.
