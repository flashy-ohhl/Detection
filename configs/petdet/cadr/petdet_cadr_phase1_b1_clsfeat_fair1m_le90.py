"""CADR Phase 1 -- B1: classify from F_feat, keep D as a backbone regularizer.

Diagnostic to separate "the 128-d cls bottleneck" from "the disentangle
objective itself":

* classification path goes back to F_feat(1024) -> fc_cls (== baseline, no
  cls_proj bottleneck);
* F_disc / F_nuis and the bidirectional contrastive losses are still computed,
  so D now only *regularizes* the shared/backbone features (auxiliary), without
  bottlenecking classification.

Reading the result (clean AP50 on eval2k, vs Baseline B 68.24 / d128 CADR 64.21):
* clean recovers to ~67-68  -> the -4pp was purely the cls_proj bottleneck; D is
  healthy, just wired into the wrong cls path.
* clean still ~64-65        -> D's contrastive is polluting the backbone (likely
  the instance-level / class-blind disc contrastive pushing apart same-class
  objects) -> next fix the contrastive pos/neg definition (class-level SupCon).

Everything else (disc=128, K=2, contrastive weights, GRL off) is unchanged.
Run on the stable 2-GPU setup (total batch 8):
    CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash tools/dist_train.sh \
        configs/petdet/cadr/petdet_cadr_phase1_b1_clsfeat_fair1m_le90.py 2 \
        --work-dir work_dirs/petdet_cadr_phase1_b1 \
        --cfg-options data.structured_sampler.batch_base=2 data.samples_per_gpu=4
"""
_base_ = ['./petdet_cadr_phase1_sd_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(cls_from='feat')))
