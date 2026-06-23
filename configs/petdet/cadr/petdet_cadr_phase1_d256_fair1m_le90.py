"""CADR Phase 1 (S+D) with a wider disc embedding (128 -> 256).

Make-or-break experiment: the d128 run lost ~4pp clean to Baseline B (the
128-d classification bottleneck). Widen F_disc to 256 to give fine-grained
classification more capacity, keeping the corruption-invariant "cls via F_disc"
mechanism. Everything else identical to phase1_sd.

Decision rule:
* clean recovers toward ~68 (== Baseline B) -> the smaller corruption drops flip
  CADR to winning on hard corruptions -> proceed to Phase 2 (restoration).
* clean stays ~65-66 -> the invariance/clean trade-off is fundamental ->
  reconsider the pure-disentanglement route.

Run (4 GPU, total batch 8, no overrides needed):
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
        configs/petdet/cadr/petdet_cadr_phase1_d256_fair1m_le90.py 4 \
        --work-dir work_dirs/petdet_cadr_phase1_d256
"""
_base_ = ['./petdet_cadr_phase1_sd_fair1m_le90.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            disentangle=dict(
                hidden_channels=512,
                disc_channels=256,   # 128 -> 256 (the only change)
                nuis_channels=128))))
