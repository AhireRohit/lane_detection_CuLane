# UFLDv2 no-TTA full test (2026-11-19)
config/culane_res34_cluster_notta.py = cluster config with tta=False (only change). scripts/run_ufld_full_notta.py, PHASE
ufldv2/full_notta (gpu_entry backup .before_notta). Purpose: F1 in the same setting as speed.csv and dashcam (non-TTA).
Scored with scripts/score_culane_official.sh. The TTA result (75.98) remains the author-default reproduction.
- Job 73894: inference complete (34680 files); vendor's own post-inference call_culane_eval failed (relative path ./evaluation/culane/evaluate not found from cwd). Not needed: scored with scripts/score_culane_official.sh instead.
