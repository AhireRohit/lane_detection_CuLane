# Phase F temporal/geometry postprocessing (2026-11-19)
scripts/temporal_ablation.py: B0 raw, B1 geometry filter, B2 causal tracker (Hungarian match, EMA alpha 0.5, gate 40 canvas px,
confirm 2 hits, hold <=3 frames dashed), B3 = B1+B2. Params fixed a priori, no search. B1 thresholds = val-GT percentiles
(extent p0.5, curvature p99.5, slope p99.5) + crossing rule; false-rejection rate on CULane val GT reported.
Offline on saved baseline frames.jsonl (same checkpoint/threshold); strictly causal. CPU Condor job. No vendor code modified.
