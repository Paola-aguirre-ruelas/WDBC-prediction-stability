# Version 1.1 — post-hoc scientific interpretation correction

**Frozen material preserved:** `protocol/analysis_plan_v1.md`, original notebook, `scripts/run_wdbc_prediction_stability.py`, original splits/predictions/results/diagnostics and all legacy figures. No re-fitting, data changes or protocol amendments.

1. Added an independent, deterministic script reading *only published* observation summaries to check association of resampling widths with `median(p) × [1−median(p)]`, rank-residual concordance and probability-range sensitivity.
2. Added an explicitly exploratory local rolling-median positive-excess diagnostic (31/51/71 neighboring observations) and generated plots by diagnosis. No claims of biological mechanisms or intrinsic patient instability.
3. Replaced the long front-page README with a short methods/results overview. The complete previous README is archived verbatim as `protocol/README_v1_archive.md` by the apply script.
4. Added source citations and `CITATION.cff`; no Zenodo DOI is claimed.
5. Clarified that historical `metadata/notebook_cleanup_20260926/AUDIT.md` refers to an audit ZIP that is not in the published repository. Historical audit log remains unchanged.
6. Added three focused deterministic unit tests. New outputs are written only to `analysis/generated_v11/`.

**Not included:** bootstrap, new samples, external clinical validation, changes to original figures/notebook, shortening the underlying archived methods, an expensive 50-repeat model rerun, or a public GitHub release/DOI. Proposed release is a reviewable v1.1 commit *after* checking the generated artifacts.
