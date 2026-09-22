# kernelpool

LSST-DESC PZ data challenge entry for task sets 3 and 4 (Bonn, 2026): a pool
of FlexZBoost (two settings) and GPz with per-i-band-bin weights and a
per-bin calibration, both fitted by the likelihood of the many-band labels
of a COSMOS hold-out seen through the organizers' label error model
(rectangular kernel, true redshift on the 0–3 PDF grid, labels on 0–6);
raw-pool point estimate. Method: `pz_method_writeup.md`.

Code and text written by Claude (Fable models); direction, critical review
and all decisions by Thomas Reiprich (Bonn). `kernelpool/pilot/` holds the
scripts the entry was developed and validated with (ours, except the label
model table in `manyband_mixture.py`, copied from RAIL's
`GaussianSkewtScatterSelector`); `kernelpool/pipeline.py` is the driver
the organizers' harness calls.

Entry points: `train_and_estimate(train_file, test_file, output_file,
model_file=None)` and `estimate_only(model_file, test_file, output_file)`;
command line `python -m kernelpool train|estimate ...`.
`environment_desc-pz.txt` is the `pip freeze` of the environment the
shipped models were built in; `build_submission.sh` rebuilds the eight
submission files and model bundles and runs the checks.
