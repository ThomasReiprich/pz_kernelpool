# kernelpool

LSST-DESC PZ data challenge entry (Bonn, 2026; task sets 3 and 4 in
v1.0.0, their files unchanged since; task sets 1 and 2 added in v1.1.0
and first submitted in v1.2.0): a pool
of FlexZBoost (two settings) and GPz with per-i-band-bin weights and a
per-bin calibration, both fitted by the likelihood of the many-band labels
of a COSMOS hold-out seen through the organizers' label error model
(rectangular kernel, true redshift on the 0–3 PDF grid, labels on 0–6);
raw-pool point estimate. On task sets 1 and 2 (exact labels) the same
pool is fitted against a hold-out of the training file, without the
kernel and without the PIT map; task set 2 gets a 3 % uniform floor in
every PDF, and since v1.2.0 its objects fainter than i = 24.5 get an
equal-weight mixture of the pool PDF and a BPZ template-fitting
posterior (31 COSMOS templates, HDFN prior, zero points from the training
file), with the median of the mixed PDF as point estimate (see the
write-up). Method: `pz_method_writeup.md`.

Code and text written by Claude (Fable models); direction, critical review
and all decisions by Thomas Reiprich (Bonn). `kernelpool/pilot/` holds the
scripts the entry was developed and validated with (ours, except the label
model table in `manyband_mixture.py`, copied from RAIL's
`GaussianSkewtScatterSelector`); `kernelpool/pilot/bpz_member.py` (ours)
wraps RAIL's BPZ port for the v1.2.0 faint-end mixture, and
`kernelpool/pilot/bpz_data/` holds the COSMOS templates, the LSST and
Roman filter curves and the derived template–filter tables, copied or
generated from `pz-rail-base`'s example data; `kernelpool/pipeline.py` is
the driver the organizers' harness calls.

Entry points: `train_and_estimate(train_file, test_file, output_file,
model_file=None)` and `estimate_only(model_file, test_file, output_file)`;
command line `python -m kernelpool train|estimate ...`.
`environment_desc-pz.txt` is the `pip freeze` of the environment the
shipped models were built in; `build_submission.sh` rebuilds the sixteen
submission files and model bundles and runs the checks.

## License

The code in this repository is released under the MIT license (see
`LICENSE`). It builds on RAIL, FlexZBoost/flexcode, GPz and XGBoost,
which are not part of this repository and carry their own (MIT / BSD /
Apache) licenses, and since v1.2.0 on BPZ (`desc-bpz`, `pz-rail-bpz`,
MIT), whose templates and filter curves are redistributed under
`kernelpool/pilot/bpz_data/`; the label-model parameter table in
`kernelpool/pilot/manyband_mixture.py` is copied from RAIL's
`GaussianSkewtScatterSelector` (`rail_astro_tools`, MIT).
