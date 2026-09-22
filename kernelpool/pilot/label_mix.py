#!/usr/bin/env python3
"""Fraction of exact (spectroscopic) labels per label-kernel magnitude bin of a trained variant (ours).

Why (expert review 2026-09-17, finding 1): the `union`/`union_w` variants train on the spectroscopic
redshift where one exists and on the many-band label otherwise, so at bright magnitudes most training
labels were exact and the estimator's PDF is not p(z_label | x) but a mixture. The Richardson-Lucy
step must then use the effective kernel K_eff = f delta + (1 - f) K, with f the (weighted) fraction
of exact labels among the training rows of that kernel magnitude bin. This script reads the run's
`fit_<variant>.hdf5` (the rows the estimator actually saw, with the `weight` column for union_w) and
writes `label_mix_<variant>.json` next to it; `manyband_mixture.py apply --label-mix auto` uses it.

Label source per row: the fit file's `redshift` column is the training label; a row's label was the
many-band value iff it equals `redshift_manyband` (exact equality; a spectroscopic redshift never
coincides with an independent noisy draw).

  python pilot/label_mix.py pilot/runs/ts3exp_cardinal_1yr_E1_magerr --variant union_w
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--variant", default="union_w"); ap.add_argument("--mag-col", default="mag_i_lsst")
    args = ap.parse_args()
    import tables_io
    from manyband_mixture import M, N_MAGBIN, mag_bin
    from pz_pilot import log

    run = Path(args.run)
    fit = tables_io.read(str(run / f"fit_{args.variant}.hdf5"))
    fit = {k: np.asarray(v) for k, v in fit.items()}
    label = fit["redshift"].astype(float)
    zmb = fit["redshift_manyband"].astype(float) if "redshift_manyband" in fit else np.full(len(label), np.nan)
    from_manyband = np.isfinite(zmb) & (label == zmb)
    exact = ~from_manyband
    w = fit["weight"].astype(float) if "weight" in fit else np.ones(len(label))
    mi = fit[args.mag_col].astype(float)
    kb = mag_bin(mi)
    f_w, f_u, n = [], [], []
    for k in range(N_MAGBIN):
        s = kb == k
        n.append(int(s.sum()))
        f_w.append(float(np.sum(w[s] * exact[s]) / np.sum(w[s])) if s.any() else float("nan"))
        f_u.append(float(exact[s].mean()) if s.any() else float("nan"))
    out = dict(variant=args.variant, kernel_mag_edges=[float(v) for v in M["mag_i_bin_edges"]], n_rows=n,
               f_exact_by_kernel_bin=f_w, f_exact_unweighted=f_u, weighted="weight" in fit,
               n_exact=int(exact.sum()), n_manyband=int(from_manyband.sum()))
    (run / f"label_mix_{args.variant}.json").write_text(json.dumps(out, indent=1))
    logf = open(run / "run.log", "a")
    log(f"=== label_mix variant={args.variant}: exact-label fraction per kernel bin {M['mag_i_bin_edges'].tolist()}: "
        f"weighted {np.round(f_w, 3).tolist()} (unweighted {np.round(f_u, 3).tolist()}, rows {n}) -> label_mix_{args.variant}.json", logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
