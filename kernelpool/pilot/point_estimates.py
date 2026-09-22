#!/usr/bin/env python3
"""Point estimators from p(z) and the organizers' scored point metrics.

Why: the PZ challenge scores the point estimate it finds in ancil["zmode"] with the
biweight statistics of RAIL's `PZPlotterPointEstimateVsTrueHist2D` (rail_projects,
src/rail/plotting/pz_plotters.py): dz = (z_point - z_true)/(1 + z_true) is sigma-clipped
at 3 sigma four times, then mean = biweight location, std = biweight scale of the clipped
sample, outlier_rate = fraction with |dz| > 3 * that scale, abs_outlier_rate = fraction with
|dz| > 0.2 (all defaults at pz_data_challenge main 3bcbd12). The point estimate is a free
choice, decoupled from the PDF (adversarial review 2026-09-15, item B), so this module
(ours) provides several estimators and the scorer's own metrics to choose between them.

Estimators (all evaluated on the grid PDF):
  mode        argmax of the PDF (RAIL's zmode)
  mean        first moment
  median      50 % quantile of the CDF (linearly interpolated between grid points)
  massmode    the local maximum whose window of +-w(1+z) around it holds the most
              probability (w = 0.05 by default; `massmode03`, `massmode10` for w = 0.03, 0.10)
              -- for multimodal faint PDFs this prefers the broad, probable solution over the
              narrowest spike

Study mode compares them on a run's validation files against the spec-z rows (truth) and the
many-band labels per i-band bin, for the raw and the calibrated PDFs:
  python pilot/point_estimates.py pilot/runs/ts3exp_cardinal_1yr --variant union_w
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

METHODS = ("mode", "mean", "median", "massmode03", "massmode", "massmode10")


def organizer_point_metrics(z_true: np.ndarray, z_point: np.ndarray, n_clip: int = 3, abs_thresh: float = 0.2) -> dict:
    """The scored point metrics, computed as in rail.plotting.pz_plotters (RAIL's code, re-implemented here)."""
    from astropy.stats import biweight_location, biweight_scale
    from scipy.stats import sigmaclip
    dz = (np.asarray(z_point, float) - z_true) / (1.0 + z_true)
    dz = dz[np.isfinite(dz)]
    clipped, _, _ = sigmaclip(dz, low=3, high=3)
    for _ in range(n_clip):
        clipped, _, _ = sigmaclip(clipped, low=3, high=3)
    scale = float(biweight_scale(clipped))
    return dict(n=int(len(dz)), n_clipped=int(len(clipped)), mean=float(biweight_location(clipped)), std=scale,
                outlier_rate=float(np.mean(np.abs(dz) > 3 * scale)), abs_outlier_rate=float(np.mean(np.abs(dz) > abs_thresh)))


def _local_maxima(p: np.ndarray) -> np.ndarray:
    """Boolean mask of interior local maxima (plateaus count at their first cell), plus the edges if rising."""
    left = np.concatenate([np.full((p.shape[0], 1), -np.inf), p[:, :-1]], axis=1)
    right = np.concatenate([p[:, 1:], np.full((p.shape[0], 1), -np.inf)], axis=1)
    return (p > left) & (p >= right)


def point_estimate(p: np.ndarray, x: np.ndarray, method: str = "mode") -> np.ndarray:
    from calibrate_pdfs import cdf_grid, normalise
    p = normalise(np.maximum(p, 0), x)
    if method == "mode":
        return x[np.argmax(p, axis=1)]
    if method == "mean":
        return np.trapezoid(p * x, x, axis=1) if hasattr(np, "trapezoid") else np.trapz(p * x, x, axis=1)
    if method == "median":
        # linear interpolation of the CDF crossing; the first grid point with F >= 0.5 is biased upward by
        # half a grid step on average (+0.005 in z; expert review 2026-09-17, finding 2)
        F = cdf_grid(p, x)
        idx = np.clip(np.argmax(F >= 0.5, axis=1), 1, len(x) - 1)
        rows = np.arange(p.shape[0])
        F0, F1 = F[rows, idx - 1], F[rows, idx]
        t = np.where(F1 > F0, (0.5 - F0) / np.maximum(F1 - F0, 1e-300), 1.0)
        return x[idx - 1] + np.clip(t, 0.0, 1.0) * (x[idx] - x[idx - 1])
    if method.startswith("massmode"):
        w = {"massmode03": 0.03, "massmode": 0.05, "massmode10": 0.10}[method]
        F = cdf_grid(p, x)
        peaks = _local_maxima(p)
        n, m = p.shape
        best = np.full(n, np.nan); best_mass = np.full(n, -1.0)
        dx = x[1] - x[0]
        for j in range(m):
            has = peaks[:, j]
            if not has.any():
                continue
            zc = x[j]; half = w * (1.0 + zc)
            lo = int(np.clip(round((zc - half - x[0]) / dx), 0, m - 1)); hi = int(np.clip(round((zc + half - x[0]) / dx), 0, m - 1))
            mass = F[:, hi] - F[:, lo]
            better = has & (mass > best_mass)
            best[better] = zc; best_mass[better] = mass[better]
        # objects without any interior maximum (monotonic PDFs): fall back to the mode
        nomax = ~np.isfinite(best)
        if nomax.any():
            best[nomax] = x[np.argmax(p[nomax], axis=1)]
        return best
    raise ValueError(method)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--variant", default="union_w")
    ap.add_argument("--pdfs", default="raw,deconv_cal", help="which PDF sets to evaluate (suffixes of pz_valid_<variant>[_<suffix>].hdf5; 'raw' = none)")
    ap.add_argument("--bins", default="0,23,24,24.7,99")
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()

    import h5py
    from calibrate_pdfs import normalise
    from pz_pilot import ZMIN, ZMAX, NZBINS, log, point_metrics
    run = Path(args.run)
    logf = open(run / "run.log", "a")
    log(f"=== point_estimates study variant={args.variant} pdfs={args.pdfs}", logf)
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    edges = [float(v) for v in args.bins.split(",")]
    methods = args.methods.split(",")

    def load(fn):
        with h5py.File(run / fn) as f:
            x = f["meta/xvals"][:].ravel(); p = f["data/yvals"][:]
            anc = {k: f["ancil"][k][:].ravel() for k in f["ancil"]}
        assert np.allclose(x, zgrid)
        return normalise(np.maximum(p, 0), x), anc

    results = {}
    for vname, base_fn in (("V1", f"pz_valid_{args.variant}.hdf5"), ("V2", f"pz_valid_{args.variant}_v2.hdf5")):
        for suffix in args.pdfs.split(","):
            fn = base_fn if suffix == "raw" else base_fn.replace(".hdf5", f"_{suffix}.hdf5")
            if not (run / fn).exists():
                log(f"{fn} missing, skipped", logf); continue
            p, anc = load(fn)
            mi = anc["mag_i"].astype(float); zl = anc["z_manyband"].astype(float); zs = anc["z_spec"].astype(float)
            key = f"{vname}/{suffix}"; results[key] = {}
            ests = {m: point_estimate(p, zgrid, m) for m in methods}
            for lo, hi in list(zip(edges[:-1], edges[1:])) + [(0.0, 99.0)]:
                s = (mi >= lo) & (mi < hi)
                bkey = "all" if (lo, hi) == (0.0, 99.0) else f"i{lo}_{hi}"
                if s.sum() < 100:
                    continue
                hs = s & np.isfinite(zs)
                results[key][bkey] = dict(n=int(s.sum()), n_specz=int(hs.sum()), methods={})
                for m in methods:
                    r = dict(vs_label=organizer_point_metrics(zl[s], ests[m][s]), vs_label_nmad=point_metrics(zl[s], ests[m][s])["sigma_nmad"])
                    if hs.sum() > 200:
                        r["vs_specz"] = organizer_point_metrics(zs[hs], ests[m][hs]); r["vs_specz_nmad"] = point_metrics(zs[hs], ests[m][hs])["sigma_nmad"]
                    results[key][bkey]["methods"][m] = r
                    v = r["vs_label"]
                    line = (f"[{key}] {bkey} (n={s.sum()}) {m:>10s}: vs label  mean={v['mean']:+.4f} std={v['std']:.4f} "
                            f"outl3s={100*v['outlier_rate']:.1f}% abs>0.2={100*v['abs_outlier_rate']:.1f}% (sNMAD {r['vs_label_nmad']:.4f})")
                    if "vs_specz" in r:
                        w = r["vs_specz"]
                        line += (f" | vs spec-z ({hs.sum()}) mean={w['mean']:+.4f} std={w['std']:.4f} outl3s={100*w['outlier_rate']:.1f}% "
                                 f"abs>0.2={100*w['abs_outlier_rate']:.1f}% (sNMAD {r['vs_specz_nmad']:.4f})")
                    log(line, logf)
    out = run / f"point_estimates_{args.variant}.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    log(f"-> {out}", logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
