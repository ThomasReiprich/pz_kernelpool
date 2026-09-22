#!/usr/bin/env python3
"""PZ pilot: FlexZBoost photo-z with a proper held-out validation split.

What it does, for one (taskset, simulation, scenario):
  1. reads the challenge *training* catalogue and splits it at random into a
     fit part and a held-out validation part (default 80/20, fixed seed);
  2. trains FlexZBoost (RAIL) on the fit part only;
  3. estimates p(z) for the validation part and computes point-estimate and
     PDF-calibration metrics against the true redshifts there;
  4. estimates p(z) for the challenge *test* catalogue (no truth) and writes it
     in the challenge submission format (qp ensemble + object_id ancil);
  5. writes metrics.json, a few diagnostic plots and a run log.

Validation on the held-out part of the training set is only representative of
the population that the training set represents; for the PZ task set 1 that is
i < 23 with representative (not spectroscopically selected) labels.

Usage (from the DESC_NZ_Challenge folder, env desc-pz activated):
  python pilot/pz_pilot.py --taskset 1 --sim cardinal --scenario 1yr --nmax 5000   # quick test
  python pilot/pz_pilot.py --taskset 1 --sim cardinal --scenario 1yr               # full run
  python pilot/pz_pilot.py --taskset 1 --sim cardinal --scenario 1yr --estimator knn   # other algorithm, same split

Outputs go to pilot/runs/<taskset>_<sim>_<scenario>[_nmaxN]/.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

# RAIL / qp imports are deliberately not wrapped in try/except: if they fail
# we want the real error, not a fallback.
import qp
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

ESTIMATORS = {"flexzboost": None, "knn": None, "pzflow": None, "gpz": None}   # filled lazily in build_stages()


def fill_nondetections(path: Path, mag_limits: dict, outdir: Path) -> Path:
    """Write a copy of an HDF5 table with NaN magnitudes replaced by the band limit
    (and the error set to 1.0). Needed for estimators that do not treat NaN as a
    non-detection themselves (PZFlow only recognises 99.0)."""
    t = tables_io.read(str(path))
    out = {k: np.array(v) for k, v in t.items()}
    for col, lim in mag_limits.items():
        if col not in out:
            continue
        m = ~np.isfinite(out[col])
        out[col][m] = lim
        ecol = col + "_err"
        if ecol in out:
            out[ecol][m] = 1.0
    # RAIL's PZFlowEstimator selects the redshift column from the input table even when
    # estimating (its values are not used); the challenge test files have no such column.
    if "redshift" not in out:
        out["redshift"] = np.zeros(len(next(iter(out.values()))), dtype=float)
    new = outdir / (path.stem + "_filled.hdf5")
    tables_io.write(out, str(new))
    return new


def build_stages(name: str, tag: str, model_file, args):
    """Return (informer, estimator_kwargs, EstimatorClass) for the chosen algorithm.

    Every algorithm gets the same grid (ZMIN, ZMAX, NZBINS), the same non-detection
    handling (NaN -> band magnitude limit from the catalog config) and the same point
    estimates, so their outputs are directly comparable and downstream scripts
    (plot_nz_true_vs_estimates.py, calibrate_pdfs.py) can read them unchanged.
    """
    common = dict(hdf5_groupname="", zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS, nondetect_val=np.nan)
    if name == "flexzboost":
        informer = FlexZBoostInformer.make_stage(
            name=f"inform_{tag}", model=str(model_file), **common,
            trainfrac=args.trainfrac,
            bumpmin=args.bump_min, bumpmax=args.bump_max, nbump=args.n_bump,
            sharpmin=args.sharp_min, sharpmax=args.sharp_max, nsharp=args.n_sharp,
            max_basis=args.max_basis,
        )
        est_kwargs = dict(model=str(model_file), **common, chunk_size=20000,
                          calculated_point_estimates=["zmode", "zmean"], qp_representation="interp")
        return informer, est_kwargs, FlexZBoostEstimator
    if name == "knn":
        # k-nearest-neighbour in colour space (RAIL's KNearNeighEstimator, pz-rail-sklearn):
        # p(z) = Gaussian mixture from the redshifts of the k nearest training galaxies,
        # with k and the kernel width chosen on an internal validation fraction.
        from rail.estimation.algos.k_nearneigh import KNearNeighEstimator, KNearNeighInformer
        informer = KNearNeighInformer.make_stage(name=f"inform_{tag}", model=str(model_file), **common)
        # zmean is not requested here: qp's mixture-model .mean() fails with the current
        # qp/scipy versions; as_interp() computes it on the grid instead.
        est_kwargs = dict(model=str(model_file), **common, chunk_size=20000,
                          calculated_point_estimates=["zmode"])
        return informer, est_kwargs, KNearNeighEstimator
    if name == "pzflow":
        # Normalising flow (pzflow, Crenshaw et al. 2024): models the joint density of
        # (z, magnitudes) and returns p(z | magnitudes) on the grid. Smooth PDFs, no bump
        # removal. Uses all bands of the catalog config (LSST + Roman where present).
        from rail.estimation.algos.pzflow_nf import PZFlowEstimator, PZFlowInformer
        probe = PZFlowInformer.make_stage(name=f"probe_{tag}", hdf5_groupname="")
        cols = list(probe.config["mag_limits"].keys())
        flow_common = dict(hdf5_groupname="", zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS, column_names=cols)
        informer = PZFlowInformer.make_stage(name=f"inform_{tag}", model=str(model_file), **flow_common,
                                             n_training_epochs=args.flow_epochs, seed=args.seed)
        # small chunks: the posterior evaluation holds (chunk x grid) samples in memory (~6 GB for 5000 objects)
        est_kwargs = dict(model=str(model_file), **flow_common, chunk_size=1000)
        return informer, est_kwargs, PZFlowEstimator
    if name == "gpz":
        # Sparse Gaussian process with heteroscedastic noise (GPz, Almosallam et al. 2016):
        # one Gaussian per object whose width grows away from the training data. Inputs are
        # the magnitudes and log errors; NaN handled internally.
        from rail.estimation.algos.gpz import GPzEstimator, GPzInformer
        informer = GPzInformer.make_stage(name=f"inform_{tag}", model=str(model_file), hdf5_groupname="",
                                          nondetect_val=np.nan, n_basis=args.gpz_basis, max_iter=args.gpz_iter,
                                          seed=args.seed)
        est_kwargs = dict(model=str(model_file), **common, chunk_size=20000)
        return informer, est_kwargs, GPzEstimator
    raise ValueError(f"unknown estimator {name!r}; choose from {list(ESTIMATORS)}")


def as_interp(ens: qp.Ensemble, logf=None, what: str = "") -> qp.Ensemble:
    """Convert any qp representation to the common z grid (keeps the ancillary table).
    Anything worth knowing (replaced PDFs) is written to the run log via log()."""
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    if ens.gen_class.name != "interp":
        new = ens.convert_to(qp.interp_gen, xvals=zgrid)
        if ens.ancil is not None:
            new.set_ancil(dict(ens.ancil))
        ens = new
    anc = dict(ens.ancil) if ens.ancil is not None else {}
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    # Guard against unusable PDFs (pzflow can return all-NaN rows for objects with extreme
    # colours; a zero row would break normalisation). Such objects get the sample-average
    # PDF, i.e. the least informative statement consistent with the population, and the
    # count is printed so it can be checked in the log.
    with np.errstate(invalid="ignore"):
        pdfs = np.asarray(ens.pdf(zgrid))
    bad = ~np.all(np.isfinite(pdfs), axis=1) | (np.nansum(pdfs, axis=1) <= 0)
    if bad.any():
        mean_pdf = np.nanmean(pdfs[~bad], axis=0)
        pdfs = pdfs.copy(); pdfs[bad] = mean_pdf
        ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pdfs))
        anc.pop("zmode", None); anc.pop("zmean", None)
        log(f"NOTE ({what}): {bad.sum()} of {len(bad)} PDFs were non-finite or empty and were replaced by the sample-average PDF", logf)
        NOTES.append(dict(what=what, n_replaced=int(bad.sum()), n_total=int(len(bad))))
    if "zmean" not in anc or "zmode" not in anc:
        norm = trap(pdfs, zgrid, axis=1)
        anc["zmode"] = zgrid[np.argmax(pdfs, axis=1)][:, None]
        anc["zmean"] = (trap(pdfs * zgrid, zgrid, axis=1) / norm)[:, None]
    ens.set_ancil(anc)
    return ens

# Redshift grid. Default 0-3 with 301 points (step 0.01) = RAIL's shared defaults and the organizers' example, inherited
# by every stage. Grid scan (2026-09-22, ours): the environment variables PZ_ZMAX and PZ_NZBINS override the upper edge
# and the number of points for EVERY script that imports these constants (training, pool, apply, point study), so one
# configuration is selected per shell; loaders assert that a file's grid matches, which catches a missing override.
import os as _os
ZMIN, ZMAX, NZBINS = 0.0, float(_os.environ.get("PZ_ZMAX", 3.0)), int(_os.environ.get("PZ_NZBINS", 301))
NOTES: list = []                             # anomalies recorded by as_interp(), written to metrics.json
OUTLIER_CUT = 0.15                           # |dz/(1+z)| above this counts as outlier


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return ru / 1e9 if sys.platform == "darwin" else ru / 1e6


def split_training(train_file: Path, outdir: Path, frac_valid: float, seed: int, nmax: int | None):
    """Write fit/valid HDF5 files; return (fit_path, valid_path, valid_table)."""
    data = tables_io.read(str(train_file))
    n = len(data["object_id"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    if nmax is not None:
        idx = idx[:nmax]
    n_valid = int(round(frac_valid * len(idx)))
    valid_idx, fit_idx = np.sort(idx[:n_valid]), np.sort(idx[n_valid:])
    fit = {k: v[fit_idx] for k, v in data.items()}
    valid = {k: v[valid_idx] for k, v in data.items()}
    fit_path, valid_path = outdir / "split_fit.hdf5", outdir / "split_valid.hdf5"
    tables_io.write(fit, str(fit_path))
    tables_io.write(valid, str(valid_path))
    # fit and valid are disjoint by construction (disjoint row indices). object_id is
    # NOT used as the key: in the Flagship files it is stored as float64 and the
    # ~5.8e16 values lose precision, so ~0.4 % of rows share an id with a different galaxy.
    assert len(np.intersect1d(fit_idx, valid_idx)) == 0
    n_unique = len(np.unique(data["object_id"]))
    if n_unique != n:
        print(f"note: object_id is not unique in {train_file.name} ({n} rows, {n_unique} distinct ids); rows are split by index")
    return fit_path, valid_path, fit, valid


def point_metrics(z_true: np.ndarray, z_point: np.ndarray) -> dict:
    dz = (z_point - z_true) / (1.0 + z_true)
    ok = np.isfinite(dz)
    dz = dz[ok]
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    return dict(
        n=int(ok.sum()),
        n_nonfinite=int((~ok).sum()),
        bias_median=float(np.median(dz)),
        bias_mean=float(np.mean(dz)),
        sigma_nmad=float(nmad),
        sigma_std=float(np.std(dz)),
        outlier_frac=float(np.mean(np.abs(dz) > OUTLIER_CUT)),
        # the PZ challenge's own definition (docs, "Metrics for per-object point estimates"):
        # outliers are |dz| > max(0.06, 3 sigma_IQR), sigma_IQR = IQR / 1.349
        outlier_frac_pz=float(np.mean(np.abs(dz) > max(0.06, 3 * (np.subtract(*np.percentile(dz, [75, 25])) / 1.349)))) if len(dz) else float("nan"),
    )


def binned_metrics(z_true, z_point, by, edges, label):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (by >= lo) & (by < hi)
        if m.sum() < 20:
            continue
        d = point_metrics(z_true[m], z_point[m])
        d.update({label + "_lo": float(lo), label + "_hi": float(hi)})
        out.append(d)
    return out


def pit_metrics(ens: qp.Ensemble, z_true: np.ndarray) -> tuple[dict, np.ndarray]:
    """PIT = CDF of each object's p(z) evaluated at its true z. Uniform if calibrated."""
    pit = np.squeeze(ens.cdf(z_true[:, None]))
    pit = np.clip(pit, 0.0, 1.0)
    hist, _ = np.histogram(pit, bins=20, range=(0, 1))
    frac = hist / hist.sum()
    # Kolmogorov-Smirnov distance to the uniform distribution
    srt = np.sort(pit)
    ecdf = np.arange(1, len(srt) + 1) / len(srt)
    ks = float(np.max(np.abs(ecdf - srt)))
    return dict(
        pit_frac_below_0p05=float(np.mean(pit < 0.05)),
        pit_frac_above_0p95=float(np.mean(pit > 0.95)),
        pit_frac_extreme_expected=0.10,
        pit_ks_to_uniform=ks,
        pit_hist_20bins=frac.round(4).tolist(),
    ), pit


def make_plots(outdir: Path, z_true, z_point, mag_i, pit, key: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.hexbin(z_true, z_point, gridsize=80, extent=(0, 2.5, 0, 2.5), bins="log", mincnt=1)
    ax.plot([0, 2.5], [0, 2.5], "r-", lw=1)
    for s in (+1, -1):
        ax.plot([0, 2.5], [s * OUTLIER_CUT, 2.5 + s * OUTLIER_CUT * 3.5], "r--", lw=0.7)
    ax.set_xlabel("true z"); ax.set_ylabel("zmode"); ax.set_title(f"{key}: validation split")
    ax = axes[1]
    dz = (z_point - z_true) / (1 + z_true)
    ax.hexbin(mag_i, dz, gridsize=80, extent=(17, 26, -0.5, 0.5), bins="log", mincnt=1)
    ax.axhline(0, color="r", lw=1); ax.axhline(OUTLIER_CUT, color="r", ls="--", lw=0.7); ax.axhline(-OUTLIER_CUT, color="r", ls="--", lw=0.7)
    ax.set_xlabel("i (LSST)"); ax.set_ylabel("(zmode - z)/(1+z)")
    ax = axes[2]
    ax.hist(pit, bins=20, range=(0, 1), density=True, histtype="stepfilled", alpha=0.6)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xlabel("PIT"); ax.set_ylabel("density"); ax.set_title("PIT (flat = calibrated)")
    fig.tight_layout()
    fig.savefig(outdir / "diagnostics.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taskset", type=int, default=1)
    ap.add_argument("--sim", default="cardinal", choices=["cardinal", "flagship"])
    ap.add_argument("--scenario", default="1yr")
    ap.add_argument("--base", default=None, help="DESC_NZ_Challenge folder (default: parent of this script's folder)")
    ap.add_argument("--data-subdir", default="data/pz/public", help="where the pz_challenge_*.hdf5 files are (relative to --base)")
    ap.add_argument("--frac-valid", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nmax", type=int, default=None, help="use only this many training objects (quick test)")
    ap.add_argument("--skip-test", action="store_true", help="do not run on the challenge test catalogue")
    ap.add_argument("--label", default="", help="suffix for the run directory, so that variants do not overwrite each other")
    ap.add_argument("--estimator", default="flexzboost", choices=list(ESTIMATORS), help="photo-z algorithm (RAIL stage)")
    ap.add_argument("--bump-min", type=float, default=0.02)
    ap.add_argument("--bump-max", type=float, default=0.35)
    ap.add_argument("--n-bump", type=int, default=20)
    ap.add_argument("--sharp-min", type=float, default=0.7)
    ap.add_argument("--sharp-max", type=float, default=2.1)
    ap.add_argument("--n-sharp", type=int, default=15)
    ap.add_argument("--max-basis", type=int, default=35)
    ap.add_argument("--trainfrac", type=float, default=0.75, help="FlexZBoost-internal fraction used for fitting vs. tuning bump/sharpen")
    ap.add_argument("--flow-epochs", type=int, default=50, help="pzflow training epochs")
    ap.add_argument("--gpz-basis", type=int, default=50, help="GPz number of basis functions")
    ap.add_argument("--gpz-iter", type=int, default=200, help="GPz max optimisation iterations")
    args = ap.parse_args()

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    key = f"taskset_{args.taskset}_{args.sim}_{args.scenario}"
    tag = key + (f"_{args.estimator}" if args.estimator != "flexzboost" else "") + (f"_nmax{args.nmax}" if args.nmax else "") + (f"_{args.label}" if args.label else "")
    outdir = base / "pilot" / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    # RAIL writes its intermediate products (output_<stage>.hdf5, inprogress_*) into the
    # current directory; run from the output directory so they land there.
    import os
    os.chdir(outdir)
    logf = open(outdir / "run.log", "a")
    log(f"=== pz_pilot {key}  base={base}", logf)
    log(f"args: {vars(args)}", logf)

    datadir = base / args.data_subdir
    train_file = datadir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_training_{args.scenario}.hdf5"
    test_file = datadir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_test_{args.scenario}.hdf5"
    for f in (train_file, test_file):
        if not f.exists():
            raise FileNotFoundError(f)

    # --- band configuration: tells RAIL which columns are magnitudes/errors
    catalog_utils.clear()
    catalog_utils.load_yaml(str(base / "repos" / "nz_data_challenge" / "tests" / "catalogs.yaml"))
    catalog_utils.apply(f"{args.sim}_roman_rubin")
    log(f"catalog tag: {args.sim}_roman_rubin", logf)

    # --- 1. split
    t0 = time.time()
    fit_path, valid_path, fit, valid = split_training(train_file, outdir, args.frac_valid, args.seed, args.nmax)
    log(f"split: fit={len(fit['object_id'])} valid={len(valid['object_id'])}  ({time.time()-t0:.1f}s)", logf)
    log("columns: " + ", ".join(sorted(fit.keys())), logf)

    # --- 2. train
    model_file = outdir / f"{args.estimator}_model.pkl"
    informer, est_kwargs, EstimatorClass = build_stages(args.estimator, tag, model_file, args)
    log("informer config: " + json.dumps({k: v for k, v in informer.config.to_dict().items()
                                         if k in ("bands", "err_bands", "ref_band", "column_names", "mag_limits", "nondetect_val", "zmin", "zmax", "nzbins", "trainfrac", "max_basis", "nbump", "nsharp", "nneigh_min", "nneigh_max", "sigma_grid_min", "sigma_grid_max", "n_training_epochs", "n_basis", "max_iter", "gpz_method")},
                                        default=str), logf)
    if args.estimator == "pzflow":
        # pzflow does not treat NaN as a non-detection: feed it copies with NaN -> band limit
        lims = dict(informer.config["mag_limits"])
        fit_path, valid_path = fill_nondetections(fit_path, lims, outdir), fill_nondetections(valid_path, lims, outdir)
        test_file_est = fill_nondetections(test_file, lims, outdir) if not args.skip_test else test_file
        log("non-detections filled with band limits for pzflow", logf)
    else:
        test_file_est = test_file
    t0 = time.time()
    informer.inform(TableHandle(f"fit_{tag}", path=str(fit_path)))
    log(f"inform done ({time.time()-t0:.1f}s, peak RSS {peak_rss_gb():.2f} GB)", logf)

    # --- 3. estimate on the validation split and score
    t0 = time.time()
    est_valid = EstimatorClass.make_stage(name=f"estimate_valid_{tag}", **est_kwargs)
    valid_handle = est_valid.estimate(TableHandle(f"valid_{tag}", path=str(valid_path)))
    ens_valid = as_interp(qp.read(valid_handle.path), logf, "validation")
    log(f"estimate valid done ({time.time()-t0:.1f}s)", logf)

    z_true = np.asarray(valid["redshift"], dtype=float)
    z_mode = np.squeeze(ens_valid.ancil["zmode"])
    z_mean = np.squeeze(ens_valid.ancil["zmean"])
    assert len(z_mode) == len(z_true), "row-order mismatch between estimates and validation truth"
    mag_i = np.asarray(valid["mag_i_lsst"], dtype=float)

    metrics = dict(
        key=key, n_fit=int(len(fit["object_id"])), n_valid=int(len(z_true)),
        point_zmode=point_metrics(z_true, z_mode),
        point_zmean=point_metrics(z_true, z_mean),
        zmode_vs_imag=binned_metrics(z_true, z_mode, mag_i, np.arange(18, 26.5, 1.0), "i"),
        zmode_vs_ztrue=binned_metrics(z_true, z_mode, z_true, np.arange(0, 2.6, 0.25), "z"),
    )
    pit_m, pit = pit_metrics(ens_valid, z_true)
    metrics.update(pit_m)
    pm = metrics["point_zmode"]
    log(f"VALIDATION (zmode): bias_med={pm['bias_median']:+.4f}  sigma_NMAD={pm['sigma_nmad']:.4f}  "
        f"outliers(>{OUTLIER_CUT})={100*pm['outlier_frac']:.2f}%  PIT extremes={100*(pit_m['pit_frac_below_0p05']+pit_m['pit_frac_above_0p95']):.1f}% (10% if calibrated)  KS={pit_m['pit_ks_to_uniform']:.3f}", logf)
    make_plots(outdir, z_true, z_mode, mag_i, pit, key)

    # keep the validation ensemble with truth attached, for later comparisons
    ens_valid.set_ancil(dict(ens_valid.ancil, object_id=np.asarray(valid["object_id"]), z_true=z_true))
    ens_valid.write_to(str(outdir / "pz_valid_with_truth.hdf5"))

    # --- 4. challenge test catalogue -> submission-format file
    if not args.skip_test:
        t0 = time.time()
        est_test = EstimatorClass.make_stage(name=f"estimate_test_{tag}", **est_kwargs)
        test_handle = est_test.estimate(TableHandle(f"test_{tag}", path=str(test_file_est)))
        ens_test = as_interp(qp.read(test_handle.path), logf, "test")
        test_ids = tables_io.read(str(test_file))["object_id"]
        assert ens_test.npdf == len(test_ids), "test estimate length != test catalogue length"
        ens_test.set_ancil(dict(ens_test.ancil, object_id=np.asarray(test_ids).astype(int)))
        sub_dir = outdir / "submission"; sub_dir.mkdir(exist_ok=True)
        sub_file = sub_dir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_pz_estimate_{args.scenario}.hdf5"
        ens_test.write_to(str(sub_file))
        metrics["n_test"] = int(ens_test.npdf)
        log(f"test estimate written: {sub_file} ({time.time()-t0:.1f}s)", logf)

    metrics["peak_rss_gb"] = peak_rss_gb()
    metrics["notes"] = list(NOTES)
    if NOTES:
        log(f"{len(NOTES)} NOTE line(s) above -- see metrics.json['notes']", logf)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"metrics -> {outdir/'metrics.json'};  plots -> {outdir/'diagnostics.png'}", logf)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
