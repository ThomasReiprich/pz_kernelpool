#!/usr/bin/env python3
"""PZ task set 2 experiment: spectroscopically selected training sample.

Question: how much does a simple importance reweighting of the selected
training sample (the "targeted intervention" for a training set that is
brighter and bluer than the test sample) buy for a per-object photo-z, and
where does it stop helping?

Setup, for one (simulation, scenario):
  training  = PZ task-set-2 training file (100k rows; every row carries at
              least one spectroscopic-survey selection flag, i < ~25, 0.1 %
              fainter than i = 24)
  target    = PZ task-set-2 test photometry (20k rows, 59 % fainter than i = 24;
              no labels) -- used ONLY to estimate where the test sample lives in
              colour/magnitude space
  validation= NZ task-set-2 ddf_00 file of the same simulation/scenario: WFD-depth
              photometry of a reference region with a many-band photo-z
              (`redshift_manyband`) for every object and a spectroscopic
              redshift for ~25 %. Its magnitude distribution is the same as
              that of the PZ test sample. Rows that are also PZ training or
              test rows (same ra/dec) are excluded; a random subset of
              --nvalid rows is used.
              Caveat: the many-band redshifts carry deliberate errors (on the
              spectroscopic subset ~2 % have |dz|/(1+z) > 0.15), so absolute
              numbers against them are slightly pessimistic; the comparison
              between variants is unaffected. Metrics against the spectroscopic
              subset are reported too, but that subset is itself selected.

Estimators (--estimator, same settings as pz_pilot.py): flexzboost (default),
gpz, pzflow, knn. Variants:
  unweighted : train on the selected sample as is
  weighted   : per-object weights w = n_target(x) / n_train(x) from k-nearest-
               neighbour density estimates in the standardised space of
               (i, colours) -- Lima et al. 2008, MNRAS 390, 118. For
               FlexZBoost they are passed to XGBoost as sample weights (RAIL
               `use_weights`); the other estimators have no weight input, so
               the training sample is importance-resampled instead (drawn
               with replacement with probability proportional to w, same
               size as the original; a standard equivalent, at the cost of
               duplicated rows).

Reported per variant, on the representative validation set: point-estimate
metrics (all / i <= 24 / i > 24 / per magnitude bin), PIT, mean log p(z_true),
and, for the NZ task-set-2 tomographic bins, the bin-assignment accuracy and
the difference between the stacked p(z) and the true n(z) mean/width per bin.
A coverage diagnostic says which fraction of validation objects has no
training neighbour nearby at all (reweighting cannot fix that).

Usage (from the DESC_NZ_Challenge folder, env desc-pz activated):
  python pilot/pz_ts2_experiment.py --sim cardinal --scenario 1yr --nmax 8000 --nvalid 5000   # quick test
  python pilot/pz_ts2_experiment.py --sim cardinal --scenario 1yr                             # full run

Outputs go to pilot/runs/ts2exp_<sim>_<scenario>[_<estimator>][_nmaxN][_label]/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qp
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

from pz_pilot import ZMIN, ZMAX, NZBINS, OUTLIER_CUT, ESTIMATORS, as_interp, binned_metrics, build_stages, fill_nondetections, log, peak_rss_gb, pit_metrics, point_metrics
from calibrate_pdfs import nz_check, normalise, tomo_rms, trapezoid

TS2_EDGES = np.array([0.0, 0.42, 0.64, 0.87, 1.20, 2.5])   # NZ task-set-2/3 tomographic bins
ZGRID = np.linspace(ZMIN, ZMAX, NZBINS)
FAINT_CUT = 24.0


# ----------------------------------------------------------------- helpers
def pos_key(t) -> np.ndarray:
    """Position key used throughout the project for row matching."""
    return np.round(np.asarray(t["ra"]), 7) * 1e5 + np.round(np.asarray(t["dec"]), 7)


def band_columns(cfg) -> tuple[list[str], list[str], dict]:
    """Magnitude columns, error columns and NaN-replacement limits from the RAIL catalog config."""
    return list(cfg["bands"]), list(cfg["err_bands"]), dict(cfg["mag_limits"])


def feature_matrix(t, bands: list[str], mag_limits: dict, ref_band: str) -> np.ndarray:
    """(ref mag, adjacent-band colours), non-detections replaced by the band limit -- same
    inputs FlexZBoost uses (make_color_data), so the density ratio lives in the same space."""
    mags = []
    for b in bands:
        m = np.array(t[b], dtype=float)
        m[~np.isfinite(m)] = mag_limits[b]
        mags.append(m)
    mags = np.stack(mags, axis=1)
    ref = mags[:, bands.index(ref_band)]
    cols = mags[:, :-1] - mags[:, 1:]
    return np.column_stack([ref, cols])


def knn_density_ratio(x_train: np.ndarray, x_target: np.ndarray, k: int, wmax: float):
    """w_i = [n_target(x_i)/N_target] / [k/N_train], with n_target counted inside the radius
    that encloses the k nearest *training* neighbours of x_i (Lima et al. 2008).
    Returns (weights normalised to mean 1, per-object radius, per-object target count)."""
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    xt, xg = x_train / scale, x_target / scale
    tree_tr, tree_tg = cKDTree(xt), cKDTree(xg)
    d, _ = tree_tr.query(xt, k=k + 1)          # column 0 is the object itself
    r = d[:, -1]
    n_tg = tree_tg.query_ball_point(xt, r, return_length=True).astype(float)
    w = (n_tg / len(xg)) / (k / len(xt))
    w = np.minimum(w, wmax)
    w /= w.mean()
    return w, r, n_tg, scale


def coverage(x_train: np.ndarray, x_other: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, float]:
    """Distance from each 'other' object to its nearest training object, and the 99th
    percentile of the training sample's own nearest-neighbour distance (a scale for
    'is there a training object nearby at all')."""
    tree = cKDTree(x_train / scale)
    d_self, _ = tree.query(x_train / scale, k=2)
    ref = float(np.percentile(d_self[:, 1], 99))
    d_other, _ = tree.query(x_other / scale, k=1)
    return d_other, ref


def loglik(p, x, z_true, floor=1e-6):
    val = np.array([np.interp(zi, x, pi) for zi, pi in zip(z_true, p)])
    return float(np.mean(np.log(np.maximum(val, floor))))


def subset_metrics(z_true, z_mode, mag_i, mask) -> dict:
    d = point_metrics(z_true[mask], z_mode[mask])
    d["frac_of_sample"] = float(mask.mean())
    d["i_median"] = float(np.nanmedian(mag_i[mask]))
    return d


def build_validation(nz_file: Path, exclude_keys: np.ndarray, nvalid: int | None, seed: int, outdir: Path,
                     ref_col: str, imax: float):
    t = tables_io.read(str(nz_file))
    n = len(t["ra"])
    keys = pos_key(t)
    overlap = np.isin(keys, exclude_keys)
    # the PZ test sample stops at i = imax (25.5 for Cardinal 1yr); ddf_00 goes ~0.5 mag deeper
    too_faint = ~(np.asarray(t[ref_col], float) <= imax)
    keep = np.flatnonzero(~overlap & ~too_faint)
    rng = np.random.default_rng(seed)
    if nvalid is not None and nvalid < len(keep):
        keep = np.sort(rng.choice(keep, nvalid, replace=False))
    sub = {k: np.asarray(v)[keep] for k, v in t.items()}
    path = outdir / "valid_representative.hdf5"
    tables_io.write(sub, str(path))
    info = dict(n_file=int(n), n_overlap_with_pz=int(overlap.sum()), n_fainter_than_test=int(too_faint.sum()),
                i_max=float(imax), n_used=int(len(keep)),
                n_with_specz=int(np.isfinite(sub["redshift"]).sum()),
                n_with_manyband=int(np.isfinite(sub["redshift_manyband"]).sum()))
    return path, sub, info


def make_plots(outdir: Path, variant: str, z_true, z_point, mag_i, pit) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.hexbin(z_true, z_point, gridsize=80, extent=(0, 2.5, 0, 2.5), bins="log", mincnt=1)
    ax.plot([0, 2.5], [0, 2.5], "r-", lw=1)
    ax.set_xlabel("many-band z (ddf_00)"); ax.set_ylabel("zmode"); ax.set_title(f"{variant}: representative validation")
    ax = axes[1]
    dz = (z_point - z_true) / (1 + z_true)
    ax.hexbin(mag_i, dz, gridsize=80, extent=(17, 26, -0.5, 0.5), bins="log", mincnt=1)
    ax.axhline(0, color="r", lw=1); ax.axhline(OUTLIER_CUT, color="r", ls="--", lw=0.7); ax.axhline(-OUTLIER_CUT, color="r", ls="--", lw=0.7)
    ax.axvline(FAINT_CUT, color="k", ls=":", lw=0.8)
    ax.set_xlabel("i (LSST)"); ax.set_ylabel("(zmode - z)/(1+z)")
    ax = axes[2]
    ax.hist(pit, bins=20, range=(0, 1), density=True, histtype="stepfilled", alpha=0.6)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xlabel("PIT"); ax.set_ylabel("density"); ax.set_title("PIT vs many-band z")
    fig.tight_layout()
    fig.savefig(outdir / f"diagnostics_{variant}.png", dpi=130)
    plt.close(fig)


def plot_weights(outdir: Path, i_train, w, i_target, i_valid, d_valid, ref_dist) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = np.arange(17, 26.01, 0.25)
    ax = axes[0]
    ax.hist(i_train, bins, density=True, histtype="step", label="training (selected)")
    ax.hist(i_train, bins, weights=w, density=True, histtype="step", label="training, reweighted")
    ax.hist(i_target, bins, density=True, histtype="step", label="target (PZ test photometry)")
    ax.hist(i_valid, bins, density=True, histtype="step", ls="--", label="validation (ddf_00)")
    ax.set_xlabel("i (LSST)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.hexbin(i_train, np.log10(np.maximum(w, 1e-3)), gridsize=60, extent=(17, 26, -3, 2), bins="log", mincnt=1)
    ax.set_xlabel("i (training)"); ax.set_ylabel("log10 weight")
    ax = axes[2]
    ax.hexbin(i_valid, d_valid, gridsize=60, extent=(17, 26, 0, max(1.0, float(np.percentile(d_valid, 99.5)))), bins="log", mincnt=1)
    ax.axhline(ref_dist, color="r", lw=1, label="99th pct of training self-NN distance")
    ax.set_xlabel("i (validation)"); ax.set_ylabel("distance to nearest training object"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "weights_and_coverage.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="cardinal", choices=["cardinal", "flagship"])
    ap.add_argument("--scenario", default="1yr")
    ap.add_argument("--base", default=None, help="DESC_NZ_Challenge folder (default: parent of this script's folder)")
    ap.add_argument("--data-subdir", default="data/pz/public")
    ap.add_argument("--nz-data-subdir", default="data/nz/public")
    ap.add_argument("--nmax", type=int, default=None, help="use only this many training objects (quick test)")
    ap.add_argument("--nvalid", type=int, default=50000, help="size of the representative validation subset drawn from ddf_00")
    ap.add_argument("--valid-imax", type=float, default=None, help="faint i limit for the validation set (default: faintest object in the PZ test file)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=100, help="neighbours for the density-ratio weights")
    ap.add_argument("--wmax", type=float, default=50.0, help="cap on the raw density-ratio weight before normalisation")
    ap.add_argument("--variants", default="unweighted,weighted")
    ap.add_argument("--estimator", default="flexzboost", choices=list(ESTIMATORS))
    ap.add_argument("--flow-epochs", type=int, default=50)
    ap.add_argument("--gpz-basis", type=int, default=50)
    ap.add_argument("--gpz-iter", type=int, default=200)
    ap.add_argument("--skip-test", action="store_true", help="do not write submission files for the PZ test catalogue")
    ap.add_argument("--label", default="")
    # FlexZBoost settings, same defaults as pz_pilot.py
    ap.add_argument("--trainfrac", type=float, default=0.75)
    ap.add_argument("--bump-min", type=float, default=0.02); ap.add_argument("--bump-max", type=float, default=0.35); ap.add_argument("--n-bump", type=int, default=20)
    ap.add_argument("--sharp-min", type=float, default=0.7); ap.add_argument("--sharp-max", type=float, default=2.1); ap.add_argument("--n-sharp", type=int, default=15)
    ap.add_argument("--max-basis", type=int, default=35)
    args = ap.parse_args()

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    key = f"ts2exp_{args.sim}_{args.scenario}"
    tag = key + (f"_{args.estimator}" if args.estimator != "flexzboost" else "") + (f"_nmax{args.nmax}" if args.nmax else "") + (f"_{args.label}" if args.label else "")
    outdir = base / "pilot" / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    logf = open(outdir / "run.log", "a")
    log(f"=== pz_ts2_experiment {key}  base={base}", logf)
    log(f"args: {vars(args)}", logf)

    train_file = base / args.data_subdir / f"pz_challenge_taskset_2_{args.sim}_training_{args.scenario}.hdf5"
    test_file = base / args.data_subdir / f"pz_challenge_taskset_2_{args.sim}_test_{args.scenario}.hdf5"
    nz_file = base / args.nz_data_subdir / f"nz_challenge_taskset_2_{args.sim}_{args.scenario}_ddf_00.hdf5"
    for f in (train_file, test_file, nz_file):
        if not f.exists():
            raise FileNotFoundError(f)

    catalog_utils.clear()
    catalog_utils.load_yaml(str(base / "repos" / "nz_data_challenge" / "tests" / "catalogs.yaml"))
    catalog_utils.apply(f"{args.sim}_roman_rubin")
    common = dict(hdf5_groupname="", zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS, nondetect_val=np.nan)
    fzb_kwargs = dict(trainfrac=args.trainfrac, bumpmin=args.bump_min, bumpmax=args.bump_max, nbump=args.n_bump,
                      sharpmin=args.sharp_min, sharpmax=args.sharp_max, nsharp=args.n_sharp, max_basis=args.max_basis)
    probe = FlexZBoostInformer.make_stage(name="probe", **common)
    bands, err_bands, mag_limits = band_columns(probe.config)
    ref_band = probe.config["ref_band"]
    log(f"bands: {bands}  ref: {ref_band}", logf)

    # --- training sample (selected), optional subset
    train = tables_io.read(str(train_file))
    n_train_file = len(train["ra"])
    if args.nmax is not None and args.nmax < n_train_file:
        idx = np.sort(np.random.default_rng(args.seed).choice(n_train_file, args.nmax, replace=False))
        train = {k: np.asarray(v)[idx] for k, v in train.items()}
    n_train = len(train["ra"])
    flag_cols = [c for c in train if c not in bands + err_bands and c not in ("ra", "dec", "object_id", "redshift")
                 and set(np.unique(np.asarray(train[c]).astype(float))) <= {0.0, 1.0}]
    flagged_any = np.zeros(n_train, bool)
    for c in flag_cols:
        flagged_any |= np.asarray(train[c]) > 0
    log(f"training: {n_train} rows (file: {n_train_file}); selection flags {flag_cols}; rows with >=1 flag: {flagged_any.mean():.4f}", logf)
    i_train = np.asarray(train[f"mag_{ref_band}_lsst"] if f"mag_{ref_band}_lsst" in train else train[ref_band], float)
    log(f"training i quantiles 10/50/90: {np.nanpercentile(i_train, [10, 50, 90]).round(2)}; frac i>{FAINT_CUT}: {np.nanmean(i_train > FAINT_CUT):.4f}", logf)

    # --- target photometry (PZ test) and representative validation set (ddf_00)
    test = tables_io.read(str(test_file))
    i_test = np.asarray(test[ref_band], float)
    log(f"target (PZ test): {len(i_test)} rows; i quantiles {np.nanpercentile(i_test, [10, 50, 90]).round(2)}; frac i>{FAINT_CUT}: {np.nanmean(i_test > FAINT_CUT):.4f}", logf)
    exclude = np.concatenate([pos_key(tables_io.read(str(train_file))), pos_key(test)])
    imax = args.valid_imax if args.valid_imax is not None else float(np.nanmax(i_test))
    valid_path, valid, vinfo = build_validation(nz_file, exclude, args.nvalid, args.seed, outdir, ref_band, imax)
    log(f"validation (ddf_00): {vinfo}", logf)
    z_mb = np.asarray(valid["redshift_manyband"], float)
    z_sp = np.asarray(valid["redshift"], float)
    has_sp = np.isfinite(z_sp)
    i_valid = np.asarray(valid[ref_band], float)
    log(f"validation i quantiles {np.nanpercentile(i_valid, [10, 50, 90]).round(2)}; frac i>{FAINT_CUT}: {np.nanmean(i_valid > FAINT_CUT):.4f}", logf)
    # how good are the many-band labels where a spec-z exists (the label-noise reference)
    mb_vs_sp = point_metrics(z_sp[has_sp], z_mb[has_sp]) if has_sp.sum() > 20 else None
    if mb_vs_sp:
        log(f"many-band vs spec-z on the {has_sp.sum()} spec-z rows: sigma_NMAD={mb_vs_sp['sigma_nmad']:.4f} outliers={100*mb_vs_sp['outlier_frac']:.2f}% bias_med={mb_vs_sp['bias_median']:+.4f}", logf)

    # --- density-ratio weights and coverage
    x_train = feature_matrix(train, bands, mag_limits, ref_band)
    x_test = feature_matrix(test, bands, mag_limits, ref_band)
    x_valid = feature_matrix(valid, bands, mag_limits, ref_band)
    t0 = time.time()
    w, r_k, n_tg, scale = knn_density_ratio(x_train, x_test, args.k, args.wmax)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    d_valid, ref_dist = coverage(x_train, x_valid, scale)
    uncovered = d_valid > ref_dist
    log(f"weights: k={args.k} cap={args.wmax}  n_eff={n_eff:.0f} of {n_train}  frac(raw count=0)={np.mean(n_tg == 0):.3f}  "
        f"weight quantiles 10/50/90/99: {np.percentile(w, [10, 50, 90, 99]).round(3)}  ({time.time()-t0:.1f}s)", logf)
    log(f"coverage: {100*uncovered.mean():.1f}% of validation objects are farther from any training object than the 99th pct "
        f"of the training self-NN distance ({ref_dist:.3f}); by i: "
        + ", ".join(f"{lo:.0f}-{lo+1:.0f}: {100*uncovered[(i_valid >= lo) & (i_valid < lo + 1)].mean():.0f}%"
                    for lo in np.arange(18, 26) if ((i_valid >= lo) & (i_valid < lo + 1)).sum() > 20), logf)
    weighted_i_q = np.percentile(np.repeat(i_train, np.maximum(1, np.round(w * 10).astype(int))), [10, 50, 90])
    log(f"reweighted training i quantiles 10/50/90 (approx.): {weighted_i_q.round(2)}", logf)
    plot_weights(outdir, i_train, w, i_test, i_valid, d_valid, ref_dist)

    metrics = dict(key=key, estimator=args.estimator, n_train=n_train, target="pz_test", validation=vinfo, k=args.k, wmax=args.wmax, n_eff=n_eff,
                   frac_valid_uncovered=float(uncovered.mean()), ref_nn_distance=ref_dist,
                   frac_valid_uncovered_faint=float(uncovered[i_valid > FAINT_CUT].mean()) if (i_valid > FAINT_CUT).sum() else None,
                   manyband_vs_specz=mb_vs_sp, variants={})

    fit_path = outdir / "fit_with_weights.hdf5"
    tables_io.write(dict(train, weight=w), str(fit_path))
    test_ids = np.asarray(test["object_id"]).astype(int)
    test_file_est = test_file
    if args.estimator == "pzflow":
        # pzflow does not treat NaN as a non-detection and needs a redshift column present
        valid_path = fill_nondetections(valid_path, mag_limits, outdir)
        test_file_est = fill_nondetections(test_file, mag_limits, outdir)

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if variant not in ("unweighted", "weighted"):
            raise ValueError(f"unknown variant {variant!r}")
        vtag = f"{tag}_{variant}"
        model_file = outdir / f"{args.estimator}_model_{variant}.pkl"
        this_fit = fit_path
        if args.estimator == "flexzboost":
            informer = FlexZBoostInformer.make_stage(name=f"inform_{vtag}", model=str(model_file), **common, **fzb_kwargs,
                                                     use_weights=(variant == "weighted"), weights_column="weight")
            est_kwargs = dict(model=str(model_file), **common, chunk_size=20000,
                              calculated_point_estimates=["zmode", "zmean"], qp_representation="interp")
            EstimatorClass = FlexZBoostEstimator
        else:
            informer, est_kwargs, EstimatorClass = build_stages(args.estimator, vtag, model_file, args)
            if variant == "weighted":
                rng = np.random.default_rng(args.seed)
                pick = rng.choice(n_train, n_train, replace=True, p=w / w.sum())
                res = {k: np.asarray(v)[pick] for k, v in train.items()}
                this_fit = outdir / f"fit_resampled_{variant}.hdf5"
                tables_io.write(res, str(this_fit))
                log(f"[{variant}] {args.estimator} has no weight input: training sample importance-resampled "
                    f"({len(np.unique(pick))} distinct of {n_train} rows)", logf)
            if args.estimator == "pzflow":
                this_fit = fill_nondetections(this_fit, mag_limits, outdir)
        t0 = time.time()
        informer.inform(TableHandle(f"fit_{vtag}", path=str(this_fit)))
        log(f"[{variant}] inform done ({time.time()-t0:.1f}s, peak RSS {peak_rss_gb():.2f} GB)", logf)

        t0 = time.time()
        est = EstimatorClass.make_stage(name=f"estimate_valid_{vtag}", **est_kwargs)
        ens = as_interp(qp.read(est.estimate(TableHandle(f"valid_{vtag}", path=str(valid_path))).path), logf, f"{variant} validation")
        assert ens.npdf == len(z_mb), "row-order mismatch between estimates and validation truth"
        z_mode = np.squeeze(ens.ancil["zmode"])
        pdfs = normalise(np.asarray(ens.pdf(ZGRID)), ZGRID)
        log(f"[{variant}] estimate valid done ({time.time()-t0:.1f}s)", logf)

        pit_m, pit = pit_metrics(ens, z_mb)
        acc, nz_rows = nz_check(pdfs, ZGRID, z_mb, TS2_EDGES)
        vm = dict(
            point_zmode_vs_manyband=point_metrics(z_mb, z_mode),
            point_zmode_vs_specz=point_metrics(z_sp[has_sp], z_mode[has_sp]) if has_sp.sum() > 20 else None,
            point_zmode_vs_manyband_on_specz_rows=point_metrics(z_mb[has_sp], z_mode[has_sp]) if has_sp.sum() > 20 else None,
            bright=subset_metrics(z_mb, z_mode, i_valid, i_valid <= FAINT_CUT),
            faint=subset_metrics(z_mb, z_mode, i_valid, i_valid > FAINT_CUT),
            uncovered=subset_metrics(z_mb, z_mode, i_valid, uncovered) if uncovered.sum() > 20 else None,
            zmode_vs_imag=binned_metrics(z_mb, z_mode, i_valid, np.arange(18, 26.5, 1.0), "i"),
            zmode_vs_z=binned_metrics(z_mb, z_mode, z_mb, np.arange(0, 2.6, 0.25), "z"),
            mean_log_p=loglik(pdfs, ZGRID, z_mb),
            tomo_bin_accuracy=acc,
            tomo_bins=nz_rows,
            tomo_rms_dmean=tomo_rms(nz_rows, "mean"),
            tomo_rms_dstd=tomo_rms(nz_rows, "std"),
            **pit_m,
        )
        metrics["variants"][variant] = vm
        pm, pf, pb = vm["point_zmode_vs_manyband"], vm["faint"], vm["bright"]
        log(f"[{variant}] ALL   : bias_med={pm['bias_median']:+.4f} sigma_NMAD={pm['sigma_nmad']:.4f} outliers={100*pm['outlier_frac']:.2f}%  "
            f"PIT extremes={100*(pit_m['pit_frac_below_0p05']+pit_m['pit_frac_above_0p95']):.1f}%  <log p>={vm['mean_log_p']:+.3f}", logf)
        log(f"[{variant}] i<={FAINT_CUT:.0f}: bias_med={pb['bias_median']:+.4f} sigma_NMAD={pb['sigma_nmad']:.4f} outliers={100*pb['outlier_frac']:.2f}%", logf)
        log(f"[{variant}] i>{FAINT_CUT:.0f} : bias_med={pf['bias_median']:+.4f} sigma_NMAD={pf['sigma_nmad']:.4f} outliers={100*pf['outlier_frac']:.2f}%  ({100*pf['frac_of_sample']:.0f}% of sample)", logf)
        log(f"[{variant}] tomo (ts2 edges): accuracy={acc:.3f}  rms dmean={vm['tomo_rms_dmean']:.4f}  rms dstd={vm['tomo_rms_dstd']:.4f}  "
            + "  ".join(f"b{r['bin']}: n={r['n']} dmean={r['mean_stack']-r['mean_true']:+.3f} dstd={r['std_stack']-r['std_true']:+.3f}" for r in nz_rows), logf)
        make_plots(outdir, variant, z_mb, z_mode, i_valid, pit)
        ens.set_ancil(dict(ens.ancil, object_id=np.asarray(valid["object_id"]), z_manyband=z_mb, z_spec=z_sp, mag_i=i_valid))
        ens.write_to(str(outdir / f"pz_valid_{variant}.hdf5"))

        if not args.skip_test:
            t0 = time.time()
            est_t = EstimatorClass.make_stage(name=f"estimate_test_{vtag}", **est_kwargs)
            ens_t = as_interp(qp.read(est_t.estimate(TableHandle(f"test_{vtag}", path=str(test_file_est))).path), logf, f"{variant} test")
            assert ens_t.npdf == len(test_ids)
            ens_t.set_ancil(dict(ens_t.ancil, object_id=test_ids))
            sub_dir = outdir / f"submission_{variant}"; sub_dir.mkdir(exist_ok=True)
            sub_file = sub_dir / f"pz_challenge_taskset_2_{args.sim}_pz_estimate_{args.scenario}.hdf5"
            ens_t.write_to(str(sub_file))
            log(f"[{variant}] test estimate written: {sub_file} ({time.time()-t0:.1f}s)", logf)

    metrics["peak_rss_gb"] = peak_rss_gb()
    from pz_pilot import NOTES
    metrics["notes"] = list(NOTES)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"metrics -> {outdir/'metrics.json'}", logf)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
