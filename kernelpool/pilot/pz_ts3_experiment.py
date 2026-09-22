#!/usr/bin/env python3
"""PZ task sets 3/4: spectroscopic + many-band (COSMOS2020-like) reference sample.

Question: how much do the representative-but-noisy many-band redshifts of the
COSMOS field add to the clean-but-selected spectroscopic redshifts, per object
and for the tomographic n(z)? (Docs: "the point of this taskset is to find a
way to optimally use the additional information from the COSMOS2020 field".)

Training file (100k rows, Cardinal 1yr): `redshift` (spec-z) for ~65k rows,
`redshift_manyband` for the ~39k COSMOS-flagged rows (4k have both). The
many-band rows reach i = 25.4 with the same magnitude distribution as the test
sample; the spec-z rows stop at i ~ 23.5. The many-band labels carry an
emulated error (challenge docs, figures foutlier_vs_mag_i / sigma_nmad_vs_mag_i,
matched to Khostovan et al. 2026): outlier fraction ~0 at i < 22 rising to
~12 % (Cardinal) / ~8 % (Flagship) at i > 24, sigma_NMAD 0.01 -> 0.033.

Label variants (training label = spec-z where available, else many-band):
  spec      : spec-z rows only                      (= the task-set-2 situation)
  manyband  : COSMOS rows only, many-band labels    (representative, noisy)
  union     : all rows                              (both)
  union_w   : all rows, with importance weights that make the training
              distribution in (i, colours) match that of the many-band
              (representative) rows: w = n_manyband(x) / n_union(x) from
              k-nearest-neighbour densities (Lima et al. 2008, as in
              pz_ts2_experiment.py). FlexZBoost takes them as sample weights;
              the other estimators resample with probability ∝ w.

Validation:
  V1 = a random --holdout fraction of the COSMOS rows, removed from *every*
       training variant (same rows for all), scored against their many-band
       label (and spec-z where present). Representative to i = 25.4.
  V2 = NZ task-set-2 ddf_00 of the same simulation (many-band label for all
       rows; PZ train/test overlaps removed; PZ test faint limit applied), as
       in pz_ts2_experiment.py.
  Both truths carry the many-band label error, so absolute outlier fractions
  at i > 24 are inflated by roughly the label-outlier fraction (reported
  alongside as `expected_label_outliers`); comparisons between variants are
  unaffected. Where spec-z exist the metrics are also given against them.

Estimators: --estimator flexzboost (default) | gpz | pzflow | knn, settings as
in pz_pilot.py. Outputs per variant: metrics on V1 and V2 (all / i<=24 /
i>24 / per magnitude bin, PIT, mean log p, task-set-2 tomographic-bin
accuracy and stacked-vs-true mean/width), coverage diagnostic, diagnostics
plots, PDFs with truth attached, and a submission file for the PZ test set.

Usage (from the DESC_NZ_Challenge folder, env desc-pz activated):
  python pilot/pz_ts3_experiment.py --taskset 3 --sim cardinal --scenario 1yr --nmax 12000 --nvalid 5000  # test
  python pilot/pz_ts3_experiment.py --taskset 3 --sim cardinal --scenario 1yr                              # full
  python pilot/pz_ts3_experiment.py --taskset 3 --sim cardinal --scenario 1yr --estimator gpz

Outputs go to pilot/runs/ts<N>exp_<sim>_<scenario>[_<estimator>][_nmaxN][_label]/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qp
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

from pz_pilot import ZMIN, ZMAX, NZBINS, ESTIMATORS, NOTES, as_interp, binned_metrics, build_stages, fill_nondetections, log, peak_rss_gb, pit_metrics, point_metrics
from pz_ts2_experiment import TS2_EDGES, ZGRID, FAINT_CUT, band_columns, build_validation, coverage, feature_matrix, knn_density_ratio, loglik, make_plots, pos_key, subset_metrics
from calibrate_pdfs import nz_check, normalise, tomo_rms

# Many-band (COSMOS2020-like) label outlier fraction vs i, read off the challenge
# docs figure foutlier_vs_mag_i.jpg (points for the taskset-3 training files).
# Used only for reporting the expected label-outlier contamination of a sample.
LABEL_OUTLIER_TABLE = {
    "cardinal": ([18.0, 21.5, 22.2, 22.7, 23.2, 23.7, 24.2, 24.7, 25.3], [0.0, 0.0, 0.02, 0.023, 0.045, 0.07, 0.12, 0.12, 0.125]),
    "flagship": ([18.0, 21.5, 22.2, 22.7, 23.2, 23.7, 24.2, 24.7, 25.3], [0.0, 0.0, 0.003, 0.005, 0.015, 0.025, 0.08, 0.085, 0.085]),
}


def expected_label_outliers(mag_i: np.ndarray, sim: str) -> float:
    x, y = LABEL_OUTLIER_TABLE[sim]
    m = np.isfinite(mag_i)
    return float(np.mean(np.interp(mag_i[m], x, y)))


def score_set(name, ens, z_lab, z_sp, mag_i, uncovered, sim, logf, variant):
    """All metrics for one validation set; z_lab = label used as truth, z_sp = spec-z (NaN where absent)."""
    z_mode = np.squeeze(ens.ancil["zmode"])
    pdfs = normalise(np.asarray(ens.pdf(ZGRID)), ZGRID)
    has_sp = np.isfinite(z_sp)
    pit_m, pit = pit_metrics(ens, z_lab)
    acc, nz_rows = nz_check(pdfs, ZGRID, z_lab, TS2_EDGES)
    faint = mag_i > FAINT_CUT
    d = dict(
        n=int(len(z_lab)),
        point_zmode_vs_label=point_metrics(z_lab, z_mode),
        point_zmode_vs_specz=point_metrics(z_sp[has_sp], z_mode[has_sp]) if has_sp.sum() > 20 else None,
        n_specz=int(has_sp.sum()),
        bright=subset_metrics(z_lab, z_mode, mag_i, ~faint),
        faint=subset_metrics(z_lab, z_mode, mag_i, faint),
        expected_label_outliers_all=expected_label_outliers(mag_i, sim),
        expected_label_outliers_faint=expected_label_outliers(mag_i[faint], sim) if faint.sum() else None,
        uncovered_frac=float(uncovered.mean()),
        uncovered_frac_faint=float(uncovered[faint].mean()) if faint.sum() else None,
        zmode_vs_imag=binned_metrics(z_lab, z_mode, mag_i, np.arange(18, 26.5, 1.0), "i"),
        zmode_vs_z=binned_metrics(z_lab, z_mode, z_lab, np.arange(0, 2.6, 0.25), "z"),
        mean_log_p=loglik(pdfs, ZGRID, z_lab),
        tomo_bin_accuracy=acc,
        tomo_bins=nz_rows,
        tomo_rms_dmean=tomo_rms(nz_rows, "mean"),
        tomo_rms_dstd=tomo_rms(nz_rows, "std"),
        **pit_m,
    )
    pm, pf = d["point_zmode_vs_label"], d["faint"]
    log(f"[{variant}] {name}: ALL bias={pm['bias_median']:+.4f} sNMAD={pm['sigma_nmad']:.4f} outl={100*pm['outlier_frac']:.1f}% (PZ def {100*pm['outlier_frac_pz']:.1f}%) "
        f"PITextr={100*(pit_m['pit_frac_below_0p05']+pit_m['pit_frac_above_0p95']):.1f}% <logp>={d['mean_log_p']:+.2f} | "
        f"i>{FAINT_CUT:.0f} ({100*faint.mean():.0f}%): sNMAD={pf['sigma_nmad']:.4f} outl={100*pf['outlier_frac']:.1f}% "
        f"(label outliers expected {100*d['expected_label_outliers_faint']:.1f}%, uncovered {100*d['uncovered_frac_faint']:.0f}%) | "
        f"tomo acc={acc:.3f} rms dmean={d['tomo_rms_dmean']:.4f} rms dstd={d['tomo_rms_dstd']:.4f}", logf)
    if d["point_zmode_vs_specz"]:
        ps = d["point_zmode_vs_specz"]
        log(f"[{variant}] {name}: vs spec-z ({has_sp.sum()} rows): sNMAD={ps['sigma_nmad']:.4f} outl={100*ps['outlier_frac']:.2f}%", logf)
    return d, pit, z_mode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taskset", type=int, default=3, choices=[3, 4])
    ap.add_argument("--sim", default="cardinal", choices=["cardinal", "flagship"])
    ap.add_argument("--scenario", default="1yr")
    ap.add_argument("--base", default=None)
    ap.add_argument("--data-subdir", default="data/pz/public")
    ap.add_argument("--nz-data-subdir", default="data/nz/public")
    ap.add_argument("--nmax", type=int, default=None, help="use only this many training rows (quick test)")
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of COSMOS rows held out as V1")
    ap.add_argument("--nvalid", type=int, default=50000, help="size of V2 (ddf_00 subset); 0 to skip V2")
    ap.add_argument("--valid-imax", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="spec,manyband,union,union_w")
    ap.add_argument("--k", type=int, default=100); ap.add_argument("--wmax", type=float, default=50.0)
    ap.add_argument("--estimator", default="flexzboost", choices=list(ESTIMATORS))
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--flow-epochs", type=int, default=50)
    ap.add_argument("--gpz-basis", type=int, default=50)
    ap.add_argument("--gpz-iter", type=int, default=200)
    ap.add_argument("--trainfrac", type=float, default=0.75)
    ap.add_argument("--bump-min", type=float, default=0.02); ap.add_argument("--bump-max", type=float, default=0.35); ap.add_argument("--n-bump", type=int, default=20)
    ap.add_argument("--sharp-min", type=float, default=0.7); ap.add_argument("--sharp-max", type=float, default=2.1); ap.add_argument("--n-sharp", type=int, default=15)
    ap.add_argument("--max-basis", type=int, default=35)
    # estimator-level experiments (2026-09-17): RAIL/flexcode defaults are include_mag_err False, max_depth 8,
    # xgboost defaults n_estimators 100, learning_rate 0.3, subsample 1, min_child_weight 1
    ap.add_argument("--include-mag-err", action="store_true", help="add the colour errors to the FlexZBoost feature matrix")
    ap.add_argument("--xgb-depth", type=int, default=8); ap.add_argument("--xgb-trees", type=int, default=100)
    ap.add_argument("--xgb-lr", type=float, default=0.3); ap.add_argument("--xgb-subsample", type=float, default=1.0)
    ap.add_argument("--xgb-min-child", type=float, default=1.0)
    ap.add_argument("--catalogs", default=None, help="RAIL catalog config yaml (default: repos/nz_data_challenge/tests/catalogs.yaml under --base; the PZ repo ships the identical file)")
    args = ap.parse_args()

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    key = f"ts{args.taskset}exp_{args.sim}_{args.scenario}"
    tag = key + (f"_{args.estimator}" if args.estimator != "flexzboost" else "") + (f"_nmax{args.nmax}" if args.nmax else "") + (f"_{args.label}" if args.label else "")
    outdir = base / "pilot" / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    logf = open(outdir / "run.log", "a")
    log(f"=== pz_ts3_experiment {key}  base={base}", logf)
    log(f"args: {vars(args)}", logf)

    train_file = base / args.data_subdir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_training_{args.scenario}.hdf5"
    test_file = base / args.data_subdir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_test_{args.scenario}.hdf5"
    nz_file = base / args.nz_data_subdir / f"nz_challenge_taskset_2_{args.sim}_{args.scenario}_ddf_00.hdf5"
    for f in (train_file, test_file) + ((nz_file,) if args.nvalid else ()):
        if not f.exists():
            raise FileNotFoundError(f)

    catalog_utils.clear()
    catalog_utils.load_yaml(str(Path(args.catalogs) if args.catalogs else base / "repos" / "nz_data_challenge" / "tests" / "catalogs.yaml"))
    catalog_utils.apply(f"{args.sim}_roman_rubin")
    common = dict(hdf5_groupname="", zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS, nondetect_val=np.nan)
    if args.estimator == "flexzboost":
        common["include_mag_err"] = bool(args.include_mag_err)     # informer AND estimator build the same feature matrix
    xgb_params = dict(max_depth=args.xgb_depth, objective="reg:squarederror", n_estimators=args.xgb_trees,
                      learning_rate=args.xgb_lr, subsample=args.xgb_subsample, min_child_weight=args.xgb_min_child)
    fzb_kwargs = dict(trainfrac=args.trainfrac, bumpmin=args.bump_min, bumpmax=args.bump_max, nbump=args.n_bump,
                      sharpmin=args.sharp_min, sharpmax=args.sharp_max, nsharp=args.n_sharp, max_basis=args.max_basis,
                      regression_params=xgb_params)
    log(f"FlexZBoost settings: include_mag_err={args.include_mag_err} max_basis={args.max_basis} xgb={xgb_params}", logf)
    probe = FlexZBoostInformer.make_stage(name="probe", **common)
    bands, err_bands, mag_limits = band_columns(probe.config)
    ref_band = probe.config["ref_band"]

    # --- training file and label bookkeeping
    train_all = tables_io.read(str(train_file))
    train_all = {k: np.asarray(v) for k, v in train_all.items()}
    n_file = len(train_all["ra"])
    rng = np.random.default_rng(args.seed)
    if args.nmax is not None and args.nmax < n_file:
        keep = np.sort(rng.choice(n_file, args.nmax, replace=False))
        train_all = {k: v[keep] for k, v in train_all.items()}
    n_all = len(train_all["ra"])
    z_sp_all = train_all["redshift"].astype(float)
    z_mb_all = train_all["redshift_manyband"].astype(float)
    has_sp, has_mb = np.isfinite(z_sp_all), np.isfinite(z_mb_all)
    i_all = train_all[ref_band].astype(float)
    log(f"training file: {n_all} rows (file {n_file}); spec-z {has_sp.sum()}, many-band {has_mb.sum()}, both {(has_sp & has_mb).sum()}, neither {(~has_sp & ~has_mb).sum()}", logf)
    log(f"  i quantiles 10/50/90: spec-z rows {np.nanpercentile(i_all[has_sp], [10, 50, 90]).round(2)}, many-band rows {np.nanpercentile(i_all[has_mb], [10, 50, 90]).round(2)}", logf)
    both = has_sp & has_mb
    if both.sum() > 20:
        pm = point_metrics(z_sp_all[both], z_mb_all[both])
        log(f"  many-band vs spec-z on the {both.sum()} rows with both: sNMAD={pm['sigma_nmad']:.4f} outl={100*pm['outlier_frac']:.2f}% (i median {np.nanmedian(i_all[both]):.2f})", logf)

    # V1: hold out a fraction of the many-band rows (representative), removed from all variants
    mb_idx = np.flatnonzero(has_mb)
    n_hold = int(round(args.holdout * len(mb_idx)))
    hold = np.zeros(n_all, bool)
    hold[rng.choice(mb_idx, n_hold, replace=False)] = True
    v1 = {k: v[hold] for k, v in train_all.items()}
    v1_path = outdir / "valid_v1_cosmos_holdout.hdf5"
    tables_io.write(v1, str(v1_path))
    z1_lab, z1_sp, i1 = v1["redshift_manyband"].astype(float), v1["redshift"].astype(float), v1[ref_band].astype(float)
    log(f"V1 (COSMOS hold-out): {hold.sum()} rows, i quantiles {np.nanpercentile(i1, [10, 50, 90]).round(2)}, frac i>{FAINT_CUT}: {np.mean(i1 > FAINT_CUT):.3f}, spec-z for {np.isfinite(z1_sp).sum()}; "
        f"expected label outliers all/faint: {100*expected_label_outliers(i1, args.sim):.1f}% / {100*expected_label_outliers(i1[i1 > FAINT_CUT], args.sim):.1f}%", logf)
    pool = {k: v[~hold] for k, v in train_all.items()}
    p_sp, p_mb = np.isfinite(pool["redshift"]), np.isfinite(pool["redshift_manyband"])

    # V2: NZ ddf_00 representative set
    test = tables_io.read(str(test_file))
    test = {k: np.asarray(v) for k, v in test.items()}
    i_test = test[ref_band].astype(float)
    test_ids = test["object_id"].astype(int)
    log(f"PZ test: {len(i_test)} rows, i quantiles {np.nanpercentile(i_test, [10, 50, 90]).round(2)}, frac i>{FAINT_CUT}: {np.nanmean(i_test > FAINT_CUT):.3f}", logf)
    v2 = None
    if args.nvalid:
        exclude = np.concatenate([pos_key(tables_io.read(str(train_file))), pos_key(test)])
        imax = args.valid_imax if args.valid_imax is not None else float(np.nanmax(i_test))
        v2_path, v2, vinfo = build_validation(nz_file, exclude, args.nvalid, args.seed, outdir, ref_band, imax)
        z2_lab, z2_sp, i2 = v2["redshift_manyband"].astype(float), v2["redshift"].astype(float), v2[ref_band].astype(float)
        log(f"V2 (ddf_00): {vinfo}; expected label outliers all/faint: {100*expected_label_outliers(i2, args.sim):.1f}% / {100*expected_label_outliers(i2[i2 > FAINT_CUT], args.sim):.1f}%", logf)

    test_file_est = test_file
    if args.estimator == "pzflow":
        v1_path = fill_nondetections(v1_path, mag_limits, outdir)
        if v2 is not None:
            v2_path = fill_nondetections(v2_path, mag_limits, outdir)
        test_file_est = fill_nondetections(test_file, mag_limits, outdir)

    metrics = dict(key=key, taskset=args.taskset, estimator=args.estimator, n_train_file=n_all, holdout=args.holdout,
                   n_v1=int(hold.sum()), label_outlier_table=LABEL_OUTLIER_TABLE[args.sim], variants={})
    x_v1 = feature_matrix(v1, bands, mag_limits, ref_band)
    x_v2 = feature_matrix(v2, bands, mag_limits, ref_band) if v2 is not None else None

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if variant == "spec":
            sel = p_sp; label = pool["redshift"].astype(float)
        elif variant == "manyband":
            sel = p_mb; label = pool["redshift_manyband"].astype(float)
        elif variant in ("union", "union_w"):
            sel = p_sp | p_mb; label = np.where(p_sp, pool["redshift"], pool["redshift_manyband"]).astype(float)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        fit = {k: v[sel] for k, v in pool.items()}
        fit["redshift"] = label[sel]           # the training label the estimator sees
        # Labels outside the grid are clipped to its edge (2026-09-22, ours). Before, they were passed through: flexcode's
        # cosine basis is symmetric about the box edge, so a label at zmax + d was fitted exactly like one at zmax - d
        # (and a kernel-outlier label at 5.9 like one at 0.1). Clipping keeps the row and puts its mass at the edge.
        n_out = int(((fit["redshift"] < ZMIN) | (fit["redshift"] > ZMAX)).sum())
        fit["redshift"] = np.clip(fit["redshift"], ZMIN, ZMAX)
        log(f"[{variant}] grid {ZMIN}-{ZMAX} with {NZBINS} points; {n_out} training labels outside the grid clipped to its edge", logf)
        weights = None
        if variant == "union_w":
            x_u = feature_matrix(fit, bands, mag_limits, ref_band)
            x_t = feature_matrix({k: v[p_mb] for k, v in pool.items()}, bands, mag_limits, ref_band)
            weights, _, n_tg, _ = knn_density_ratio(x_u, x_t, args.k, args.wmax)
            n_eff = float(weights.sum() ** 2 / (weights ** 2).sum())
            fit["weight"] = weights
            log(f"[{variant}] weights towards the many-band distribution: n_eff={n_eff:.0f} of {sel.sum()}, quantiles 10/50/90/99 {np.percentile(weights, [10, 50, 90, 99]).round(3)}, raw count 0 for {np.mean(n_tg == 0):.3f}", logf)
        fit_path = outdir / f"fit_{variant}.hdf5"
        tables_io.write(fit, str(fit_path))
        i_fit = fit[ref_band].astype(float)
        log(f"[{variant}] training rows {sel.sum()} (spec-z label {int((p_sp & sel).sum())}, many-band label {int((~p_sp & p_mb & sel).sum())}); "
            f"i quantiles {np.nanpercentile(i_fit, [10, 50, 90]).round(2)}, frac i>{FAINT_CUT}: {np.nanmean(i_fit > FAINT_CUT):.3f}", logf)
        x_fit = feature_matrix(fit, bands, mag_limits, ref_band)
        scale = x_fit.std(axis=0); scale[scale == 0] = 1.0
        d1, ref1 = coverage(x_fit, x_v1, scale); unc1 = d1 > ref1
        if x_v2 is not None:
            d2, ref2 = coverage(x_fit, x_v2, scale); unc2 = d2 > ref2

        vtag = f"{tag}_{variant}"
        model_file = outdir / f"{args.estimator}_model_{variant}.pkl"
        this_fit = fit_path
        if args.estimator == "flexzboost":
            informer = FlexZBoostInformer.make_stage(name=f"inform_{vtag}", model=str(model_file), **common, **fzb_kwargs,
                                                     use_weights=(weights is not None), weights_column="weight")
            est_kwargs = dict(model=str(model_file), **common, chunk_size=20000,
                              calculated_point_estimates=["zmode", "zmean"], qp_representation="interp")
            EstimatorClass = FlexZBoostEstimator
        else:
            informer, est_kwargs, EstimatorClass = build_stages(args.estimator, vtag, model_file, args)
            if weights is not None:
                pick = rng.choice(len(weights), len(weights), replace=True, p=weights / weights.sum())
                res = {k: np.asarray(v)[pick] for k, v in fit.items()}
                this_fit = outdir / f"fit_{variant}_resampled.hdf5"
                tables_io.write(res, str(this_fit))
                log(f"[{variant}] {args.estimator} has no weight input: importance-resampled ({len(np.unique(pick))} distinct rows)", logf)
            if args.estimator == "pzflow":
                this_fit = fill_nondetections(this_fit, mag_limits, outdir)
        t0 = time.time()
        if args.estimator == "gpz":
            # RAIL's GPzInformer draws its internal train/validation split with np.random.permutation (the global,
            # unseeded numpy RNG); only the GP's own draws take `seed`. Seed the global RNG here so that a retraining
            # is reproducible (recipe S review 2026-09-21, item 7). The GPz models trained before this line was added
            # (2026-09-16/17) used an unseeded split; their pickles are kept and are what the submissions use.
            np.random.seed(args.seed)
            log(f"[{variant}] numpy global RNG seeded with {args.seed} for the GPz train/validation split", logf)
        informer.inform(TableHandle(f"fit_{vtag}", path=str(this_fit)))
        log(f"[{variant}] inform done ({time.time()-t0:.1f}s, peak RSS {peak_rss_gb():.2f} GB)", logf)

        vm = dict(n_train=int(sel.sum()), n_train_specz_label=int((p_sp & sel).sum()), n_train_manyband_label=int((~p_sp & p_mb & sel).sum()))
        t0 = time.time()
        est = EstimatorClass.make_stage(name=f"estimate_v1_{vtag}", **est_kwargs)
        ens1 = as_interp(qp.read(est.estimate(TableHandle(f"v1_{vtag}", path=str(v1_path))).path), logf, f"{variant} V1")
        assert ens1.npdf == len(z1_lab)
        vm["V1_cosmos_holdout"], pit1, zm1 = score_set("V1", ens1, z1_lab, z1_sp, i1, unc1, args.sim, logf, variant)
        make_plots(outdir, f"{variant}_V1", z1_lab, zm1, i1, pit1)
        ens1.set_ancil(dict(ens1.ancil, object_id=v1["object_id"], z_manyband=z1_lab, z_spec=z1_sp, mag_i=i1))
        ens1.write_to(str(outdir / f"pz_valid_{variant}.hdf5"))
        if v2 is not None:
            est2 = EstimatorClass.make_stage(name=f"estimate_v2_{vtag}", **est_kwargs)
            ens2 = as_interp(qp.read(est2.estimate(TableHandle(f"v2_{vtag}", path=str(v2_path))).path), logf, f"{variant} V2")
            assert ens2.npdf == len(z2_lab)
            vm["V2_ddf00"], pit2, zm2 = score_set("V2", ens2, z2_lab, z2_sp, i2, unc2, args.sim, logf, variant)
            make_plots(outdir, f"{variant}_V2", z2_lab, zm2, i2, pit2)
            ens2.set_ancil(dict(ens2.ancil, object_id=v2["object_id"], z_manyband=z2_lab, z_spec=z2_sp, mag_i=i2))
            ens2.write_to(str(outdir / f"pz_valid_{variant}_v2.hdf5"))
        log(f"[{variant}] validation estimates done ({time.time()-t0:.1f}s)", logf)
        metrics["variants"][variant] = vm

        if not args.skip_test:
            t0 = time.time()
            est_t = EstimatorClass.make_stage(name=f"estimate_test_{vtag}", **est_kwargs)
            ens_t = as_interp(qp.read(est_t.estimate(TableHandle(f"test_{vtag}", path=str(test_file_est))).path), logf, f"{variant} test")
            assert ens_t.npdf == len(test_ids)
            ens_t.set_ancil(dict(ens_t.ancil, object_id=test_ids))
            sub_dir = outdir / f"submission_{variant}"; sub_dir.mkdir(exist_ok=True)
            sub_file = sub_dir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_pz_estimate_{args.scenario}.hdf5"
            ens_t.write_to(str(sub_file))
            log(f"[{variant}] test estimate written: {sub_file} ({time.time()-t0:.1f}s)", logf)

    metrics["peak_rss_gb"] = peak_rss_gb()
    metrics["notes"] = list(NOTES)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"metrics -> {outdir/'metrics.json'}", logf)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
