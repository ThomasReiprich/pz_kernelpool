#!/usr/bin/env python3
"""Treat the many-band (COSMOS2020-like) labels as noisy: exact label kernel + deconvolution.

The challenge's many-band redshifts were generated from the true redshift with
RAIL's `GaussianSkewtScatterSelector` (rail_astro_tools,
rail/creation/degraders/gaussian_skewt_scatter_selector.py; referenced by the
`rail_projects` project library the challenge docs point to). Its default
model ("gaussian_yin+25_tail_khostovan+26") is

    z_label = clip(z_true + delta, 0, 6),
    delta ~ (1 - f(i)) * Normal(mu(i, z), sigma(i, z)) + f(i) * SkewT_i(delta),

with i-band and redshift bins, a Gaussian core (parameters from Yin et al.
2025) and a Jones-Faddy skew-t tail (fit to Khostovan et al. 2026, COSMOS2020)
whose fraction f(i) is 0.07 for i < 24 and 0.24 for i > 24. The parameter
table below is copied from that file (other people's code; the model and its
parameters are the organizers'/RAIL's). We verified (2026-09-15) that it
reproduces the empirical sigma_NMAD, outlier fraction and bias of the labels
on the spec-z pairs of the ddf_00 file in every magnitude bin up to i = 24.

What this module does (ours):
  * builds the label kernel K_i[z_label, z_true] on the common redshift grid
    for each i-band bin (RAIL's re-drawing of out-of-range labels = truncation);
  * deconvolves per-object PDFs that were trained on many-band labels
    (they estimate p(z_label | x), not p(z | x)) with Richardson-Lucy
    iterations, p_{k+1} = p_k * K^T[ p_lab / (K p_k) ], which keeps
    positivity and normalisation; and the inverse operation (smear);
  * `synthetic` mode: an end-to-end test on real photometry with KNOWN truth:
    the spec-z rows of the task-set-3 training file get synthetic many-band
    labels drawn from the model (optionally with the faint-end parameters
    for every object, as a stress test), FlexZBoost is trained on those
    labels, and the raw and deconvolved PDFs of held-out rows are scored
    against the true (spectroscopic) redshift;
  * `apply` mode: deconvolves the validation and test PDFs of a
    pz_ts3_experiment.py run, scores raw vs deconvolved against the spec-z
    rows (bright), checks forward consistency (the re-smeared deconvolved
    PDFs must fit the many-band labels at least as well as the raw ones),
    reports the change of the stacked n(z) per tomographic bin, and writes
    `pz_valid_<variant>_deconv.hdf5` and `submission_<variant>_deconv/`.

  * `check` mode: compares the model with the spec-z/many-band overlap rows of
    a training file per i-band bin (to confirm the same model in every
    simulation/scenario; the organizers' pipeline configuration applies the
    selector with its default parameters to Cardinal and Flagship alike).

Usage (from the DESC_NZ_Challenge folder, env desc-pz activated):
  python pilot/manyband_mixture.py check --taskset 3 --sim flagship --scenario 1yr
  python pilot/manyband_mixture.py synthetic --taskset 3 --sim cardinal --scenario 1yr [--faint-mode]
  python pilot/manyband_mixture.py apply pilot/runs/ts3exp_cardinal_1yr --variant union_w --iters 5 --calibrate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import jf_skew_t, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------- label model (RAIL, copied)
KHOSTOVAN26 = dict(
    mag_i_bin_edges=[15.5, 22.0, 23.0, 24.0, 29.0],
    z_bin_edges=[0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    bias_median_lookup_table=[[0.0, 0.002, -0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
                              [0.0, -0.0, -0.002, -0.004, 0.01, 0.01, 0.01, 0.01],
                              [0.003, -0.0, -0.001, -0.006, -0.002, 0.024, 0.007, 0.0],
                              [0.008, -0.005, 0.007, -0.015, -0.019, 0.017, 0.011, 0.0]],
    bias_std_lookup_table=[[0.01, 0.02, 0.026, 0.038, 0.038, 0.038, 0.038, 0.038],
                           [0.011, 0.019, 0.025, 0.036, 0.062, 0.062, 0.062, 0.062],
                           [0.011, 0.022, 0.027, 0.044, 0.063, 0.115, 0.093, 0.074],
                           [0.013, 0.023, 0.025, 0.051, 0.069, 0.12, 0.103, 0.069]],
    f_tail_by_mag_i=[0.0709, 0.0672, 0.0618, 0.2425],
    tail_loc_by_mag_i=[0.0431, -0.0847, -0.0371, -0.0979],
    tail_scale_by_mag_i=[0.1585, 0.1018, 0.4109, 0.2499],
    tail_a_by_mag_i=[8.4098, 0.5426, 1.1277, 0.5394],
    tail_b_by_mag_i=[8.747, 0.4436, 0.8467, 0.5454],
)
M = {k: np.array(v) for k, v in KHOSTOVAN26.items()}
N_MAGBIN = len(M["f_tail_by_mag_i"])


def mag_bin(mag_i: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(mag_i, M["mag_i_bin_edges"]) - 1, 0, N_MAGBIN - 1)


def z_bin(z: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(z, M["z_bin_edges"]) - 1, 0, M["bias_median_lookup_table"].shape[1] - 1)


def sample_labels(z_true: np.ndarray, mag_i: np.ndarray, rng: np.random.Generator, force_bin: int | None = None) -> np.ndarray:
    """Draw synthetic many-band labels exactly as the RAIL selector does (incl. resampling of out-of-range draws)."""
    ib = np.full(len(z_true), force_bin) if force_bin is not None else mag_bin(mag_i)
    zb = z_bin(z_true)
    out = np.full(len(z_true), np.nan)
    todo = np.ones(len(z_true), bool)
    for _ in range(11):
        n = todo.sum()
        if n == 0:
            break
        d = rng.normal(M["bias_median_lookup_table"][ib[todo], zb[todo]], M["bias_std_lookup_table"][ib[todo], zb[todo]])
        tail = rng.random(n) < M["f_tail_by_mag_i"][ib[todo]]
        for k in np.unique(ib[todo][tail]):
            s = tail & (ib[todo] == k)
            d[s] = jf_skew_t.rvs(a=M["tail_a_by_mag_i"][k], b=M["tail_b_by_mag_i"][k], loc=M["tail_loc_by_mag_i"][k],
                                 scale=M["tail_scale_by_mag_i"][k], size=int(s.sum()), random_state=rng)
        zl = z_true[todo] + d
        ok = (zl >= 0) & (zl <= 6)
        idx = np.flatnonzero(todo)
        out[idx[ok]] = zl[ok]
        todo[idx[ok]] = False
    out[todo] = np.clip(z_true[todo], 0, 6)   # RAIL clips after 10 retries
    return out


LABEL_ZMAX = 6.0     # the selector accepts labels in [0, 6] and re-draws outside (RAIL); labels live on 0-6, true z on the PDF grid


def label_grid(zgrid: np.ndarray) -> np.ndarray:
    """The label grid: the true-z grid's origin and step, extended to LABEL_ZMAX (0-6 at step 0.01 for the 0-3/301 grid).
    Smeared PDFs live on this grid; the true-z PDFs stay on zgrid (rectangular kernel, pre-submission review 2026-09-22)."""
    dz = zgrid[1] - zgrid[0]
    n = int(round((LABEL_ZMAX - zgrid[0]) / dz)) + 1
    return zgrid[0] + dz * np.arange(max(n, len(zgrid)))


def label_kernel(zgrid: np.ndarray, k: int) -> np.ndarray:
    """K[j, m] = P(label in cell j of label_grid(zgrid) | true z = zgrid[m]) for i-band bin k: a rectangular
    (n_label x n_true) operator. Columns sum to 1 over the label grid, i.e. over 0-6: RAIL re-draws labels that fall
    outside [0, 6] (up to 10 times), so the out-of-range mass is redistributed, not clipped. Until 2026-09-22 the
    kernel was square on the PDF grid (labels above the grid top were folded back by the renormalisation and
    evaluated at the grid's last point); the independent pre-submission review pointed this out."""
    lgrid = label_grid(zgrid)
    dz = zgrid[1] - zgrid[0]
    edges = np.concatenate([[lgrid[0] - dz / 2], lgrid[:-1] + dz / 2, [lgrid[-1] + dz / 2]])
    K = np.zeros((len(lgrid), len(zgrid)))
    f = M["f_tail_by_mag_i"][k]
    a, b, loc, sc = (M[key][k] for key in ("tail_a_by_mag_i", "tail_b_by_mag_i", "tail_loc_by_mag_i", "tail_scale_by_mag_i"))
    zb = z_bin(zgrid)
    mu, sig = M["bias_median_lookup_table"][k, zb], M["bias_std_lookup_table"][k, zb]
    for m_, zt in enumerate(zgrid):
        cdf_core = norm.cdf(edges - zt, loc=mu[m_], scale=sig[m_])
        cdf_tail = jf_skew_t.cdf(edges - zt, a=a, b=b, loc=loc, scale=sc)
        cdf = (1 - f) * cdf_core + f * cdf_tail
        col = np.diff(cdf)
        K[:, m_] = col / col.sum()
    return K


_KERNELS: dict = {}


def kernels(zgrid: np.ndarray) -> list[np.ndarray]:
    key = (len(zgrid), float(zgrid[0]), float(zgrid[-1]))
    if key not in _KERNELS:
        _KERNELS[key] = [label_kernel(zgrid, k) for k in range(N_MAGBIN)]
    return _KERNELS[key]


def smear(pdfs: np.ndarray, mag_i: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    """p(z_label | x) from p(z | x): forward application of the label kernel. The result lives on label_grid(zgrid)
    (same step, extended to LABEL_ZMAX), so evaluate it with that grid, not with zgrid."""
    Ks = kernels(zgrid)
    ib = mag_bin(mag_i)
    out = np.empty((pdfs.shape[0], Ks[0].shape[0]), dtype=pdfs.dtype)
    for k in range(N_MAGBIN):
        s = ib == k
        if s.any():
            out[s] = pdfs[s] @ Ks[k].T
    return out


def deconvolve(pdfs: np.ndarray, mag_i: np.ndarray, zgrid: np.ndarray, n_iter: int = 20, floor: float = 1e-12,
               f_exact: np.ndarray | None = None) -> np.ndarray:
    """Richardson-Lucy deconvolution of label-space PDFs to true-z space, per i-band bin.

    f_exact (per object, optional): fraction of the estimator's training labels in this object's kernel bin that
    were exact (spectroscopic) rather than many-band draws. The estimator then learned a mixture, and the kernel
    to remove is K_eff = f * identity + (1 - f) * K (expert review 2026-09-17, finding 1). Objects are grouped by
    (kernel bin, f rounded to 3 decimals) so that only a handful of kernels are built. None = f 0 everywhere."""
    Ks = kernels(zgrid)
    n_l, n_t = Ks[0].shape
    ib = mag_bin(mag_i)
    fx = np.zeros(len(mag_i)) if f_exact is None else np.round(np.nan_to_num(np.asarray(f_exact, float), nan=0.0), 3)
    out = np.empty_like(pdfs)
    for k, fval in sorted(set(zip(ib.tolist(), fx.tolist()))):
        s = (ib == k) & (fx == fval)
        if not s.any():
            continue
        K = Ks[k] if fval == 0.0 else fval * np.eye(n_l, n_t) + (1.0 - fval) * Ks[k]
        p0 = pdfs[s] / pdfs[s].sum(axis=1, keepdims=True)
        p_lab = np.zeros((p0.shape[0], n_l)); p_lab[:, :n_t] = p0     # the estimator's output on the true grid, seen as label-space, padded to the label grid
        p = p0.copy()
        for _ in range(n_iter):
            pred = p @ K.T                       # K p on the label grid
            ratio = p_lab / np.maximum(pred, floor)
            p = p * (ratio @ K)                  # K^T ratio
            p /= p.sum(axis=1, keepdims=True)
        out[s] = p
    dz = zgrid[1] - zgrid[0]
    return out / dz


# ---------------------------------------------------------------- calibration through the kernel
EPS_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)   # 0.4, 0.5 added 2026-09-22: the faint-bin fits sat at 0.3
SIGMA_GRID = (0.03, 0.05, 0.1, 0.2, 0.4)
ALPHA_GRID = (0.7, 0.85, 1.0, 1.25, 1.5, 2.0)   # < 1 broadens, > 1 sharpens
CAL_EDGES_DEFAULT = (0.0, 23.0, 24.0, 24.7, 99.0)


WIDTH_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)   # s: Gaussian broadening of width s(1+z_mean)
CAL_MODELS = {
    # name: (eps_grid, sigma_grid, alpha_grid, width_grid)
    "tail": (EPS_GRID, SIGMA_GRID, ALPHA_GRID, (0.0,)),                         # tail mixing + exponent (2026-09-15 recipe)
    "width": ((0.0, 0.02, 0.05, 0.1, 0.2, 0.3), (0.05, 0.1, 0.2, 0.4), (1.0,), WIDTH_GRID),   # broadening + tail, no exponent
    "both": ((0.0, 0.02, 0.05, 0.1, 0.2, 0.3), (0.05, 0.1, 0.2, 0.4), (0.7, 1.0, 1.5), WIDTH_GRID),
}


def broaden(p, x, s, mu=None):
    """Convolve each PDF with a Gaussian of width s(1+z_mean) (a width-scaling calibration term, ours).
    Objects are grouped by their width in grid cells (rounded to 0.1 cell) for speed."""
    from scipy.ndimage import gaussian_filter1d
    from calibrate_pdfs import mean_pdf, normalise
    if s <= 0:
        return p
    if mu is None:
        mu = mean_pdf(p, x)
    dz = x[1] - x[0]
    cells = np.round(s * (1.0 + mu) / dz, 1)
    out = np.empty_like(p)
    for c in np.unique(cells):
        g = cells == c
        out[g] = gaussian_filter1d(p[g], sigma=float(c), axis=1, mode="constant", cval=0.0) if c > 0 else p[g]
    return normalise(out, x)


def calibrate_pdfs_step(p, x, eps, sigma_t, alpha, width=0.0):
    """One calibration transform: optional exponent (alpha), Gaussian broadening (width), tail mixing (eps, sigma_t)."""
    from calibrate_pdfs import mix_tails
    q = p if width <= 0 else broaden(p, x, width)
    return mix_tails(q, x, eps, sigma_t, alpha)


def fit_calibration_through_kernel(p, mag_i, zgrid, z_label, eps_grid=EPS_GRID,
                                   sigma_grid=SIGMA_GRID, alpha_grid=ALPHA_GRID, width_grid=(0.0,), pit_bins=40, model=None,
                                   z_exact=None):
    """Fit the calibration of TRUE-space PDFs against NOISY labels: the objective is the label
    likelihood of the forward-smeared calibrated PDFs, sum log [K cal(p)](z_label). Then a PIT map
    fitted on the label side of the smeared calibrated PDFs, applied to the true-space CDF (an
    approximation: the kernel is close to a fixed smearing per i-band bin).
    `model` selects the parameter grids (CAL_MODELS: 'tail' = eps, sigma_t, alpha; 'width' adds a
    Gaussian broadening s(1+z) and drops the exponent; 'both' has all four).
    `z_exact` (optional, NaN where absent): rows that have an exact (spectroscopic) redshift are scored on the
    UNsmeared PDF at z_exact instead of through the kernel at z_label -- the per-row forward correction with the
    identity kernel for exact labels (expert-review follow-up 2026-09-17); the PIT map is then fitted on the
    mixed PIT set (true-space PIT for exact rows, label-side PIT for the others).
    Returns dict(eps, sigma_t, alpha, width, recal) and the mean label log-likelihood on the fit set."""
    from calibrate_pdfs import PITRecalibrator, loglik, pit, eval_at
    if model is not None:
        eps_grid, sigma_grid, alpha_grid, width_grid = CAL_MODELS[model]
    ex = np.isfinite(z_exact) if z_exact is not None else np.zeros(len(z_label), bool)
    nx = ~ex

    lgrid = label_grid(zgrid)     # smeared PDFs live on the label grid (0-6)

    def objective(q):
        if not ex.any():
            return loglik(smear(q, mag_i, zgrid), lgrid, z_label)
        lp = np.empty(len(z_label))
        lp[ex] = np.log(np.maximum(eval_at(q[ex], zgrid, z_exact[ex]), 1e-6))
        if nx.any():
            lp[nx] = np.log(np.maximum(eval_at(smear(q[nx], mag_i[nx], zgrid), lgrid, z_label[nx]), 1e-6))
        return float(lp.mean())

    best = None
    prof_alpha, prof_eps = {}, {}     # profile likelihoods: best <log p> over the other parameters, per alpha / per eps
    for width in width_grid:
        pw = broaden(p, zgrid, width)
        for eps in eps_grid:
            for sig in (sigma_grid if eps > 0 else (sigma_grid[0],)):
                for alpha in alpha_grid:
                    q = calibrate_pdfs_step(pw, zgrid, eps, sig, alpha)
                    ll = objective(q)
                    if best is None or ll > best[0]:
                        best = (ll, eps, sig, alpha, width)
                    prof_alpha[alpha] = max(prof_alpha.get(alpha, -np.inf), ll)
                    prof_eps[eps] = max(prof_eps.get(eps, -np.inf), ll)
    ll, eps, sig, alpha, width = best
    # identifiability diagnostic (recipe S review 2026-09-21, item 1): the total log-likelihood gain of the best fit
    # over "no change" (alpha = 1, eps = 0) and over the profile in alpha; n_fit * delta is a likelihood-ratio statistic
    n = len(z_label)
    ll_raw = objective(p)
    profiles = dict(n_fit=n, dlogL_total_vs_raw=float(n * (ll - ll_raw)),
                    alpha_profile={str(a): float(n * (v - ll)) for a, v in sorted(prof_alpha.items())},
                    eps_profile={str(e): float(n * (v - ll)) for e, v in sorted(prof_eps.items())})
    q = calibrate_pdfs_step(p, zgrid, eps, sig, alpha, width)
    u = pit(smear(q, mag_i, zgrid), lgrid, z_label)
    if ex.any():
        u[ex] = pit(q[ex], zgrid, z_exact[ex])
    recal = PITRecalibrator(u, nbins=pit_bins)
    return dict(eps=eps, sigma_t=sig, alpha=alpha, width=width, recal=recal, loglik_fit=ll, n_exact=int(ex.sum()), profiles=profiles)


def fit_calibration_binned(p, mag_i, zgrid, z_label, edges=CAL_EDGES_DEFAULT, min_n=300, z_exact=None, **kw):
    """Through-kernel calibration fitted separately per i-band bin (edges), because the raw PDFs are
    under-confident for bright objects and over-confident for faint ones (2026-09-15 runs), which one
    global (eps, sigma_t, alpha, PIT map) cannot serve. Bins with fewer than min_n fit objects use
    the global fit. (Binning the recalibration is what the accepted `conclave` entry does, by a
    support-deficit measure; here the bins are in i magnitude.)"""
    glob = fit_calibration_through_kernel(p, mag_i, zgrid, z_label, z_exact=z_exact, **kw)
    cals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (mag_i >= lo) & (mag_i < hi)
        if s.sum() >= min_n:
            c = fit_calibration_through_kernel(p[s], mag_i[s], zgrid, z_label[s], z_exact=(None if z_exact is None else z_exact[s]), **kw); c["n_fit"] = int(s.sum())
        else:
            c = dict(glob); c["n_fit"] = int(s.sum()); c["fallback_global"] = True
        cals.append(c)
    return dict(edges=list(edges), bins=cals, global_fit=glob)


def cal_summary(cal) -> dict:
    """JSON-serialisable summary of a (binned or global) calibration."""
    if "bins" in cal:
        return dict(edges=cal["edges"], bins=[{k: c[k] for k in ("eps", "sigma_t", "alpha", "width", "loglik_fit", "n_fit", "n_exact", "fallback_global", "profiles") if k in c}
                                              for c in cal["bins"]])
    return {k: cal[k] for k in ("eps", "sigma_t", "alpha", "width", "loglik_fit", "profiles") if k in cal}


def apply_calibration(p, zgrid, cal, with_pit_map=True, mag_i=None):
    if "bins" in cal:
        if mag_i is None:
            raise ValueError("binned calibration needs mag_i")
        out = np.empty_like(p)
        edges = cal["edges"]
        for (lo, hi), c in zip(zip(edges[:-1], edges[1:]), cal["bins"]):
            s = (mag_i >= lo) & (mag_i < hi)
            if s.any():
                out[s] = apply_calibration(p[s], zgrid, c, with_pit_map)
        return out
    q = calibrate_pdfs_step(p, zgrid, cal["eps"], cal["sigma_t"], cal["alpha"], cal.get("width", 0.0))
    return cal["recal"].apply(q, zgrid) if with_pit_map else q


# ---------------------------------------------------------------- scoring helpers
def score(p, x, z_true, edges):
    from calibrate_pdfs import loglik, nz_check, normalise, pit, pit_summary
    from pz_pilot import point_metrics
    p = normalise(np.maximum(p, 0), x)
    zmode = x[np.argmax(p, axis=1)]
    d = dict(point=point_metrics(z_true, zmode), mean_log_p=loglik(p, x, z_true), pit=pit_summary(pit(p, x, z_true)))
    acc, rows = nz_check(p, x, z_true, edges)
    d["tomo_accuracy"] = acc
    from calibrate_pdfs import tomo_rms
    d["tomo_rms_dmean"] = tomo_rms(rows, "mean")
    d["tomo_rms_dstd"] = tomo_rms(rows, "std")
    d["tomo_bins"] = rows
    return d


def fmt(d):
    p = d["point"]
    return (f"sNMAD={p['sigma_nmad']:.4f} outl={100*p['outlier_frac']:.1f}% bias={p['bias_median']:+.4f} "
            f"<logp>={d['mean_log_p']:+.3f} PITextr={100*d['pit']['extreme_frac']:.1f}% KS={d['pit']['ks_to_uniform']:.3f} "
            f"| tomo acc={d['tomo_accuracy']:.3f} rms dmean={d['tomo_rms_dmean']:.4f} rms dstd={d['tomo_rms_dstd']:.4f}")


# ---------------------------------------------------------------- synthetic end-to-end test
def run_check(args) -> int:
    """Compare the label model with the rows of a training file that carry both a spec-z and a many-band label.

    The organizers' pipeline configuration (rail_project_config, data_challenge/preparation_library.yaml,
    selector `COSMOS2020`) runs `GaussianSkewtScatterSelector` with its default parameter dictionary and
    `col_name_mag_i: mag_i_lsst` on every observing-condition table, for Cardinal and Flagship alike;
    this mode checks that empirically per simulation / scenario (statistics of (z_label - z_spec)/(1+z_spec)
    per i-band bin, observed vs. `--ndraw` synthetic realisations of the same rows).
    """
    import tables_io
    from pz_pilot import log

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    if args.file:      # any file with `redshift`, `redshift_manyband`, `mag_i_lsst`, e.g. the NZ ddf_00 files (deeper spec-z coverage)
        train_file = Path(args.file) if Path(args.file).is_absolute() else base / args.file
    else:
        train_file = base / args.data_subdir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_training_{args.scenario}.hdf5"
    t = tables_io.read(str(train_file))
    zs = np.asarray(t["redshift"], float); zl = np.asarray(t["redshift_manyband"], float); mi = np.asarray(t["mag_i_lsst"], float)
    both = np.isfinite(zs) & np.isfinite(zl) & np.isfinite(mi)
    zs, zl, mi = zs[both], zl[both], mi[both]
    rng = np.random.default_rng(args.seed)
    print(f"{train_file.name}: {both.sum()} rows with spec-z and many-band label; model check with {args.ndraw} synthetic draws")
    edges = [15.0, 21.0, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5]

    def stats(d):
        return (1.4826 * np.median(np.abs(d - np.median(d))), 100 * np.mean(np.abs(d) > 0.15), np.median(d))

    print(f"{'i bin':>12} {'n':>6} | {'sNMAD obs':>9} {'model':>7} | {'outl% obs':>9} {'model':>7} | {'bias obs':>9} {'model':>8}")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (mi >= lo) & (mi < hi)
        if s.sum() < 30:
            continue
        d_obs = (zl[s] - zs[s]) / (1 + zs[s])
        sims = [stats((sample_labels(zs[s], mi[s], rng) - zs[s]) / (1 + zs[s])) for _ in range(args.ndraw)]
        m = np.mean(sims, axis=0); sd = np.std(sims, axis=0)
        o = stats(d_obs)
        flag = "  <-- differs" if (abs(o[1] - m[1]) > 3 * max(sd[1], 0.3) or abs(o[0] - m[0]) > 3 * max(sd[0], 0.001)) else ""
        print(f"{lo:5.1f}-{hi:5.1f} {s.sum():6d} | {o[0]:9.4f} {m[0]:7.4f} | {o[1]:9.2f} {m[1]:7.2f} | {o[2]:+9.4f} {m[2]:+8.4f}{flag}")
        rows.append(dict(lo=lo, hi=hi, n=int(s.sum()), obs=o, model_mean=m.tolist(), model_sd=sd.tolist()))
    print("Caveat: the overlap rows are the spectroscopically selected (bright, colour-selected) subset; at i > 24 there are few or none,")
    print("so the faint-end tail parameters cannot be checked against data here (see the docs' foutlier_vs_mag_i figure for that).")
    out = base / "pilot" / "runs" / (f"label_model_check_{train_file.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(file=train_file.name, n=int(both.sum()), rows=rows), indent=1))
    print(f"-> {out}")
    return 0


def run_synthetic(args) -> int:
    import qp
    import tables_io
    from rail.core.data import TableHandle
    from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
    from rail.utils import catalog_utils
    from pz_pilot import ZMIN, ZMAX, NZBINS, as_interp, log
    from pz_ts2_experiment import TS2_EDGES

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    tag = f"mixture_synth_ts{args.taskset}_{args.sim}_{args.scenario}" + ("_faintmode" if args.faint_mode else "") + ("_notuning" if args.no_tuning else "") + (f"_nmax{args.nmax}" if args.nmax else "") + (f"_{args.label}" if args.label else "")
    outdir = base / "pilot" / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    logf = open(outdir / "run.log", "a")
    log(f"=== manyband_mixture synthetic {tag}", logf)
    log(f"args: {vars(args)}", logf)
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)

    train_file = base / args.data_subdir / f"pz_challenge_taskset_{args.taskset}_{args.sim}_training_{args.scenario}.hdf5"
    t = {k: np.asarray(v) for k, v in tables_io.read(str(train_file)).items()}
    has_sp = np.isfinite(t["redshift"].astype(float))
    t = {k: v[has_sp] for k, v in t.items()}
    rng = np.random.default_rng(args.seed)
    n = len(t["ra"])
    if args.nmax and args.nmax < n:
        keep = np.sort(rng.choice(n, args.nmax, replace=False)); t = {k: v[keep] for k, v in t.items()}; n = len(t["ra"])
    z_true = t["redshift"].astype(float)
    mag_i = t["mag_i_lsst"].astype(float)
    force = N_MAGBIN - 1 if args.faint_mode else None
    z_lab = sample_labels(z_true, mag_i, rng, force_bin=force)
    d = (z_lab - z_true) / (1 + z_true)
    log(f"spec-z rows {n}; synthetic labels: sNMAD={1.4826*np.median(np.abs(d-np.median(d))):.4f} outliers={100*np.mean(np.abs(d)>0.15):.2f}% "
        f"({'faint-mode: i>24 parameters for all' if args.faint_mode else 'parameters by each object own i'})", logf)
    perm = rng.permutation(n); n_val = int(round(0.2 * n))
    vidx, fidx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    fit = {k: v[fidx] for k, v in t.items()}; fit["redshift"] = z_lab[fidx]
    val = {k: v[vidx] for k, v in t.items()}
    fit_path, val_path = outdir / "fit_synthetic_labels.hdf5", outdir / "valid_truth.hdf5"
    tables_io.write(fit, str(fit_path)); tables_io.write(val, str(val_path))

    catalog_utils.clear()
    catalog_utils.load_yaml(str(base / "repos" / "nz_data_challenge" / "tests" / "catalogs.yaml"))
    catalog_utils.apply(f"{args.sim}_roman_rubin")
    common = dict(hdf5_groupname="", zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS, nondetect_val=np.nan)
    model_file = outdir / "flexzboost_model.pkl"
    if args.no_tuning:
        # no bump removal, no sharpening: the estimator returns its raw conditional density,
        # which should be closest to p(z_label | x) and hence the right input for deconvolution
        informer = FlexZBoostInformer.make_stage(name=f"inform_{tag}", model=str(model_file), **common, trainfrac=1.0,
                                                 bumpmin=0.0, bumpmax=0.0, nbump=1, sharpmin=1.0, sharpmax=1.0, nsharp=1, max_basis=35)
    else:
        informer = FlexZBoostInformer.make_stage(name=f"inform_{tag}", model=str(model_file), **common, trainfrac=0.75,
                                                 bumpmin=0.02, bumpmax=0.35, nbump=args.n_bump, sharpmin=0.7, sharpmax=2.1, nsharp=args.n_sharp, max_basis=35)
    t0 = time.time(); informer.inform(TableHandle(f"fit_{tag}", path=str(fit_path))); log(f"inform done ({time.time()-t0:.1f}s)", logf)
    est = FlexZBoostEstimator.make_stage(name=f"estimate_{tag}", model=str(model_file), **common, chunk_size=20000,
                                         calculated_point_estimates=["zmode", "zmean"], qp_representation="interp")
    ens = as_interp(qp.read(est.estimate(TableHandle(f"val_{tag}", path=str(val_path))).path), logf, "synthetic validation")
    p_raw = np.asarray(ens.pdf(zgrid))
    zt, mi = val["redshift"].astype(float), val["mag_i_lsst"].astype(float)
    if args.faint_mode:
        mi = np.full_like(mi, 24.5)         # deconvolve with the same (faint) kernel the labels were drawn from
    zl_val = z_lab[vidx]

    results = {}
    log("scored against TRUE z (held-out spec-z rows):", logf)
    for n_iter in [0] + [int(x) for x in args.iters.split(",")]:
        p = p_raw if n_iter == 0 else deconvolve(p_raw, mi, zgrid, n_iter=n_iter)
        r = score(p, zgrid, zt, TS2_EDGES); results[f"iter{n_iter}"] = r
        log(f"  {'raw' if n_iter == 0 else f'deconvolved, {n_iter} it'}: {fmt(r)}", logf)
    # calibration fitted THROUGH THE KERNEL on the noisy labels of half A, scored on half B vs truth
    n_it = int(args.cal_iters)
    pd_ = deconvolve(p_raw, mi, zgrid, n_iter=n_it)
    nA = len(zt); A = np.zeros(nA, bool); A[np.random.default_rng(7).permutation(nA)[: nA // 2]] = True; B = ~A
    cal_edges = (None if args.cal_edges == "global" else tuple(float(v) for v in args.cal_edges.split(",")))
    cal = (fit_calibration_binned(pd_[A], mi[A], zgrid, zl_val[A], edges=cal_edges, model=args.cal_model) if cal_edges
           else fit_calibration_through_kernel(pd_[A], mi[A], zgrid, zl_val[A], model=args.cal_model))
    log(f"  through-kernel calibration fitted on half A (labels only): {json.dumps(cal_summary(cal))}", logf)
    for name_, pp in (("raw", p_raw[B]), (f"deconvolved {n_it} it", pd_[B]),
                      ("deconvolved + tail mixing", apply_calibration(pd_[B], zgrid, cal, with_pit_map=False, mag_i=mi[B])),
                      ("deconvolved + tail mixing + PIT map", apply_calibration(pd_[B], zgrid, cal, with_pit_map=True, mag_i=mi[B]))):
        r = score(pp, zgrid, zt[B], TS2_EDGES); results[f"halfB_{name_}"] = r
        log(f"  half B vs TRUE z, {name_}: {fmt(r)}", logf)
        for lo, hi in ((0, 23), (23, 24), (24, 99)):
            sb = (mi[B] >= lo) & (mi[B] < hi)
            if sb.sum() > 200:
                rr = score(pp[sb], zgrid, zt[B][sb], TS2_EDGES)
                log(f"      i in [{lo},{hi}) ({sb.sum()}): {fmt(rr)}", logf)
    results["calibration"] = cal_summary(cal)
    r = score(p_raw, zgrid, zl_val, TS2_EDGES); results["raw_vs_noisy_label"] = r
    log(f"  for reference, raw vs the NOISY label: {fmt(r)}", logf)
    p_best = deconvolve(p_raw, mi, zgrid, n_iter=int(args.iters.split(",")[-1]))
    r = score(smear(p_best, mi, zgrid), label_grid(zgrid), zl_val, TS2_EDGES); results["resmeared_vs_noisy_label"] = r
    log(f"  forward check, re-smeared deconvolved vs the noisy label: {fmt(r)}", logf)
    (outdir / "metrics.json").write_text(json.dumps(results, indent=1))
    log(f"metrics -> {outdir/'metrics.json'}", logf)
    return 0


# ---------------------------------------------------------------- apply to a ts3 run
def label_mix_function(run: Path, variant: str, mode: str, logf):
    """Per-object exact-label fraction f(mag_i) for the effective deconvolution kernel (ours, 2026-09-17).
    mode 'none': f = 0 (the full kernel everywhere, recipe v1/v2 behaviour); 'auto': read label_mix_<variant>.json
    written by label_mix.py (plain run) or by ts3_ensemble.py (pooled run: member fractions combined with the
    pool weights of the object's calibration bin)."""
    from pz_pilot import log
    if mode == "none":
        return lambda mi: None
    path = run / f"label_mix_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run label_mix.py (plain run) or ts3_ensemble.py with member mixes (pool) first, or use --label-mix none")
    d = json.loads(path.read_text())
    kedges = np.asarray(d["kernel_mag_edges"], float)
    assert np.allclose(kedges, M["mag_i_bin_edges"]), "label_mix kernel bins differ from the kernel's"
    if "members_f" in d:      # pooled run
        mf = np.asarray(d["members_f"], float)             # (n_members, n_kernel_bins)
        wts = np.asarray(d["weights"], float)              # (n_cal_bins, n_members)
        cedges = np.asarray(d["cal_edges"], float)
        log(f"label mix (pool): members {np.round(mf, 3).tolist()}, pool weights per calibration bin {np.round(wts, 3).tolist()}", logf)

        def f_of(mi):
            kb = mag_bin(mi)
            cb = np.clip(np.digitize(mi, cedges) - 1, 0, len(cedges) - 2)
            return np.einsum("ik,ik->i", wts[cb], mf[:, kb].T)
        return f_of
    f = np.asarray(d["f_exact_by_kernel_bin"], float)
    log(f"label mix: exact-label fraction per kernel bin {np.round(f, 3).tolist()}", logf)
    return lambda mi: f[mag_bin(mi)]


def run_apply(args) -> int:
    import h5py
    import qp
    from calibrate_pdfs import normalise
    from pz_pilot import ZMIN, ZMAX, NZBINS, log
    from pz_ts2_experiment import TS2_EDGES

    run = Path(args.run)
    logf = open(run / "run.log", "a")
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    log(f"=== manyband_mixture apply variant={args.variant} iters={args.iters} calibrate={args.calibrate} cal_model={args.cal_model} cal_edges={args.cal_edges} label_mix={args.label_mix} cal_labels={args.cal_labels} no_pit_map={args.no_pit_map} fit_half={args.fit_half} tag={args.tag!r}", logf)
    n_iter = int(args.iters.split(",")[-1])
    results = {}

    def load(path):
        with h5py.File(path) as f:
            x = f["meta/xvals"][:].ravel(); p = f["data/yvals"][:]
            anc = {k: f["ancil"][k][:] for k in f["ancil"]}
        assert np.allclose(x, zgrid)
        return normalise(np.maximum(p, 0), x), anc

    f_of = label_mix_function(run, args.variant, args.label_mix, logf)

    cal = None
    for name, fn in (("V1", f"pz_valid_{args.variant}.hdf5"), ("V2", f"pz_valid_{args.variant}_v2.hdf5")):
        path = run / fn
        if not path.exists():
            continue
        p, anc = load(path)
        mi = anc["mag_i"].ravel().astype(float); zl = anc["z_manyband"].ravel().astype(float); zs = anc["z_spec"].ravel().astype(float)
        pd = deconvolve(p, mi, zgrid, n_iter=n_iter, f_exact=f_of(mi))
        if args.calibrate:
            if cal is None:      # fit on half of V1, through the kernel, labels only
                nA = len(zl); A = np.zeros(nA, bool); A[np.random.default_rng(7).permutation(nA)[: nA // 2]] = True
                if args.fit_half == "B":
                    A = ~A       # swap-halves stability check (recipe S review 2026-09-21): fit on half B, report on half A
                rep_half = "B" if args.fit_half == "A" else "A"
                cal_edges = (None if args.cal_edges == "global" else tuple(float(v) for v in args.cal_edges.split(",")))
                zx = zs[A] if args.cal_labels == "mixed" else None
                cal = (fit_calibration_binned(pd[A], mi[A], zgrid, zl[A], edges=cal_edges, model=args.cal_model, z_exact=zx) if cal_edges
                       else fit_calibration_through_kernel(pd[A], mi[A], zgrid, zl[A], model=args.cal_model, z_exact=zx))
                calA = A
                log(f"[{args.variant}] through-kernel calibration fitted on V1 half {args.fit_half} ({A.sum()} rows): {json.dumps(cal_summary(cal))}", logf)
                import pickle      # the full calibration (parameters + PIT maps) for the packaged estimation-only path (2026-09-22)
                with open(run / f"calibration_{args.variant}{args.tag}.pkl", "wb") as fh:
                    pickle.dump(dict(edges=cal.get("edges"), cal=cal, cal_edges=cal_edges, fit_half=args.fit_half, pit_map=not args.no_pit_map), fh)
                results["calibration"] = cal_summary(cal)
            pc = apply_calibration(pd, zgrid, cal, with_pit_map=not args.no_pit_map, mag_i=mi)
            sel = ~calA if name == "V1" else np.ones(len(zl), bool)
            from calibrate_pdfs import pit_summary, pit, loglik as ll_
            calres = {}
            for lab, pp in (("deconvolved", pd), ("deconv + calibrated", pc)):
                sm = smear(pp[sel], mi[sel], zgrid); lg = label_grid(zgrid)
                ps = pit_summary(pit(sm, lg, zl[sel])); llv = ll_(sm, lg, zl[sel])
                calres[f"label_side_{lab}"] = dict(loglik=llv, pit=ps)
                log(f"[{args.variant}] {name}{f' half {rep_half}' if name == 'V1' else ''} label side through the kernel, {lab}: loglik {llv:+.3f}, PIT {ps}", logf)
            hs = np.isfinite(zs) & sel
            if hs.sum() > 50:
                calres["cal_vs_specz"] = score(pc[hs], zgrid, zs[hs], TS2_EDGES); calres["raw_vs_specz_sel"] = score(p[hs], zgrid, zs[hs], TS2_EDGES)
                log(f"[{args.variant}] {name} vs spec-z ({hs.sum()} bright rows), deconv + calibrated: {fmt(calres['cal_vs_specz'])}", logf)
                for lo, hi in ((0, 23), (23, 24), (24, 99)):
                    sb = hs & (mi >= lo) & (mi < hi)
                    if sb.sum() > 200:
                        calres[f"raw_vs_specz_i{lo}_{hi}"] = score(p[sb], zgrid, zs[sb], TS2_EDGES); calres[f"cal_vs_specz_i{lo}_{hi}"] = score(pc[sb], zgrid, zs[sb], TS2_EDGES)
                        log(f"[{args.variant}] {name} vs spec-z, i in [{lo},{hi}) ({sb.sum()}): raw {fmt(calres[f'raw_vs_specz_i{lo}_{hi}'])}", logf)
                        log(f"[{args.variant}] {name} vs spec-z, i in [{lo},{hi}) ({sb.sum()}): cal {fmt(calres[f'cal_vs_specz_i{lo}_{hi}'])}", logf)
            ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pc))
            from calibrate_pdfs import trapezoid as tz_
            a2 = dict(anc); a2["zmode"] = zgrid[np.argmax(pc, axis=1)][:, None]; a2["zmean"] = tz_(pc * zgrid, zgrid, axis=1)[:, None]
            ens.set_ancil(a2); ens.write_to(str(run / fn.replace(".hdf5", f"_deconv_cal{args.tag}.hdf5")))
        hs = np.isfinite(zs)
        r = dict(n=int(len(zl)), n_specz=int(hs.sum()))
        if args.calibrate:
            r["calibrated"] = calres
        if hs.sum() > 50:
            r["raw_vs_specz"] = score(p[hs], zgrid, zs[hs], TS2_EDGES); r["deconv_vs_specz"] = score(pd[hs], zgrid, zs[hs], TS2_EDGES)
            log(f"[{args.variant}] {name} vs spec-z ({hs.sum()} bright rows): raw  {fmt(r['raw_vs_specz'])}", logf)
            log(f"[{args.variant}] {name} vs spec-z ({hs.sum()} bright rows): decv {fmt(r['deconv_vs_specz'])}", logf)
        r["raw_vs_label"] = score(p, zgrid, zl, TS2_EDGES)
        r["resmeared_deconv_vs_label"] = score(smear(pd, mi, zgrid), label_grid(zgrid), zl, TS2_EDGES)
        r["deconv_vs_label"] = score(pd, zgrid, zl, TS2_EDGES)
        log(f"[{args.variant}] {name} vs many-band label: raw           {fmt(r['raw_vs_label'])}", logf)
        log(f"[{args.variant}] {name} vs many-band label: re-smeared dec {fmt(r['resmeared_deconv_vs_label'])}   (forward check)", logf)
        log(f"[{args.variant}] {name} vs many-band label: deconvolved   {fmt(r['deconv_vs_label'])}   (not a fair test: the label is noisy)", logf)
        # stacked n(z) per bin, raw vs deconvolved, assignment by the deconvolved PDFs
        from calibrate_pdfs import bin_by_integrated_prob, trapezoid
        b, _ = bin_by_integrated_prob(pd, zgrid, TS2_EDGES)
        rows = []
        for k in range(len(TS2_EDGES) - 1):
            s = b == k
            st_r, st_d = p[s].sum(0), pd[s].sum(0)
            m_r, m_d = trapezoid(zgrid * st_r, zgrid) / trapezoid(st_r, zgrid), trapezoid(zgrid * st_d, zgrid) / trapezoid(st_d, zgrid)
            sd_r = np.sqrt(trapezoid((zgrid - m_r) ** 2 * st_r, zgrid) / trapezoid(st_r, zgrid)); sd_d = np.sqrt(trapezoid((zgrid - m_d) ** 2 * st_d, zgrid) / trapezoid(st_d, zgrid))
            rows.append(dict(bin=k + 1, n=int(s.sum()), mean_raw=float(m_r), mean_deconv=float(m_d), std_raw=float(sd_r), std_deconv=float(sd_d),
                             mean_label=float(zl[s].mean()), std_label=float(zl[s].std())))
        r["stack_raw_vs_deconv"] = rows
        log(f"[{args.variant}] {name} stacked n(z) per bin (deconvolved assignment): " + " ".join(
            f"b{x['bin']}: mean {x['mean_raw']:.3f}->{x['mean_deconv']:.3f} (label {x['mean_label']:.3f}), std {x['std_raw']:.3f}->{x['std_deconv']:.3f} (label {x['std_label']:.3f})" for x in rows), logf)
        results[name] = r
        ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pd))
        anc2 = dict(anc); anc2["zmode"] = zgrid[np.argmax(pd, axis=1)][:, None]; anc2["zmean"] = (trapezoid(pd * zgrid, zgrid, axis=1))[:, None]
        ens.set_ancil(anc2)
        ens.write_to(str(run / fn.replace(".hdf5", f"_deconv{args.tag}.hdf5")))     # tagged too, so an --iters 0 run cannot overwrite the deconvolved files

    sub = sorted((run / f"submission_{args.variant}").glob("*.hdf5"))
    if sub and not args.skip_test:
        import tables_io
        p, anc = load(sub[0])
        # the test file's i magnitudes are needed for the kernel
        base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
        name = sub[0].name.replace("_pz_estimate_", "_test_")
        tf = base / args.data_subdir / name
        test = tables_io.read(str(tf))
        assert np.array_equal(np.asarray(test["object_id"]).astype(int), anc["object_id"].ravel().astype(int))
        mi = np.asarray(test["mag_i_lsst"], float)
        mi[~np.isfinite(mi)] = 26.0
        pd = deconvolve(p, mi, zgrid, n_iter=n_iter, f_exact=f_of(mi))
        from calibrate_pdfs import trapezoid
        for lab, pp in ((f"deconv{args.tag}", pd),) + (((f"deconv_cal{args.tag}", apply_calibration(pd, zgrid, cal, with_pit_map=not args.no_pit_map, mag_i=mi)),) if cal is not None else ()):
            ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pp))
            # The scorer takes the point estimate from ancil["zmode"] (pz_data_challenge.metrics.get_z_point), so the
            # point estimate is chosen independently of the PDF (point_estimates.py): --point "<source>_<method>",
            # source raw (the FlexZBoost PDF) or pdf (the deconvolved+calibrated PDF), method mode/mean/median/massmode*;
            # --point-faint applies a second choice to objects with i >= --point-split.
            from point_estimates import point_estimate
            def choose(spec):
                src, meth = spec.split("_", 1)
                return point_estimate(p if src == "raw" else pp, zgrid, meth)
            zmode_out = choose(args.point)
            if args.point_faint:
                faint = mi >= args.point_split
                zmode_out = np.where(faint, choose(args.point_faint), zmode_out)
            ens.set_ancil(dict(object_id=anc["object_id"], zmode=zmode_out[:, None], zmode_pdf=zgrid[np.argmax(pp, axis=1)][:, None],
                               zmode_raw=np.asarray(anc["zmode"]).reshape(-1, 1), zmean=trapezoid(pp * zgrid, zgrid, axis=1)[:, None]))
            log(f"[{args.variant}] {lab}: ancil zmode = {args.point}" + (f" (i >= {args.point_split}: {args.point_faint})" if args.point_faint else ""), logf)
            out = run / f"submission_{args.variant}_{lab}"; out.mkdir(exist_ok=True)
            ens.write_to(str(out / sub[0].name))
            log(f"[{args.variant}] {lab} test submission -> {out / sub[0].name}", logf)
    results["label_mix"] = args.label_mix; results["pit_map"] = not args.no_pit_map; results["cal_labels"] = args.cal_labels; results["fit_half"] = args.fit_half
    (run / f"mixture_{args.variant}{args.tag}.json").write_text(json.dumps(results, indent=1))
    log(f"-> {run / f'mixture_{args.variant}{args.tag}.json'}", logf)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--taskset", type=int, default=3); s.add_argument("--sim", default="cardinal"); s.add_argument("--scenario", default="1yr")
    s.add_argument("--base", default=None); s.add_argument("--data-subdir", default="data/pz/public"); s.add_argument("--nmax", type=int, default=None)
    s.add_argument("--faint-mode", action="store_true", help="draw the labels with the i>24 parameters for every object (stress test)")
    s.add_argument("--no-tuning", action="store_true", help="FlexZBoost without bump removal and sharpening (raw conditional density)")
    s.add_argument("--iters", default="5,10,20,40"); s.add_argument("--cal-iters", default="5", help="deconvolution iterations before the calibration test")
    s.add_argument("--n-bump", type=int, default=20); s.add_argument("--n-sharp", type=int, default=15)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--label", default="")
    s.add_argument("--cal-edges", default=",".join(str(v) for v in CAL_EDGES_DEFAULT), help="i-band bin edges for the calibration fit, or 'global'")
    s.add_argument("--cal-model", default="tail", choices=list(CAL_MODELS), help="calibration parameterisation (see CAL_MODELS)")
    s.add_argument("--tag", default="", help="suffix for the output files of this run (compare variants without overwriting)")
    a = sub.add_parser("apply"); a.add_argument("run"); a.add_argument("--variant", default="union_w"); a.add_argument("--iters", default="20")
    a.add_argument("--base", default=None); a.add_argument("--data-subdir", default="data/pz/public"); a.add_argument("--skip-test", action="store_true")
    a.add_argument("--calibrate", action="store_true", help="fit the tail-mixing + PIT-map calibration through the kernel on V1 half A and apply it")
    a.add_argument("--cal-edges", default=",".join(str(v) for v in CAL_EDGES_DEFAULT), help="i-band bin edges for the calibration fit, or 'global'")
    a.add_argument("--cal-model", default="tail", choices=list(CAL_MODELS), help="calibration parameterisation (see CAL_MODELS)")
    a.add_argument("--tag", default="", help="suffix for the output files of this run (compare variants without overwriting)")
    a.add_argument("--cal-labels", default="label", choices=["label", "mixed"],
                   help="label: fit the calibration on the many-band labels of V1 half A through the kernel (recipe v1/v2); "
                        "mixed: rows of half A that have a spectroscopic redshift are scored on the unsmeared PDF at that redshift instead "
                        "(identity kernel for exact labels), the others through the kernel")
    a.add_argument("--no-pit-map", action="store_true",
                   help="apply only the tail-mixing + exponent step of the calibration (fitted through the kernel) and skip the label-side "
                        "PIT map, whose transfer to true-z space is a heuristic (expert review 2026-09-17)")
    a.add_argument("--label-mix", default="none", choices=["none", "auto"],
                   help="deconvolve with the effective kernel f*identity + (1-f)*K, f = exact-label fraction of the training rows per kernel bin "
                        "(from label_mix_<variant>.json; 'none' = full kernel, recipe v1/v2)")
    a.add_argument("--fit-half", default="A", choices=["A", "B"],
                   help="V1 half the calibration is fitted on (the other half is reported); B = swap-halves stability check, use with a --tag of its own")
    a.add_argument("--point", default="raw_mode", help="ancil zmode of the test submission: <source>_<method>, source raw|pdf, method mode|mean|median|massmode03|massmode|massmode10 (see point_estimates.py)")
    a.add_argument("--point-faint", default=None, help="a second <source>_<method> for objects with i >= --point-split")
    a.add_argument("--point-split", type=float, default=24.0)
    c = sub.add_parser("check", help="compare the label model with the spec-z/many-band overlap rows of a training file")
    c.add_argument("--taskset", type=int, default=3); c.add_argument("--sim", default="cardinal"); c.add_argument("--scenario", default="1yr")
    c.add_argument("--base", default=None); c.add_argument("--data-subdir", default="data/pz/public")
    c.add_argument("--ndraw", type=int, default=50); c.add_argument("--seed", type=int, default=42)
    c.add_argument("--file", default=None, help="check this file instead (relative to the project folder), e.g. data/nz/public/nz_challenge_taskset_2_flagship_1yr_ddf_00.hdf5")
    args = ap.parse_args()
    return {"synthetic": run_synthetic, "apply": run_apply, "check": run_check}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
