#!/usr/bin/env python3
"""Post-hoc calibration of photo-z PDFs, fitted on the validation split.

Problem it addresses (see pilot/README.md): FlexZBoost PDFs have hard zeros
(no tails; 6 % of true redshifts fall on zero support) and cores about twice
as wide as the actual scatter, so the PIT histogram has spikes at 0 and 1 and
a hump in the middle. Stacking such PDFs gives biased n(z) widths.

Two steps, both simple and both chosen on data the model has not seen:

  1. Tail mixing (+ optional sharpening):   p1(z) ∝ [ (1-eps) p(z)^alpha + eps T(z) ]
     with T a Gaussian centred on the PDF mean with width sigma_t (1+z),
     so every PDF has support everywhere. (eps, sigma_t, alpha) are picked
     by grid search to maximise the mean log-likelihood of the true z.
  2. PIT recalibration (Bordoloi, Lilly & Amara 2010, MNRAS 406, 881):
     p2(z) = p1(z) * h(F1(z)), where F1 is the CDF of p1 and h is the
     density of the PIT values measured on the calibration sample. If the
     PIT histogram has a hump at 0.5, h down-weights the core, etc. By
     construction the recalibrated PITs are flat on the calibration sample.

The validation split is divided 50/50: half A fits the calibration, half B
is used for the honest numbers reported. The same transformation is then
applied to the challenge test PDFs and written as a new submission file.

Usage (env desc-pz, from the DESC_NZ_Challenge folder):
  python pilot/calibrate_pdfs.py pilot/runs/taskset_1_cardinal_1yr
Options: --no-sharpen (alpha fixed at 1), --seed, --pit-bins.
Outputs in the run directory: calibration.json, pit_before_after.png,
pz_valid_calibrated.hdf5, submission_calibrated/<same file name>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
TOMO_EDGES = np.array([0.0, 0.32, 0.47, 0.61, 0.78, 2.5])   # NZ task-set-1 bins, for the n(z) check


# ----------------------------------------------------------------- basic PDF tools
def normalise(p, x):
    return p / trapezoid(p, x, axis=1)[:, None]


def cdf_grid(p, x):
    """Cumulative integral on the grid, shape like p, starts at 0 ends at 1."""
    inc = 0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(x)
    c = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)
    return c / c[:, -1:]


def eval_at(p, x, z):
    """p_i(z_i) by linear interpolation, for arrays of PDFs and one z per PDF."""
    return np.array([np.interp(zi, x, pi) for zi, pi in zip(z, p)])


def pit(p, x, z_true):
    return np.clip(eval_at(cdf_grid(p, x), x, z_true), 0.0, 1.0)


def mean_pdf(p, x):
    return trapezoid(p * x, x, axis=1)


def pit_summary(u):
    srt = np.sort(u); ecdf = np.arange(1, len(u) + 1) / len(u)
    return dict(extreme_frac=float(np.mean((u < 0.05) | (u > 0.95))),     # 0.10 if calibrated
                frac_exact_0_or_1=float(np.mean((u <= 0) | (u >= 1))),
                ks_to_uniform=float(np.max(np.abs(ecdf - srt))))


def loglik(p, x, z_true, floor=1e-6):
    return float(np.mean(np.log(np.maximum(eval_at(p, x, z_true), floor))))


# ----------------------------------------------------------------- the two steps
def mix_tails(p, x, eps, sigma_t, alpha):
    """Exponent alpha (sharpen > 1 / broaden < 1), then a mixture with a Gaussian of width sigma_t (1 + z_mean)
    CENTRED ON THE PDF MEAN, weight eps. Naming note (recipe S review, 2026-09-21): despite the function name this
    component is an outlier tail only for large sigma_t (0.2-0.4 on the grid); the fits choose sigma_t = 0.03-0.05,
    i.e. a broadening of the core around the mean, which for a bimodal PDF puts mass between the modes. Describe it as
    a mean-centred Gaussian mixing term, not as a tail."""
    q = p ** alpha if alpha != 1.0 else p
    q = normalise(q, x)
    mu = mean_pdf(q, x)
    s = sigma_t * (1.0 + mu)
    tail = np.exp(-0.5 * ((x[None, :] - mu[:, None]) / s[:, None]) ** 2)
    tail = normalise(tail, x)
    return normalise((1.0 - eps) * q + eps * tail, x)


class PITRecalibrator:
    """h(u): density of PIT values on the calibration sample, smoothed and floored."""

    def __init__(self, u, nbins=40, smooth=1, floor=0.05):
        h, e = np.histogram(np.clip(u, 0, 1), bins=nbins, range=(0, 1), density=True)
        if smooth:
            k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            hp = np.pad(h, smooth, mode="edge")
            h = np.convolve(hp, k, mode="valid")
        h = np.maximum(h, floor)
        self.centres = 0.5 * (e[1:] + e[:-1])
        self.h = h / trapezoid(np.concatenate([[h[0]], h, [h[-1]]]), np.concatenate([[0], self.centres, [1]]))

    def __call__(self, u):
        return np.interp(u, self.centres, self.h)

    def apply(self, p, x):
        F = cdf_grid(p, x)
        return normalise(p * self(F), x)


# ----------------------------------------------------------------- n(z) per tomographic bin
def bin_by_integrated_prob(p, x, edges):
    F = cdf_grid(p, x)
    P = np.stack([eval_at(F, x, np.full(len(p), hi)) - eval_at(F, x, np.full(len(p), lo))
                  for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    return P.argmax(1), P


def nz_check(p, x, z_true, edges):
    b, _ = bin_by_integrated_prob(p, x, edges)
    b_true = np.digitize(z_true, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        sel = b == k
        if sel.sum() == 0:      # empty bin (small sub-samples): record it, no NaN arithmetic
            rows.append(dict(bin=k + 1, n=0, mean_true=np.nan, mean_stack=np.nan, std_true=np.nan, std_stack=np.nan))
            continue
        st = p[sel].sum(0)
        m = trapezoid(x * st, x) / trapezoid(st, x)
        sd = np.sqrt(trapezoid((x - m) ** 2 * st, x) / trapezoid(st, x))
        rows.append(dict(bin=k + 1, n=int(sel.sum()), mean_true=float(z_true[sel].mean()), mean_stack=float(m),
                         std_true=float(z_true[sel].std()), std_stack=float(sd)))
    acc = float(np.mean(b == b_true))
    return acc, rows


def tomo_rms(rows, key="mean"):
    """rms of (stack - true) over the populated tomographic bins only."""
    d = [(r[f"{key}_stack"] - r[f"{key}_true"]) ** 2 for r in rows if r["n"] > 0]
    return float(np.sqrt(np.mean(d))) if d else np.nan


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="pilot/runs/<key> directory written by pz_pilot.py")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pit-bins", type=int, default=40)
    ap.add_argument("--no-sharpen", action="store_true")
    ap.add_argument("--eps-grid", default="0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--sigma-grid", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--alpha-grid", default="1,1.5,2,3")
    ap.add_argument("--file", default="pz_valid_with_truth.hdf5", help="validation PDF file inside the run directory (pz_ts2_experiment.py writes pz_valid_<variant>.hdf5)")
    ap.add_argument("--truth-key", default="z_true", help="ancil column with the truth (pz_ts2_experiment.py: z_manyband)")
    ap.add_argument("--edges", default="ts1", choices=["ts1", "ts2"], help="tomographic bins for the n(z) check")
    ap.add_argument("--submission", default="submission", help="submission subdirectory to calibrate as well")
    args = ap.parse_args()
    run = Path(args.run)
    edges = TOMO_EDGES if args.edges == "ts1" else np.array([0.0, 0.42, 0.64, 0.87, 1.20, 2.5])
    suffix = "" if args.file == "pz_valid_with_truth.hdf5" else "_" + Path(args.file).stem.replace("pz_valid_", "")

    with h5py.File(run / args.file) as f:
        x = f["meta/xvals"][:].ravel()
        p = f["data/yvals"][:]
        z = f["ancil/" + args.truth_key][:].ravel()
    ok = np.isfinite(z)
    if not ok.all():
        print(f"{(~ok).sum()} objects without a finite truth value dropped")
        p, z = p[ok], z[ok]
    p = normalise(np.maximum(p, 0), x)
    n = len(z)
    rng = np.random.default_rng(args.seed)
    A = np.zeros(n, bool); A[rng.permutation(n)[: n // 2]] = True
    B = ~A
    print(f"validation PDFs: {n} objects, calibration half A={A.sum()}, evaluation half B={B.sum()}")

    # --- baseline numbers on B
    base_pit = pit_summary(pit(p[B], x, z[B])); base_ll = loglik(p[B], x, z[B])
    print(f"raw          : loglik {base_ll:+.3f}  PIT extremes {100*base_pit['extreme_frac']:.1f}%  exact 0/1 {100*base_pit['frac_exact_0_or_1']:.1f}%  KS {base_pit['ks_to_uniform']:.3f}")

    # --- step 1 grid search on A (with step 2 applied in-sample so the two steps are chosen jointly)
    eps_grid = [float(v) for v in args.eps_grid.split(",")]
    sig_grid = [float(v) for v in args.sigma_grid.split(",")]
    alp_grid = [1.0] if args.no_sharpen else [float(v) for v in args.alpha_grid.split(",")]
    best = None
    for alpha in alp_grid:
        for eps in eps_grid:
            for sig in sig_grid:
                p1A = mix_tails(p[A], x, eps, sig, alpha)
                rec = PITRecalibrator(pit(p1A, x, z[A]), nbins=args.pit_bins)
                p2A = rec.apply(p1A, x)
                ll = loglik(p2A, x, z[A])
                if best is None or ll > best[0]:
                    best = (ll, eps, sig, alpha)
    ll_A, eps, sig, alpha = best
    print(f"chosen on A  : eps={eps} sigma_t={sig} alpha={alpha}  (in-sample loglik {ll_A:+.3f})")

    # --- refit the PIT map on A with the chosen parameters, evaluate on B
    p1A = mix_tails(p[A], x, eps, sig, alpha)
    rec = PITRecalibrator(pit(p1A, x, z[A]), nbins=args.pit_bins)
    p1B = mix_tails(p[B], x, eps, sig, alpha)
    p2B = rec.apply(p1B, x)
    s1 = pit_summary(pit(p1B, x, z[B])); s2 = pit_summary(pit(p2B, x, z[B]))
    ll1 = loglik(p1B, x, z[B]); ll2 = loglik(p2B, x, z[B])
    print(f"after step 1 : loglik {ll1:+.3f}  PIT extremes {100*s1['extreme_frac']:.1f}%  exact 0/1 {100*s1['frac_exact_0_or_1']:.1f}%  KS {s1['ks_to_uniform']:.3f}")
    print(f"after step 2 : loglik {ll2:+.3f}  PIT extremes {100*s2['extreme_frac']:.1f}%  exact 0/1 {100*s2['frac_exact_0_or_1']:.1f}%  KS {s2['ks_to_uniform']:.3f}   (evaluated on B, not used in the fit)")

    # --- effect on point estimates and on per-bin n(z), on B
    zm0 = mean_pdf(p[B], x); zm2 = mean_pdf(p2B, x)
    d0 = (zm0 - z[B]) / (1 + z[B]); d2 = (zm2 - z[B]) / (1 + z[B])
    nmad = lambda d: 1.4826 * np.median(np.abs(d - np.median(d)))
    print(f"zmean on B   : sigma_NMAD {nmad(d0):.4f} -> {nmad(d2):.4f}, outliers {100*np.mean(np.abs(d0)>0.15):.2f}% -> {100*np.mean(np.abs(d2)>0.15):.2f}%")
    acc0, rows0 = nz_check(p[B], x, z[B], edges)
    acc2, rows2 = nz_check(p2B, x, z[B], edges)
    print(f"bin assignment (max integrated prob) accuracy: {100*acc0:.1f}% -> {100*acc2:.1f}%")
    print("per-bin n(z) of assigned galaxies, stacked vs true  [bin: d<z> raw -> calibrated | d(std) raw -> calibrated]")
    for r0, r2 in zip(rows0, rows2):
        print(f"   bin {r0['bin']}: {r0['mean_stack']-r0['mean_true']:+.4f} -> {r2['mean_stack']-r2['mean_true']:+.4f} | {r0['std_stack']-r0['std_true']:+.4f} -> {r2['std_stack']-r2['std_true']:+.4f}")

    # --- figure: PIT before / after
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, u, t in zip(axes, [pit(p[B], x, z[B]), pit(p1B, x, z[B]), pit(p2B, x, z[B])],
                        ["raw FlexZBoost", f"+ tails (eps={eps}, sigma_t={sig}, alpha={alpha})", "+ PIT recalibration"]):
        ax.hist(u, bins=20, range=(0, 1), density=True, color="#0072B2", alpha=0.7)
        ax.axhline(1, color="k", ls="--", lw=1); ax.set_ylim(0, 2.0); ax.set_xlabel("PIT"); ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("density (half B)")
    fig.tight_layout(); fig.savefig(run / f"pit_before_after{suffix}.png", dpi=130); plt.close(fig)

    # --- persist: parameters + the PIT map, calibrated validation PDFs, calibrated test submission
    calib = dict(eps=eps, sigma_t=sig, alpha=alpha, pit_bins=args.pit_bins, seed=args.seed,
                 h_centres=rec.centres.tolist(), h_values=rec.h.tolist(),
                 metrics_B=dict(raw=dict(loglik=base_ll, **base_pit), step1=dict(loglik=ll1, **s1), step2=dict(loglik=ll2, **s2),
                                bin_accuracy_raw=acc0, bin_accuracy_cal=acc2, nz_raw=rows0, nz_cal=rows2))
    (run / f"calibration{suffix}.json").write_text(json.dumps(calib, indent=1))

    try:
        import qp
    except ImportError:
        print("qp not importable: skipping the qp output files (run inside the desc-pz env to write them)")
        return 0

    def write_ens(pdfs, ancil, path):
        ens = qp.Ensemble(qp.interp, data=dict(xvals=x, yvals=pdfs))
        ancil = dict(ancil)
        ancil["zmode"] = x[np.argmax(pdfs, axis=1)][:, None]
        ancil["zmean"] = mean_pdf(pdfs, x)[:, None]
        ens.set_ancil(ancil)
        ens.write_to(str(path))

    p2_all = rec.apply(mix_tails(p, x, eps, sig, alpha), x)
    with h5py.File(run / args.file) as f:
        anc = {k: f["ancil"][k][:][ok] for k in f["ancil"] if k not in ("zmode", "zmean")}
    write_ens(p2_all, anc, run / f"pz_valid_calibrated{suffix}.hdf5")

    subs = sorted((run / args.submission).glob("*.hdf5"))
    if subs:
        with h5py.File(subs[0]) as f:
            pt = normalise(np.maximum(f["data/yvals"][:], 0), x)
            anc_t = {k: f["ancil"][k][:] for k in f["ancil"] if k not in ("zmode", "zmean")}
        pt2 = rec.apply(mix_tails(pt, x, eps, sig, alpha), x)
        out = run / f"{args.submission}_calibrated"; out.mkdir(exist_ok=True)
        write_ens(pt2, anc_t, out / subs[0].name)
        print(f"calibrated test submission -> {out / subs[0].name}")
    print(f"calibration{suffix}.json, pit_before_after{suffix}.png, pz_valid_calibrated{suffix}.hdf5 written to {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
