#!/usr/bin/env python3
"""Lever 3: pool the raw PDFs of several estimator runs into one run directory.

Why: the scored point metrics (biweight scatter, |dz| > 0.2 fraction) are set by the raw
per-object PDFs; the deconvolution + calibration of manyband_mixture.py leaves the point
estimate essentially unchanged (point-estimate study, README 2026-09-16). A linear pool of
two estimators with different failure modes (FlexZBoost: XGBoost + cosine basis; GPz: sparse
Gaussian process with heteroscedastic noise) can have fewer catastrophic outliers than either.
Pooling photo-z PDFs from several codes is standard practice (e.g. Dahlen et al. 2013, ApJ 775,
93; Carrasco Kind & Brunner 2014, MNRAS 442, 3380), and the accepted ts1/ts2 entry `conclave`
pools PZFlow, GPz and FlexZBoost. What is ours here: fitting the pool weights per i-band bin
by maximum likelihood *through the many-band label kernel* (the smeared pool vs the noisy
labels, consistent with the calibration step), and the plumbing that makes the pooled run
directory a drop-in for `manyband_mixture.py apply` and `point_estimates.py`.

Members are pz_ts3_experiment.py run directories with a variant each, e.g.
    pilot/runs/ts3exp_cardinal_1yr:union_w,pilot/runs/ts3exp_cardinal_1yr_gpz:union
They must share the V1 split (same --seed and --holdout), the V2 draw and the test file;
object ids are checked (and re-ordered if only the order differs). The first member is the
reference whose validation inputs are copied into the output directory.

Weights: EM for the mixture weights (the maximum-likelihood weights of a linear pool) on V1
half A (the same half the calibration uses), per i-band bin (--bins; a bin with fewer than
--min-n label rows falls back to the global weights), objective --objective kernel (default:
each member smeared through the label kernel, likelihood of the labels) or label (raw PDFs
vs labels; noisy, for comparison only). Everything is then scored on V1 half B and V2 for
each member and the pool: label side (log-likelihood, PIT) and spec-z rows per bin
(PDF score + the organizers' point metrics for mode/mean/median). --fit-half B swaps the
halves (fit on B, report on A) for the stability check of the fitted weights; use it with a
different --variant-out (e.g. ensB) so the files do not overwrite the recipe outputs.

Outputs (in --out, default pilot/runs/ts{N}ens_{sim}_{scenario}[_{label}]):
  pz_valid_<variant-out>.hdf5, pz_valid_<variant-out>_v2.hdf5, submission_<variant-out>/,
  valid_v1_cosmos_holdout.hdf5 + valid_representative.hdf5 (copied), ensemble_<variant-out>.json, run.log

Usage (Mac, env desc-pz, from the DESC_NZ_Challenge folder; ~1 min):
  python pilot/ts3_ensemble.py --sim cardinal --scenario 1yr \
      --members pilot/runs/ts3exp_cardinal_1yr:union_w,pilot/runs/ts3exp_cardinal_1yr_gpz:union
  python pilot/manyband_mixture.py apply pilot/runs/ts3ens_cardinal_1yr --variant ens --iters 5 --calibrate \
      --point raw_mean --point-faint raw_median --point-split 23
  python pilot/point_estimates.py pilot/runs/ts3ens_cardinal_1yr --variant ens
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FLOOR = 1e-6


def em_weights(pz: np.ndarray, w0: np.ndarray | None = None, n_iter: int = 500, tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """Maximum-likelihood weights of a linear pool. pz[i, k] = p_k(z_i) for object i, member k.
    Standard EM for mixture proportions; returns (weights, mean log-likelihood)."""
    n, k = pz.shape
    pz = np.maximum(pz, FLOOR)
    w = np.full(k, 1.0 / k) if w0 is None else np.asarray(w0, float)
    ll_old = -np.inf
    for _ in range(n_iter):
        mix = pz @ w
        r = pz * w / mix[:, None]
        w = r.mean(axis=0)
        ll = float(np.mean(np.log(pz @ w)))
        if ll - ll_old < tol:
            break
        ll_old = ll
    return w, ll


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", required=True, help="comma-separated 'run_dir:variant' pairs; the first is the reference")
    ap.add_argument("--taskset", type=int, default=3); ap.add_argument("--sim", default="cardinal"); ap.add_argument("--scenario", default="1yr")
    ap.add_argument("--base", default=None); ap.add_argument("--out", default=None); ap.add_argument("--label", default="")
    ap.add_argument("--variant-out", default="ens")
    ap.add_argument("--objective", default="kernel", choices=["kernel", "label"])
    ap.add_argument("--bins", default="0,23,24,24.7,99"); ap.add_argument("--global", dest="global_only", action="store_true")
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--fixed-weights", default=None, help="skip the fit and use these weights (comma-separated, one per member)")
    ap.add_argument("--data-subdir", default="data/pz/public", help="where the test file lives (its i magnitudes pick the per-bin weights)")
    ap.add_argument("--fit-half", default="A", choices=["A", "B"],
                    help="which V1 half the weights are fitted on (the other half is reported); B = swap-halves stability check (recipe S review, 2026-09-21)")
    args = ap.parse_args()

    import h5py
    import qp
    from calibrate_pdfs import loglik, normalise, pit, pit_summary, trapezoid
    from point_estimates import organizer_point_metrics, point_estimate
    from pz_pilot import ZMIN, ZMAX, NZBINS, log
    from pz_ts2_experiment import TS2_EDGES
    import manyband_mixture as mm

    base = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    members = []
    for item in args.members.split(","):
        d, v = item.rsplit(":", 1)
        members.append((Path(d), v))
    out = Path(args.out) if args.out else base / "pilot" / "runs" / (f"ts{args.taskset}ens_{args.sim}_{args.scenario}" + (f"_{args.label}" if args.label else ""))
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")
    log(f"=== ts3_ensemble members={[f'{d.name}:{v}' for d, v in members]} objective={args.objective} bins={args.bins} global={args.global_only}", logf)
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    edges = [float(v) for v in args.bins.split(",")]
    vout = args.variant_out
    K = len(members)

    def load(path):
        with h5py.File(path) as f:
            x = f["meta/xvals"][:].ravel(); p = f["data/yvals"][:]
            anc = {k: f["ancil"][k][:] for k in f["ancil"]}
        assert np.allclose(x, zgrid), f"{path}: grid differs"
        return normalise(np.maximum(p, 0), x), anc

    def load_aligned(fn_of):
        """Load one file per member, aligned on object_id to the reference; None if the reference lacks it."""
        ref_path = fn_of(*members[0])
        if not ref_path.exists():
            return None, None
        p0, anc0 = load(ref_path)
        ids0 = anc0["object_id"].ravel().astype(np.int64)
        unique = len(np.unique(ids0)) == len(ids0)     # Flagship object ids are not unique; rows then align by order only
        if not unique:
            log(f"{ref_path}: {len(ids0) - len(np.unique(ids0))} duplicate object ids, members must match row by row", logf)
        ps = [p0]
        for d, v in members[1:]:
            path = fn_of(d, v)
            if not path.exists():
                raise FileNotFoundError(path)
            p, anc = load(path)
            ids = anc["object_id"].ravel().astype(np.int64)
            if np.array_equal(ids, ids0):
                ps.append(p)
            elif unique and len(ids) == len(ids0) and np.array_equal(np.sort(ids), np.sort(ids0)):
                order = np.argsort(ids)[np.searchsorted(ids[np.argsort(ids)], ids0)]
                assert np.array_equal(ids[order], ids0)
                ps.append(p[order]); log(f"{path}: same objects, re-ordered to the reference", logf)
            else:
                raise ValueError(f"{path}: object ids differ from {ref_path} ({len(ids)} vs {len(ids0)} rows, "
                                 f"{len(np.intersect1d(ids, ids0))} in common) -- the members must share the V1 split / V2 draw / test file")
        return np.stack(ps, axis=0), anc0     # (K, n, m)

    P1, anc1 = load_aligned(lambda d, v: d / f"pz_valid_{v}.hdf5")
    P2, anc2 = load_aligned(lambda d, v: d / f"pz_valid_{v}_v2.hdf5")
    subs = sorted((members[0][0] / f"submission_{members[0][1]}").glob("*.hdf5"))
    PT, ancT = (load_aligned(lambda d, v, name=subs[0].name: d / f"submission_{v}" / name) if subs else (None, None))
    if P1 is None:
        raise FileNotFoundError(members[0][0] / f"pz_valid_{members[0][1]}.hdf5")
    log(f"V1: {P1.shape[1]} rows; V2: {P2.shape[1] if P2 is not None else 0}; test: {PT.shape[1] if PT is not None else 0}", logf)

    mi1 = anc1["mag_i"].ravel().astype(float); zl1 = anc1["z_manyband"].ravel().astype(float); zs1 = anc1["z_spec"].ravel().astype(float)
    n1 = len(zl1); A = np.zeros(n1, bool); A[np.random.default_rng(7).permutation(n1)[: n1 // 2]] = True; B = ~A   # same halves as the calibration
    if args.fit_half == "B":
        A, B = B, A          # fit on half B, report on half A (same permutation, so the halves are exactly swapped)
    rep_name = "B" if args.fit_half == "A" else "A"
    log(f"weights fitted on V1 half {args.fit_half} ({A.sum()} rows), reported on half {rep_name} ({B.sum()} rows)", logf)

    # --- likelihood matrix of the members on half A: p_k(z_label) (smeared or raw)
    lgrid = mm.label_grid(zgrid)      # smeared PDFs live on the label grid (0-6); raw PDFs on zgrid

    def member_lik(P, mi, zl, sel):
        cols = []
        for k in range(K):
            pk, g = (mm.smear(P[k][sel], mi[sel], zgrid), lgrid) if args.objective == "kernel" else (P[k][sel], zgrid)
            cols.append(np.array([np.interp(z, g, row) for z, row in zip(zl[sel], pk)]))
        return np.stack(cols, axis=1)

    likA = member_lik(P1, mi1, zl1, A)
    w_global, ll_global = em_weights(likA)
    log(f"global weights (half {args.fit_half}, {A.sum()} rows, {args.objective}): {np.round(w_global, 3).tolist()}  <logp>={ll_global:+.4f}  "
        f"(single members: {[round(float(np.mean(np.log(np.maximum(likA[:, k], FLOOR)))), 4) for k in range(K)]})", logf)
    weights = dict(edges=edges, global_w=w_global.tolist(), global_loglik=ll_global, bins=[])
    if args.fixed_weights:
        wf = np.array([float(v) for v in args.fixed_weights.split(",")]); wf = wf / wf.sum()
        assert len(wf) == K
        log(f"using fixed weights {wf.round(3).tolist()} in every bin", logf)
    miA = mi1[A]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (miA >= lo) & (miA < hi)
        if args.fixed_weights:
            wb, llb, fb = wf, float(np.mean(np.log(np.maximum(likA[s] @ wf, FLOOR)))) if s.any() else float("nan"), False
        elif args.global_only or s.sum() < args.min_n:
            wb, llb, fb = w_global, float(np.mean(np.log(np.maximum(likA[s] @ w_global, FLOOR)))) if s.any() else float("nan"), True
        else:
            wb, llb = em_weights(likA[s]); fb = False
        weights["bins"].append(dict(lo=lo, hi=hi, n_fit=int(s.sum()), w=np.asarray(wb).tolist(), loglik_fit=llb, fallback_global=fb))
        log(f"  bin i[{lo},{hi}) n={s.sum()}: w={np.round(wb, 3).tolist()} <logp>={llb:+.4f}{' (global fallback)' if fb else ''}", logf)

    def pool(P, mi):
        wmat = np.tile(np.asarray(weights["global_w"]), (P.shape[1], 1))     # objects outside every bin (NaN i) keep the global weights
        for b in weights["bins"]:
            s = (mi >= b["lo"]) & (mi < b["hi"]); wmat[s] = b["w"]
        return normalise(np.einsum("kim,ik->im", P, wmat), zgrid)

    # --- scoring: each member and the pool, V1 half B and V2
    def evaluate(P, mi, zl, zs, sel, name):
        res = {}
        cands = {f"member{k}_{members[k][0].name}:{members[k][1]}": P[k] for k in range(K)}
        cands["pool"] = pool(P, mi)
        for cname, p in cands.items():
            r = {}
            for lo, hi in list(zip(edges[:-1], edges[1:])) + [(0.0, 99.0)]:
                s = sel & (mi >= lo) & (mi < hi)
                if s.sum() < 100:
                    continue
                bkey = "all" if (lo, hi) == (0.0, 99.0) else f"i{lo}_{hi}"
                sm = mm.smear(p[s], mi[s], zgrid)
                r[bkey] = dict(n=int(s.sum()), label_side=dict(loglik_raw=loglik(p[s], zgrid, zl[s]), loglik_smeared=loglik(sm, lgrid, zl[s]),
                                                              pit_smeared=pit_summary(pit(sm, lgrid, zl[s]))))
                line = (f"[{name}] {cname:>40s} {bkey:>9s} (n={s.sum()}): label side <logp> raw {r[bkey]['label_side']['loglik_raw']:+.3f} "
                        f"smeared {r[bkey]['label_side']['loglik_smeared']:+.3f} PITextr {100*r[bkey]['label_side']['pit_smeared']['extreme_frac']:.1f}%")
                hs = s & np.isfinite(zs)
                if hs.sum() > 200:
                    r[bkey]["n_specz"] = int(hs.sum()); r[bkey]["vs_specz"] = mm.score(p[hs], zgrid, zs[hs], TS2_EDGES)
                    r[bkey]["vs_specz_points"] = {m: organizer_point_metrics(zs[hs], point_estimate(p[hs], zgrid, m)) for m in ("mode", "mean", "median")}
                    pts = r[bkey]["vs_specz_points"]
                    line += (f" | spec-z ({hs.sum()}): {mm.fmt(r[bkey]['vs_specz'])} | scored std/abs>0.2: "
                             + " ".join(f"{m} {pts[m]['std']:.4f}/{100*pts[m]['abs_outlier_rate']:.1f}%" for m in pts))
                log(line, logf)
            res[cname] = r
        return res, cands["pool"]

    results = dict(members=[f"{d}:{v}" for d, v in members], objective=args.objective, fit_half=args.fit_half, weights=weights)
    results[f"V1_half{rep_name}"], pool1 = evaluate(P1, mi1, zl1, zs1, B, f"V1 half {rep_name}")
    if P2 is not None:
        mi2 = anc2["mag_i"].ravel().astype(float); zl2 = anc2["z_manyband"].ravel().astype(float); zs2 = anc2["z_spec"].ravel().astype(float)
        results["V2"], pool2 = evaluate(P2, mi2, zl2, zs2, np.ones(len(zl2), bool), "V2")

    # --- write the pooled run directory
    def write(p, anc, path):
        ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=p))
        a = {k: v for k, v in anc.items() if k not in ("zmode", "zmean", "distribution_type", "id")}
        a["zmode"] = zgrid[np.argmax(p, axis=1)][:, None]; a["zmean"] = trapezoid(p * zgrid, zgrid, axis=1)[:, None]
        ens.set_ancil(a); ens.write_to(str(path))
        log(f"-> {path}", logf)

    write(pool1, anc1, out / f"pz_valid_{vout}.hdf5")
    if P2 is not None:
        write(pool2, anc2, out / f"pz_valid_{vout}_v2.hdf5")
    if PT is not None:
        import tables_io
        name = subs[0].name
        tf = base / args.data_subdir / name.replace("_pz_estimate_", "_test_")
        test = tables_io.read(str(tf))
        assert np.array_equal(np.asarray(test["object_id"]).astype(np.int64), ancT["object_id"].ravel().astype(np.int64)), "test ids differ from the test file"
        miT = np.asarray(test["mag_i_lsst"], float); miT[~np.isfinite(miT)] = 26.0
        poolT = pool(PT, miT)
        (out / f"submission_{vout}").mkdir(exist_ok=True)
        write(poolT, dict(object_id=ancT["object_id"]), out / f"submission_{vout}" / name)
    for fn in ("valid_v1_cosmos_holdout.hdf5", "valid_representative.hdf5"):
        src = members[0][0] / fn
        if src.exists() and not (out / fn).exists():
            shutil.copy(src, out / fn)
    # label mix of the pool (expert review 2026-09-17, finding 1): the members' exact-label fractions per kernel
    # bin, combined at apply time with the pool weights of the object's calibration bin
    mixes = [d / f"label_mix_{v}.json" for d, v in members]
    if all(m.exists() for m in mixes):
        md = [json.loads(m.read_text()) for m in mixes]
        (out / f"label_mix_{vout}.json").write_text(json.dumps(dict(
            members=[f"{d}:{v}" for d, v in members], kernel_mag_edges=md[0]["kernel_mag_edges"],
            members_f=[x["f_exact_by_kernel_bin"] for x in md], cal_edges=edges,
            weights=[b["w"] for b in weights["bins"]]), indent=1))
        log(f"-> {out / f'label_mix_{vout}.json'} (member exact-label fractions {[np.round(x['f_exact_by_kernel_bin'], 3).tolist() for x in md]})", logf)
    else:
        log(f"no label_mix_<variant>.json for {[str(m) for m in mixes if not m.exists()]}; run label_mix.py on the members to enable --label-mix auto", logf)
    (out / f"ensemble_{vout}.json").write_text(json.dumps(results, indent=1, default=float))
    log(f"-> {out / f'ensemble_{vout}.json'}", logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
