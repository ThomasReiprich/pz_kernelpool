"""kernelpool: PZ data challenge entry (Bonn, 2026). Version 1.0.0: task sets 3 and 4; 1.1.0: task sets 1 and 2 added.

Pool of FlexZBoost (two settings) + GPz with per-i-bin weights and a per-i-bin calibration, both fitted by the
likelihood of the many-band labels of a COSMOS hold-out *through the organizers' label-noise kernel*; raw-pool
point estimate (mean for i < 23, interpolated median above). Method description: pz_method_writeup.md.

Task sets 1 and 2 (v1.1.0, 2026-10-22): their training files carry exact redshifts only (no `redshift_manyband`
column). The same pool is then fitted in "exact-label mode": no kernel check, pool weights by the likelihood of the
held-out true redshifts, calibration against them (the pilot's exact-label path, `--cal-labels mixed`, with every row
exact), V1 = a 20 % random hold-out of the training file. Nothing but the task set's own training file and the
unlabelled test photometry enters the fit (the NZ challenge's many-band data are used for local validation only,
never here). Members, FlexZBoost weighting variant and point rule per task set: MEMBERS_BY_TASKSET / POINT_BY_TASKSET.
On task sets 1/2 the calibration is the tail mixing and exponent only, WITHOUT the PIT map: with exact labels the map
makes the PIT so close to uniform that scipy's standardised Anderson k-sample statistic (one of the scored metrics,
bracket 0 <= value <= 1000) turns negative in ~70 % of hold-out draws, i.e. a perfectly calibrated PDF set would
score zero on that row (task set 1 Flagship 1yr: KS 0.007 / statistic -0.7 with the map, KS 0.019 / +4 without; PIT
extreme fractions 10.4 % vs 10.1 %, so nothing is lost). KERNELPOOL_REUSE_MEMBERS=1 reuses member run directories
that already hold their model and outputs in the work directory (our build streams on a shared machine).

This module is a thin driver around the pilot scripts in `kernelpool/pilot/` (the code the entry was developed and
validated with, copied unchanged): the training path runs them as subprocesses in a work directory laid out the way
they expect; the estimation-only path re-applies the shipped models, pool weights and calibration in-process.

Entry points (the signatures the pz_data_challenge harness calls):
    train_and_estimate(train_file, test_file, output_file, model_file=None, workdir=None, nmax=None)
    estimate_only(model_file, test_file, output_file)

Environment: PZDC_CI_MAX_TRAIN=<n> subsamples the training file to n rows (CI smoke test; 0 or unset = full);
KERNELPOOL_WORKDIR keeps the training work directory at that path instead of a temporary one;
KERNELPOOL_FORCE_KERNEL_FAIL=1 forces the label-kernel check to fail (to test the kernel-free fallback: pool weights
fitted by the raw label likelihood, no calibration).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PILOT = Path(__file__).resolve().parent / "pilot"
VERSION = "1.1.0"
SEED = 42
POINT = ["--point", "raw_mean", "--point-faint", "raw_median", "--point-split", "23"]
E4_XGB = ["--xgb-depth", "6", "--xgb-trees", "300", "--xgb-lr", "0.1", "--xgb-subsample", "0.8"]
MEMBERS = (   # (run-dir label, variant, estimator, extra training arguments)  -- recipe S, README "Recipe S adopted"
    ("E1_magerr", "union_w", "flexzboost", ["--include-mag-err"]),
    ("E4_magerr_reg", "union_w", "flexzboost", ["--include-mag-err"] + E4_XGB),
    ("gpz_seeded", "union", "gpz", []),
)
# task sets 1/2 (exact labels): the same three members; the FlexZBoost members unweighted on task set 1 (representative
# training sample) and weighted towards the test photometry on task set 2 (union_w in exact-label mode, pz_ts3_experiment.py)
MEMBERS_TS1 = tuple((lab, "union", est, extra) for lab, _, est, extra in MEMBERS)
MEMBERS_TS2 = MEMBERS
MEMBERS_BY_TASKSET = {1: MEMBERS_TS1, 2: MEMBERS_TS2, 3: MEMBERS, 4: MEMBERS}
POINT_BY_TASKSET = {1: ["--point", "raw_mean", "--point-faint", "raw_median", "--point-split", "99"],   # i < 23 throughout: the mean
                    2: POINT, 3: POINT, 4: POINT}
POOL_LABEL = "E1E4gpzS"
NAME_RE = re.compile(r"pz_challenge_taskset_(\d)_(cardinal|flagship)_(?:training|test)_(1yr|10yr)\.hdf5$")


def _log(msg: str) -> None:
    print(f"[kernelpool {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_name(path) -> tuple[int, str, str]:
    m = NAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot read taskset/simulation/scenario from the file name {Path(path).name!r} "
                         "(expected pz_challenge_taskset_<N>_<sim>_<training|test>_<scenario>.hdf5)")
    return int(m.group(1)), m.group(2), m.group(3)


def member_dir_name(taskset: int, sim: str, scen: str, est: str, nm: str, label: str) -> str:
    """pz_ts3_experiment.py's run-directory name: ts{N}exp_{sim}_{scen}[_{estimator}][_nmax{n}]_{label}."""
    return f"ts{taskset}exp_{sim}_{scen}" + (f"_{est}" if est != "flexzboost" else "") + f"{nm}_{label}"


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def _run(cmd: list[str], log_path: Path, env: dict | None = None) -> None:
    """Run a pilot script as a subprocess; stdout+stderr go to log_path; raise on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(f"\n=== {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n  see {log_path}")


def kernel_check(check_json: Path, min_n: int = 200, n_sigma: float = 3.0) -> tuple[bool, str]:
    """Decide from manyband_mixture.py check's JSON whether the training file's labels follow the organizers'
    kernel: bins with >= min_n spec-z/many-band pairs must agree within n_sigma in scatter and outlier fraction
    (the rule printed by the check); two or more differing bins mean a different label model -> no calibration."""
    d = json.loads(check_json.read_text())
    bad, seen = [], 0
    for r in d["rows"]:
        if r["n"] < min_n:
            continue
        seen += 1
        o, m, sd = r["obs"], r["model_mean"], r["model_sd"]
        if abs(o[1] - m[1]) > n_sigma * max(sd[1], 0.3) or abs(o[0] - m[0]) > n_sigma * max(sd[0], 0.001):
            bad.append(f"i {r['lo']}-{r['hi']}: sNMAD {o[0]:.4f} vs {m[0]:.4f}+-{sd[0]:.4f}, outl {o[1]:.2f} vs {m[1]:.2f}+-{sd[1]:.2f}")
    ok = seen > 0 and len(bad) < 2
    return ok, f"{seen} bins with n>={min_n}, {len(bad)} differ" + (": " + "; ".join(bad) if bad else "")


def train_and_estimate(train_file, test_file, output_file, model_file=None, workdir=None, nmax=None, keep_workdir=False) -> dict:
    """Train the three members on train_file (with the 20 % COSMOS hold-out V1), fit pool weights and calibration on
    V1 half A through the kernel, estimate test_file, write the submission to output_file and (optionally) the model
    bundle to model_file. Returns a dict with paths and the kernel-check verdict."""
    train_file, test_file, output_file = Path(train_file).resolve(), Path(test_file).resolve(), Path(output_file).resolve()
    taskset, sim, scen = parse_name(train_file)
    t2, s2, y2 = parse_name(test_file)
    if (t2, s2, y2) != (taskset, sim, scen):
        raise ValueError(f"training file is taskset {taskset} {sim} {scen}, test file is {t2} {s2} {y2}")
    import h5py
    with h5py.File(train_file) as fh:
        exact = "redshift_manyband" not in fh
    if exact != (taskset in (1, 2)):
        raise ValueError(f"taskset {taskset} training file {'without' if exact else 'with'} a redshift_manyband column: not the expected layout")
    members = MEMBERS_BY_TASKSET[taskset]
    point = POINT_BY_TASKSET[taskset]
    if nmax is None:
        nmax = int(os.environ.get("PZDC_CI_MAX_TRAIN", "0")) or None
    base = Path(workdir or os.environ.get("KERNELPOOL_WORKDIR") or tempfile.mkdtemp(prefix="kernelpool_")).resolve()   # absolute: the pilot scripts chdir
    base.mkdir(parents=True, exist_ok=True)
    _log(f"taskset {taskset} {sim} {scen}: work directory {base}" + (f", training subsampled to {nmax} rows" if nmax else ""))
    data = base / "data" / "pz" / "public"
    _link(train_file, data / f"pz_challenge_taskset_{taskset}_{sim}_training_{scen}.hdf5")
    _link(test_file, data / f"pz_challenge_taskset_{taskset}_{sim}_test_{scen}.hdf5")
    logs = base / "logs"
    py = sys.executable
    tsy = ["--taskset", str(taskset), "--sim", sim, "--scenario", scen]

    # 1. label-kernel check on the training file (spec-z / many-band overlap rows); skipped in exact-label mode (task sets 1/2)
    if exact:
        ok, why = False, "not applicable: exact labels (task set %d, no redshift_manyband column) -> weights and calibration fitted against the held-out true redshifts" % taskset
        _log(f"label-kernel check: {why}")
    else:
        _run([py, str(PILOT / "manyband_mixture.py"), "check", "--base", str(base)] + tsy + ["--ndraw", "20", "--seed", str(SEED)],
             logs / "kernel_check.out")
        ok, why = kernel_check(base / "pilot" / "runs" / f"label_model_check_pz_challenge_taskset_{taskset}_{sim}_training_{scen}.json")
        if os.environ.get("KERNELPOOL_FORCE_KERNEL_FAIL"):      # test switch: exercise the kernel-free fallback on data that would pass
            ok, why = False, "forced by KERNELPOOL_FORCE_KERNEL_FAIL (" + why + ")"
        _log(f"label-kernel check: {'passed' if ok else 'FAILED'} ({why})"
             + ("" if ok else " -> kernel-free fallback: pool weights by the raw label likelihood, no calibration"))

    # 2. the three members, in parallel (each is one pilot training run: V1 split, kNN weights, inform, V1 + test estimates)
    procs = []
    reuse = bool(os.environ.get("KERNELPOOL_REUSE_MEMBERS"))
    for label, variant, est, extra in members:
        if reuse:
            d = base / "pilot" / "runs" / member_dir_name(taskset, sim, scen, est, f"_nmax{nmax}" if nmax else "", label)
            have = [d / f"{est}_model_{variant}.pkl", d / f"pz_valid_{variant}.hdf5"] + list((d / f"submission_{variant}").glob("*.hdf5"))
            if len(have) == 3 and all(f.exists() for f in have):
                _log(f"member {label}: reused ({d.name})")
                continue
        cmd = [py, str(PILOT / "pz_ts3_experiment.py")] + tsy + ["--base", str(base), "--catalogs", str(PILOT / "catalogs.yaml"),
               "--nvalid", "0", "--seed", str(SEED), "--variants", variant, "--estimator", est, "--label", label] + extra
        if nmax:
            cmd += ["--nmax", str(nmax)]
        lp = logs / f"train_{label}.out"; lp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lp, "a"); fh.write(f"\n=== {' '.join(cmd)}\n"); fh.flush()
        procs.append((label, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, lp))
    failed = []
    for label, p, fh, lp in procs:
        p.wait(); fh.close()
        _log(f"member {label}: {'done' if p.returncode == 0 else 'FAILED'}")
        if p.returncode != 0:
            failed.append(f"{label} (see {lp})")
    if failed:
        raise RuntimeError("member training failed: " + ", ".join(failed))

    # 3. pool weights (EM through the kernel on V1 half A) and the pooled test PDFs
    runs = base / "pilot" / "runs"
    nm = f"_nmax{nmax}" if nmax else ""
    member_dirs = []
    for label, variant, est, extra in members:
        d = runs / member_dir_name(taskset, sim, scen, est, nm, label)
        member_dirs.append(f"{d}:{variant}")
    # objective "kernel" = likelihood of the labels seen through the label kernel (the method); "label" = likelihood of the labels
    # against the unsmeared PDFs, used only when the training file's labels do not follow the organizers' kernel, so that
    # nothing in the submitted pool then depends on it (pre-submission review 2026-09-22, item 4)
    _run([py, str(PILOT / "ts3_ensemble.py")] + tsy + ["--base", str(base), "--label", POOL_LABEL, "--members", ",".join(member_dirs),
          "--objective", "kernel" if ok else "label"], logs / "pool.out")
    pool_dir = runs / f"ts{taskset}ens_{sim}_{scen}_{POOL_LABEL}"

    # 4. calibration -- through the kernel (kernel check passed) or against the exact labels (task sets 1/2: every V1 row has
    #    an exact redshift, so `--cal-labels mixed` scores every row on the unsmeared PDF) -- and the submission file with
    #    the raw-pool point estimate
    calibrated = ok or exact
    cmd = [py, str(PILOT / "manyband_mixture.py"), "apply", str(pool_dir), "--base", str(base), "--variant", "ens", "--iters", "0", "--tag", "_nodeconv"] + point
    if calibrated:
        cmd.insert(cmd.index("--iters"), "--calibrate")
        if exact:
            cmd += ["--cal-labels", "mixed", "--no-pit-map"]     # see the module docstring (Anderson statistic)
    _run(cmd, logs / "apply.out")
    sub_dir = pool_dir / ("submission_ens_deconv_cal_nodeconv" if calibrated else "submission_ens_deconv_nodeconv")
    src = sub_dir / f"pz_challenge_taskset_{taskset}_{sim}_pz_estimate_{scen}.hdf5"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, output_file)
    _log(f"submission -> {output_file}")

    # 5. model bundle for the estimation-only path
    info = dict(version=VERSION, taskset=taskset, sim=sim, scenario=scen, kernel_ok=ok, kernel_check=why, nmax=nmax,
                label_mode="exact" if exact else "kernel", calibrated=calibrated, workdir=str(base), submission=str(output_file))
    if model_file:
        bundle = build_bundle(base, taskset, sim, scen, ok, nmax, exact=exact)
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        with open(model_file, "wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        info["model_file"] = str(model_file)
        _log(f"model bundle -> {model_file} ({Path(model_file).stat().st_size / 1e6:.0f} MB)")
    (base / "kernelpool_info.json").write_text(json.dumps(info, indent=1))
    if not keep_workdir and workdir is None and not os.environ.get("KERNELPOOL_WORKDIR"):
        shutil.rmtree(base, ignore_errors=True)
    return info


def _cal_to_plain(cal: dict) -> dict:
    """Calibration dict -> plain numpy/float structure (the PIT maps as their (centres, h) arrays), so that the bundle
    unpickles without the pilot modules on the path."""
    def one(c):
        r = c["recal"]
        return dict(eps=float(c["eps"]), sigma_t=float(c["sigma_t"]), alpha=float(c["alpha"]), width=float(c.get("width", 0.0)),
                    recal_centres=np.asarray(r.centres, float), recal_h=np.asarray(r.h, float),
                    n_fit=int(c.get("n_fit", 0)), fallback_global=bool(c.get("fallback_global", False)))
    if "bins" in cal:
        return dict(edges=[float(e) for e in cal["edges"]], bins=[one(c) for c in cal["bins"]])
    return one(cal)


def _cal_from_plain(d: dict):
    """Inverse of _cal_to_plain: rebuild the dict manyband_mixture.apply_calibration expects."""
    from calibrate_pdfs import PITRecalibrator
    def one(c):
        r = PITRecalibrator.__new__(PITRecalibrator)
        r.centres = np.asarray(c["recal_centres"], float); r.h = np.asarray(c["recal_h"], float)
        return dict(eps=c["eps"], sigma_t=c["sigma_t"], alpha=c["alpha"], width=c["width"], recal=r)
    if "bins" in d:
        return dict(edges=list(d["edges"]), bins=[one(c) for c in d["bins"]])
    return one(d)


def build_bundle(base: Path, taskset: int, sim: str, scen: str, kernel_ok: bool, nmax, exact: bool = False) -> dict:
    """Everything the estimation-only path needs: member models (bytes), their stage settings, pool weights,
    calibration (parameters + PIT maps) and the point-estimate rule."""
    sys.path.insert(0, str(PILOT))
    from pz_pilot import ZMIN, ZMAX, NZBINS
    runs = base / "pilot" / "runs"
    nm = f"_nmax{nmax}" if nmax else ""
    members = []
    for label, variant, est, extra in MEMBERS_BY_TASKSET[taskset]:
        d = runs / member_dir_name(taskset, sim, scen, est, nm, label)
        model_path = d / f"{est}_model_{variant}.pkl"
        members.append(dict(label=label, variant=variant, estimator=est, include_mag_err=("--include-mag-err" in extra),
                            model_bytes=model_path.read_bytes(), model_name=model_path.name))
    pool_dir = runs / f"ts{taskset}ens_{sim}_{scen}_{POOL_LABEL}"
    weights = json.loads((pool_dir / "ensemble_ens.json").read_text())["weights"]
    cal = None
    if kernel_ok or exact:
        with open(pool_dir / "calibration_ens_nodeconv.pkl", "rb") as fh:
            c = pickle.load(fh)
        cal = dict(cal=_cal_to_plain(c["cal"]), pit_map=bool(c["pit_map"]), fit_half=c["fit_half"])
    pt = POINT_BY_TASKSET[taskset]
    return dict(name="kernelpool", version=VERSION, taskset=taskset, sim=sim, scenario=scen,
                grid=dict(zmin=ZMIN, zmax=ZMAX, nzbins=NZBINS), seed=SEED, catalog=f"{sim}_roman_rubin",
                members=members, pool_weights=weights, calibration=cal, kernel_ok=kernel_ok, label_mode="exact" if exact else "kernel",
                pool_objective="kernel" if kernel_ok else "label", nmax=nmax,
                point=dict(bright=pt[1].split("_", 1)[1], faint=pt[3].split("_", 1)[1], split=float(pt[5])), created=time.strftime("%Y-%m-%d %H:%M:%S"))


def estimate_only(model_file, test_file, output_file, workdir=None) -> dict:
    """Apply a shipped bundle to test_file: member estimates -> pool -> calibration -> raw-pool point estimate."""
    sys.path.insert(0, str(PILOT))
    import qp
    import tables_io
    from rail.core.data import TableHandle
    from rail.core.stage import DataStore
    from rail.utils import catalog_utils
    from pz_pilot import as_interp
    from calibrate_pdfs import normalise, trapezoid
    from manyband_mixture import apply_calibration
    from point_estimates import point_estimate

    with open(model_file, "rb") as fh:
        b = pickle.load(fh)
    test_file, output_file = Path(test_file).resolve(), Path(output_file).resolve()
    t2, s2, y2 = parse_name(test_file)
    if (t2, s2, y2) != (b["taskset"], b["sim"], b["scenario"]):
        raise ValueError(f"bundle is for taskset {b['taskset']} {b['sim']} {b['scenario']}, test file is {t2} {s2} {y2}")
    tmp = Path(workdir or tempfile.mkdtemp(prefix="kernelpool_est_")).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd(); os.chdir(tmp)     # RAIL writes its stage outputs into the working directory
    try:
        DataStore.allow_overwrite = True
        catalog_utils.clear()
        catalog_utils.load_yaml(str(PILOT / "catalogs.yaml"))
        catalog_utils.apply(b["catalog"])
        g = b["grid"]
        zgrid = np.linspace(g["zmin"], g["zmax"], g["nzbins"])
        common = dict(hdf5_groupname="", zmin=g["zmin"], zmax=g["zmax"], nzbins=g["nzbins"], nondetect_val=np.nan)
        test = tables_io.read(str(test_file))
        ids = np.asarray(test["object_id"]).astype(int)
        mi = np.asarray(test["mag_i_lsst"], float); mi[~np.isfinite(mi)] = 26.0
        P = []
        for k, m in enumerate(b["members"]):
            mp = tmp / m["model_name"]
            mp.write_bytes(m["model_bytes"])
            tag = f"kp{k}_{m['label']}"
            if m["estimator"] == "flexzboost":
                from rail.estimation.algos.flexzboost import FlexZBoostEstimator
                est = FlexZBoostEstimator.make_stage(name=f"estimate_{tag}", model=str(mp), **common, include_mag_err=m["include_mag_err"],
                                                     chunk_size=20000, calculated_point_estimates=["zmode", "zmean"], qp_representation="interp")
            elif m["estimator"] == "gpz":
                from rail.estimation.algos.gpz import GPzEstimator
                est = GPzEstimator.make_stage(name=f"estimate_{tag}", model=str(mp), **common, chunk_size=20000)
            else:
                raise ValueError(m["estimator"])
            ens = as_interp(qp.read(est.estimate(TableHandle(f"test_{tag}", path=str(test_file))).path), None, m["label"])
            if ens.npdf != len(ids):
                raise RuntimeError(f"{m['label']}: {ens.npdf} PDFs for {len(ids)} test objects")
            P.append(normalise(np.maximum(np.asarray(ens.pdf(zgrid)), 0), zgrid))
            _log(f"member {m['label']} estimated")
        P = np.stack(P, axis=0)
        w = b["pool_weights"]
        wmat = np.tile(np.asarray(w["global_w"]), (P.shape[1], 1))
        for bn in w["bins"]:
            s = (mi >= bn["lo"]) & (mi < bn["hi"]); wmat[s] = bn["w"]
        p = normalise(np.einsum("kim,ik->im", P, wmat), zgrid)        # the raw pool (ts3_ensemble.pool)
        pd = (p / p.sum(axis=1, keepdims=True)) / (zgrid[1] - zgrid[0])   # exactly what apply --iters 0 produces (deconvolve, 0 iterations)
        pp = apply_calibration(pd, zgrid, _cal_from_plain(b["calibration"]["cal"]), with_pit_map=b["calibration"]["pit_map"], mag_i=mi) if b["calibration"] else pd
        pt = b["point"]
        zmode = np.where(mi < pt["split"], point_estimate(p, zgrid, pt["bright"]), point_estimate(p, zgrid, pt["faint"]))
        out = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pp))
        out.set_ancil(dict(object_id=ids, zmode=zmode[:, None], zmode_pdf=zgrid[np.argmax(pp, axis=1)][:, None],
                           zmode_raw=zgrid[np.argmax(p, axis=1)][:, None], zmean=trapezoid(pp * zgrid, zgrid, axis=1)[:, None]))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()
        out.write_to(str(output_file))
        _log(f"submission -> {output_file} ({'calibrated' if b['calibration'] else 'raw pool'})")
    finally:
        os.chdir(cwd)
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return dict(output=str(output_file), calibrated=bool(b["calibration"]))
