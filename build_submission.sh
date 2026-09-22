#!/bin/bash
# Build the kernelpool submission with the shipped package = the final regeneration of the eight combinations
# (recipe S, seeded GPz, label clipping, rectangular label kernel) with exactly the code that goes into the PR. Run from
# the DESC_NZ_Challenge folder, desc-pz active, after `pip install -e submission` (once):
#   bash submission/build_submission.sh            (about 1 h 15; idempotent: a combination whose files exist is skipped)
#
# Per combination: kernel check -> three trainings (parallel) -> pool -> calibration -> submission file + model bundle,
# work directories under pilot/final/<combination>/ (kept: they hold the V1 hold-out metrics and the run logs).
# Then: one repeat training of ts3 Cardinal 1yr (retraining reproducibility: models, weights, calibration, files);
# the organizers' validator on the eight files plus our own checks (finite, non-negative, normalised PDFs, object ids
# and grid equal to the test file); the estimation-only path re-applied from each bundle and compared with the
# training-path file (must agree to floating-point precision); a comparison of the point estimates with the previous
# recipe S files (pilot/runs/ts*ens_*_E1E4gpzR if present, else _E1E4gpzS); build_info.json (environment, code
# checksums); and the release tarball submissions/kernelpool_submission.tgz (files at the top level, as the harness
# extracts them). Any failed step stops the script before the tarball (pre-submission review 2026-09-22, item 2).
set -o pipefail
export PZDC_CI_MAX_TRAIN=0          # never a subsampled production build (review item 3)
unset KERNELPOOL_FORCE_KERNEL_FAIL
OUT=submissions/kernelpool
mkdir -p $OUT/outputs_2 pilot/final logs
COMBOS="3:cardinal:1yr 3:cardinal:10yr 3:flagship:1yr 3:flagship:10yr 4:cardinal:1yr 4:cardinal:10yr 4:flagship:1yr 4:flagship:10yr"

# an existing file is reused only if its bundle is a full-size build of this package version (review item 3)
python - <<'EOF' || exit 1
import pickle, sys, os
from kernelpool import VERSION
bad = []
for c in "3:cardinal:1yr 3:cardinal:10yr 3:flagship:1yr 3:flagship:10yr 4:cardinal:1yr 4:cardinal:10yr 4:flagship:1yr 4:flagship:10yr".split():
    t, s, y = c.split(":")
    mdl = f"submissions/kernelpool/pz_challenge_taskset_{t}_{s}_pz_model_{y}.pkl"
    if not os.path.exists(mdl):
        continue
    with open(mdl, "rb") as fh:
        b = pickle.load(fh)
    if "nmax" not in b or b["nmax"] or b.get("version") != VERSION or not b.get("kernel_ok", False):
        bad.append(f"{mdl}: nmax={b.get('nmax')} version={b.get('version')} kernel_ok={b.get('kernel_ok')}")
if bad:
    print("existing files that must not be reused (subsampled, other version, or fallback build) -- move them away first:\n  " + "\n  ".join(bad))
    sys.exit(1)
EOF

train_one() {   # taskset sim scenario outdir workdir
    local t=$1 s=$2 y=$3 out=$4 wd=$5
    local est=$out/pz_challenge_taskset_${t}_${s}_pz_estimate_${y}.hdf5 mdl=$out/pz_challenge_taskset_${t}_${s}_pz_model_${y}.pkl
    if [ -f "$est" ] && [ -f "$mdl" ]; then echo "[$(date +%H:%M:%S)] exists $wd, skipped"; return 0; fi
    echo "[$(date +%H:%M:%S)] start $wd"
    if python -m kernelpool train --train-file data/pz/public/pz_challenge_taskset_${t}_${s}_training_${y}.hdf5 \
            --test-file data/pz/public/pz_challenge_taskset_${t}_${s}_test_${y}.hdf5 \
            --output $est --model-out $mdl --workdir $wd --keep-workdir > logs/build_$(basename $wd).out 2>&1; then
        echo "[$(date +%H:%M:%S)] done  $wd: $(grep -h 'label-kernel check' logs/build_$(basename $wd).out | sed 's/.*label-kernel check: //')"
    else
        echo "[$(date +%H:%M:%S)] FAILED $wd (see logs/build_$(basename $wd).out)"; return 1
    fi
}

FAILED=0
for c in $COMBOS; do
    IFS=: read t s y <<< "$c"
    train_one $t $s $y $OUT pilot/final/ts${t}_${s}_${y} || FAILED=1
done
[ $FAILED -eq 0 ] || { echo "finished with failures"; exit 1; }

echo "[$(date +%H:%M:%S)] repeat training of ts3 cardinal 1yr (retraining reproducibility)"
mkdir -p submissions/kernelpool_repeat
train_one 3 cardinal 1yr submissions/kernelpool_repeat pilot/final/ts3_cardinal_1yr_repeat || { echo "repeat training failed"; exit 1; }

echo "[$(date +%H:%M:%S)] checks: validator, PDF sanity, estimation-only re-application, repeat training, comparison with the previous files"
python - <<'EOF' 2>&1 | tee logs/build_checks.out || exit 1
import sys, glob, json, pickle, hashlib, numpy as np, h5py
sys.path.insert(0, "repos/pz_data_challenge/src")
from pz_data_challenge import submit_utils
from kernelpool import estimate_only
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
ok = True

def sane(est, test):
    """Our own checks beyond the organizers' validator: finite, non-negative, normalised PDFs; ids and grid as in the test file."""
    with h5py.File(est) as a, h5py.File(test) as t:
        y = a["data/yvals"][:]; x = a["meta/xvals"][:].ravel(); ids = a["ancil/object_id"][:].ravel(); zm = a["ancil/zmode"][:].ravel()
        tid = t["object_id"][:].ravel()
    norm = trapz(y, x, axis=1)
    checks = dict(finite=bool(np.isfinite(y).all() and np.isfinite(zm).all()), nonneg=bool((y >= 0).all()),
                  normalised=bool(np.abs(norm - 1).max() < 1e-6), ids=bool(np.array_equal(ids, tid)),
                  grid=bool(len(x) == 301 and abs(x[0]) < 1e-12 and abs(x[-1] - 3.0) < 1e-12),
                  zmode_in_grid=bool(zm.min() >= 0 and zm.max() <= 3.0))
    return checks, float(np.abs(norm - 1).max())

for c in "3:cardinal:1yr 3:cardinal:10yr 3:flagship:1yr 3:flagship:10yr 4:cardinal:1yr 4:cardinal:10yr 4:flagship:1yr 4:flagship:10yr".split():
    t, s, y = c.split(":")
    est = f"submissions/kernelpool/pz_challenge_taskset_{t}_{s}_pz_estimate_{y}.hdf5"
    mdl = f"submissions/kernelpool/pz_challenge_taskset_{t}_{s}_pz_model_{y}.pkl"
    test = f"data/pz/public/pz_challenge_taskset_{t}_{s}_test_{y}.hdf5"
    flags = submit_utils.check_pz_submission_file(est, test)
    good = flags == [1, 2, 3, 4, 5, 6, 7]; ok &= good
    checks, nerr = sane(est, test); ok &= all(checks.values())
    out2 = f"submissions/kernelpool/outputs_2/pz_challenge_taskset_{t}_{s}_pz_estimate_{y}.hdf5"
    estimate_only(mdl, test, out2)
    with h5py.File(est) as a, h5py.File(out2) as b:
        ya, yb = a["data/yvals"][:], b["data/yvals"][:]
        za, zb = a["ancil/zmode"][:].ravel(), b["ancil/zmode"][:].ravel()
        same = np.allclose(ya, yb, rtol=0, atol=1e-9) and np.allclose(za, zb, rtol=0, atol=1e-9)
        ok &= same
        dmax = float(np.abs(ya - yb).max()); zdmax = float(np.abs(za - zb).max())
    cmp = ""
    for prev in ("E1E4gpzR", "E1E4gpzS"):
        old = glob.glob(f"pilot/runs/ts{t}ens_{s}_{y}_{prev}/submission_ens_deconv_cal_nodeconv/*.hdf5")
        if old:
            with h5py.File(old[0]) as o:
                zo = o["ancil/zmode"][:].ravel(); ido = o["ancil/object_id"][:].ravel(); yo = o["data/yvals"][:]
            with h5py.File(est) as a:
                ida = a["ancil/object_id"][:].ravel()
            assert np.array_equal(ido, ida)
            d = (za - zo) / (1 + zo)
            cmp = (f"; vs previous {prev}: median |dz/(1+z)| {np.median(np.abs(d)):.4f}, 99% {np.quantile(np.abs(d), 0.99):.4f}, "
                   f"frac > 0.02: {np.mean(np.abs(d) > 0.02):.3f}, mean |dPDF| {np.abs(ya - yo).mean():.4f}")
            break
    bad = [k for k, v in checks.items() if not v]
    print(f"{'PASS' if good and not bad else 'FAIL'} ts{t} {s} {y}: validator {flags}{' sanity FAILED: ' + ','.join(bad) if bad else ''} "
          f"(max |norm-1| {nerr:.1e}); estimation-only {'reproduces' if same else 'DIFFERS FROM'} the training path "
          f"(max |dPDF| {dmax:.2e}, max |dzmode| {zdmax:.2e}){cmp}")

# repeat training of ts3 cardinal 1yr: same models / weights / calibration / files?
def h(b): return hashlib.md5(b).hexdigest()
with open("submissions/kernelpool/pz_challenge_taskset_3_cardinal_pz_model_1yr.pkl", "rb") as f1, \
     open("submissions/kernelpool_repeat/pz_challenge_taskset_3_cardinal_pz_model_1yr.pkl", "rb") as f2:
    b1, b2 = pickle.load(f1), pickle.load(f2)
models_same = [h(m1["model_bytes"]) == h(m2["model_bytes"]) for m1, m2 in zip(b1["members"], b2["members"])]   # pickle bytes can differ for incidental reasons (paths); the predictions decide
member_dmax = {}
for m in ("ts3exp_cardinal_1yr_E1_magerr", "ts3exp_cardinal_1yr_E4_magerr_reg", "ts3exp_cardinal_1yr_gpz_gpz_seeded"):
    v = "union" if "gpz" in m else "union_w"
    with h5py.File(f"pilot/final/ts3_cardinal_1yr/pilot/runs/{m}/pz_valid_{v}.hdf5") as fa, h5py.File(f"pilot/final/ts3_cardinal_1yr_repeat/pilot/runs/{m}/pz_valid_{v}.hdf5") as fb:
        member_dmax[m.split("1yr_")[1]] = float(np.abs(fa["data/yvals"][:] - fb["data/yvals"][:]).max())
w1 = np.array([b["w"] for b in b1["pool_weights"]["bins"]]); w2 = np.array([b["w"] for b in b2["pool_weights"]["bins"]])
c1 = [(c["eps"], c["sigma_t"], c["alpha"]) for c in b1["calibration"]["cal"]["bins"]]; c2 = [(c["eps"], c["sigma_t"], c["alpha"]) for c in b2["calibration"]["cal"]["bins"]]
with h5py.File("submissions/kernelpool/pz_challenge_taskset_3_cardinal_pz_estimate_1yr.hdf5") as a, \
     h5py.File("submissions/kernelpool_repeat/pz_challenge_taskset_3_cardinal_pz_estimate_1yr.hdf5") as b:
    dy = float(np.abs(a["data/yvals"][:] - b["data/yvals"][:]).max()); dz = float(np.abs(a["ancil/zmode"][:] - b["ancil/zmode"][:]).max())
    zo, zn = a["ancil/zmode"][:].ravel(), b["ancil/zmode"][:].ravel()
print(f"REPEAT ts3 cardinal 1yr: member V1 predictions max |dPDF| {member_dmax} (model bytes identical {models_same}); max |dweight| {np.abs(w1 - w2).max():.4f}; "
      f"calibration params identical {c1 == c2}; max |dPDF| {dy:.2e}; max |dzmode| {dz:.2e}; frac |dz/(1+z)| > 0.001: {np.mean(np.abs(zn - zo) / (1 + zo) > 0.001):.4f}")
print("all checks passed" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
EOF

echo "[$(date +%H:%M:%S)] build_info.json"
python - <<'EOF' || exit 1
import json, sys, platform, subprocess, hashlib, glob, time, os
import kernelpool
files = sorted(glob.glob(os.path.join(os.path.dirname(kernelpool.__file__), "**", "*.py"), recursive=True) + glob.glob(os.path.join(os.path.dirname(kernelpool.__file__), "pilot", "catalogs.yaml")))
info = dict(package="kernelpool", version=kernelpool.VERSION, built=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version, platform=platform.platform(),
            code_md5={os.path.relpath(f, os.path.dirname(os.path.dirname(kernelpool.__file__))): hashlib.md5(open(f, "rb").read()).hexdigest() for f in files},
            pip_freeze=subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines(),
            PZDC_CI_MAX_TRAIN=os.environ.get("PZDC_CI_MAX_TRAIN"))
json.dump(info, open("submissions/kernelpool/build_info.json", "w"), indent=1)
open("submission/environment_desc-pz.txt", "w").write("\n".join(info["pip_freeze"]) + "\n")
print("build_info.json written;", len(info["pip_freeze"]), "packages in the environment ->", "submission/environment_desc-pz.txt")
EOF

echo "[$(date +%H:%M:%S)] tarball"
rm -rf $OUT/outputs_2 $OUT/outputs_3
(cd $OUT && tar czf ../kernelpool_submission.tgz *.hdf5 *.pkl build_info.json) || { echo "tarball FAILED"; exit 1; }
ls -la submissions/kernelpool_submission.tgz
echo "[$(date +%H:%M:%S)] done. Upload submissions/kernelpool_submission.tgz as a GitHub release asset (see submission/PR_STEPS.md)."
