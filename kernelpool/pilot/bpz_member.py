"""Template-fitting member for the faint end of task set 2 (kernelpool v1.2.0, ours, 2026-10-23).

Fainter than the spectroscopic training support (i >= BPZ_CUT) the shipped PDF becomes an equal-weight mixture of the
trained pool's calibrated PDF (with its uniform floor) and a template-fitting posterior from BPZ (Benitez 2000; RAIL's
`pz-rail-bpz` port by the RAIL team, `desc-bpz`), with the same floor on the BPZ part, and the point estimate of those
objects is the median of the mixed PDF. Brighter objects are untouched. Motivation and validation: pz_method_writeup.md
(v1.2.0 section) -- the trained members extrapolate the training set's N(z) beyond i ~ 24 and, on Flagship, get the
faint bias wrong; the template posterior with a magnitude-dependent prior does not have that failure mode, and the
mixture also broadens the faint PDFs where the pool's were too confident.

What enters: the task's own training file (per-band zero-point offsets: best template at the known redshift for the
rows with i < BPZ_ZP_IMAX, median of m_obs - m_model per band, anchored on i, iterated) and the task's test photometry
(1-sigma detection limits per band for the non-detections). Templates: the 31 COSMOS SEDs shipped with RAIL; prior:
Benitez's HDFN prior (RAIL's `BPZliteInformer` with output_hdfn=True); Madau IGM absorption on; a 2 % systematic
zero-point error per band. The template-filter tables (AB files) for the nine bands are shipped under bpz_data/ so that
nothing is generated or written into the installed RAIL tree at run time.

Functions used by kernelpool/pipeline.py:
    detection_limits(test_table, bands, err_bands)             -> {band: 1-sigma limit}
    hdfn_model(nt_array)                                        -> BPZ model dict (prior parameters, zero offsets)
    fit_zeropoints(train_table, model, cfg, grid, limits, workdir) -> offsets (mag per band)
    bpz_pdfs(test_path, model, cfg, grid, limits, zp, workdir, tag) -> (nobj, nz) normalised PDFs on the kernelpool grid
    mix_pdfs(p_entry, p_bpz, zgrid, mag_i, cut, weight, floor_eps)  -> mixed PDFs
    median_of_pdf(p, zgrid)
All of this is ours except BPZ itself and its templates, filters and prior.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BPZ_DATA = HERE / "bpz_data"           # SED/, FILTER/, AB/ (COSMOS templates x the nine bands, with Madau absorption)
os.environ.setdefault("BPZDATAPATH", str(BPZ_DATA))     # desc_bpz reads this at import time

BPZ_DEFAULTS = dict(cut=24.5, weight=0.5, spectra="COSMOS_seds.list", nt_array=[7, 12, 12], zp_imax=23.0, zp_err=0.02,
                    madau="yes", zp_iter=4, zp_tol=0.003)

# HDFN prior parameters (Benitez 2000) as RAIL's BPZliteInformer returns them with output_hdfn=True
_HDFN = dict(fo_arr=[0.35, 0.5], kt_arr=[0.45, 0.147], zo_arr=[0.431, 0.39, 0.0626], km_arr=[0.0913, 0.0636, 0.123],
             a_arr=[2.465, 1.806, 0.906], mo=20.0)

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def hdfn_model(nt_array, nbands: int) -> dict:
    pm = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in _HDFN.items()}
    pm["nt_array"] = list(nt_array)
    return dict(priormodel=pm, zp_offsets=dict(offsets=np.zeros(nbands)))


def band_setup():
    """Band, error, filter names and reference band from the catalog config currently applied (catalog_utils.apply)."""
    from rail.core.common_params import SHARED_PARAMS
    def sp(k):
        v = SHARED_PARAMS[k]
        return v.default if hasattr(v, "default") else v
    return list(sp("bands")), list(sp("err_bands")), list(sp("filter_list")), sp("ref_band")


def detection_limits(test: dict, bands, err_bands) -> dict:
    """1-sigma limit per band from the test photometry: the median magnitude of the detected rows whose error lies in
    0.9-1.3 mag (S/N ~ 1); fallback: the 99th percentile of the detected magnitudes. Label-free."""
    lim = {}
    for b, e in zip(bands, err_bands):
        m, em = np.asarray(test[b], float), np.asarray(test[e], float)
        ok = np.isfinite(m) & np.isfinite(em) & (em > 0.9) & (em < 1.3)
        lim[b] = float(np.median(m[ok])) if ok.sum() >= 20 else float(np.nanpercentile(m[np.isfinite(m)], 99))
    return lim


def _common(cfg: dict, grid: dict, limits: dict, bands, err_bands, filters, ref):
    return dict(hdf5_groupname="", zmin=grid["zmin"], zmax=grid["zmax"], nzbins=grid["nzbins"], nondetect_val=np.nan, mag_limits=limits,
                bands=bands, err_bands=err_bands, filter_list=filters, ref_band=ref, spectra_file=cfg["spectra"], nt_array=list(cfg["nt_array"]),
                bpz_ref_data_path=str(BPZ_DATA), zp_errors=[float(cfg["zp_err"])] * len(bands), madau_flag=cfg["madau"],
                override_file_offsets=True)


def _estimator(name: str, model: dict, cfg: dict, grid: dict, limits: dict, zp, workdir: Path, output: Path | None = None):
    from rail.estimation.algos.bpz_lite import BPZliteEstimator
    bands, err_bands, filters, ref = band_setup()
    kw = _common(cfg, grid, limits, bands, err_bands, filters, ref)
    kw["zp_offsets"] = [float(v) for v in zp]
    if output is not None:
        kw["output"] = str(output)
    return BPZliteEstimator.make_stage(name=name, model=model, chunk_size=5000, **kw)


def _as_float_table(path) -> dict:
    import tables_io
    d = {k: np.asarray(v) for k, v in tables_io.read(str(path)).items()}
    bands, err_bands, _, _ = band_setup()
    for b, e in zip(bands, err_bands):
        d[b] = d[b].astype(float); d[e] = d[e].astype(float)
    return d


def fit_zeropoints(train: dict, model: dict, cfg: dict, grid: dict, limits: dict, workdir: Path, log=print) -> np.ndarray:
    """Per-band offsets (mag, to be subtracted from the observed magnitudes) from the training rows with a redshift and
    i < zp_imax: best template at the known redshift with free amplitude, median of m_obs - m_model per band, anchored
    on the reference band, iterated until the largest increment is below zp_tol."""
    bands, err_bands, filters, ref = band_setup()
    i_idx = bands.index(ref); nf = len(bands)
    z = np.asarray(train["redshift"], float); mi = np.asarray(train[ref], float)
    sel = np.isfinite(z) & np.isfinite(mi) & (mi < cfg["zp_imax"]) & (mi > 17)
    sub = {k: np.asarray(v)[sel].copy() for k, v in train.items() if k in bands or k in err_bands}
    est = _estimator("bpz_zp", model, cfg, grid, limits, np.zeros(nf), workdir)
    est.open_model(model=model)
    est.flux_templates = est._load_templates()
    zgrid = np.linspace(grid["zmin"], grid["zmax"], grid["nzbins"])
    iz = np.clip(np.rint((z[sel] - zgrid[0]) / (zgrid[-1] - zgrid[0]) * (len(zgrid) - 1)).astype(int), 0, len(zgrid) - 1)
    T = est.flux_templates[iz]                                   # (n, nt, nf)
    zp = np.zeros(nf)
    for it in range(int(cfg["zp_iter"])):
        est.config["zp_offsets"] = [float(v) for v in zp]
        fd = est._preprocess_magnitudes({k: v.copy() for k, v in sub.items()})
        f, ef = fd["flux"], fd["flux_err"]
        good = (f > 0) & np.isfinite(f) & (ef < 1e20)
        w = np.where(good, 1 / ef ** 2, 0)[:, None, :]; fx = np.where(good, f, 0)[:, None, :]
        amp = (fx * T * w).sum(2) / np.maximum((T * T * w).sum(2), 1e-300)
        chi2 = (((fx - amp[:, :, None] * T) ** 2) * w).sum(2); tb = np.argmin(chi2, 1)
        rows = np.arange(len(tb))
        model_f = amp[rows, tb][:, None] * T[rows, tb]
        ratio = np.where(good & (model_f > 0), f / np.where(model_f > 0, model_f, 1), np.nan)
        delta = -2.5 * np.log10(np.nanmedian(ratio, 0)); delta = delta - delta[i_idx]
        rchi2 = float(np.median(chi2[rows, tb] / np.maximum(good.sum(1) - 1, 1)))
        zp = zp + delta
        log(f"bpz zero points, iteration {it}: n={int(sel.sum())} (i<{cfg['zp_imax']}), increment "
            + " ".join(f"{b.split('_')[1]}:{v:+.3f}" for b, v in zip(bands, delta)) + f", median reduced chi2 {rchi2:.2f}")
        if np.max(np.abs(delta)) < cfg["zp_tol"]:
            break
    log("bpz zero-point offsets (mag, subtracted from the observed magnitudes): " + ", ".join(f"{b.split('_')[1]} {v:+.3f}" for b, v in zip(bands, zp)))
    return zp


def bpz_pdfs(test_path, model: dict, cfg: dict, grid: dict, limits: dict, zp, workdir: Path, tag: str = "test") -> np.ndarray:
    """BPZ posteriors for every row of test_path on the kernelpool grid, normalised to unit integral; rows for which no
    template fits (all-zero posterior) get a flat PDF."""
    import qp
    from rail.core.data import TableHandle
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    est = _estimator(f"bpz_{tag}", model, cfg, grid, limits, zp, workdir, output=workdir / f"bpz_{tag}.hdf5")
    ens = qp.read(est.estimate(TableHandle(f"bpz_in_{tag}", path=str(test_path))).path)
    zgrid = np.linspace(grid["zmin"], grid["zmax"], grid["nzbins"])
    x = np.asarray(ens.metadata["xvals"]).ravel(); p = np.maximum(np.asarray(ens.objdata["yvals"]), 0)
    if not (len(x) == len(zgrid) and np.allclose(x, zgrid)):
        p = np.array([np.interp(zgrid, x, row) for row in p])
    n = _trapz(p, zgrid, axis=1)
    flat = n <= 0
    p = np.where(flat[:, None], 1.0 / (zgrid[-1] - zgrid[0]), p / np.where(n[:, None] > 0, n[:, None], 1))
    return p


def mix_pdfs(p_entry: np.ndarray, p_bpz: np.ndarray, zgrid: np.ndarray, mag_i: np.ndarray, cut: float, weight: float, floor_eps) -> np.ndarray:
    """(1 - w) p_entry + w [(1 - eps) p_bpz + eps / range] for i >= cut; p_entry (already floored) elsewhere."""
    eps = float(floor_eps or 0.0)
    pb = (1.0 - eps) * p_bpz + eps / (zgrid[-1] - zgrid[0])
    pm = (1.0 - weight) * p_entry + weight * pb
    faint = np.asarray(mag_i, float) >= cut
    return np.where(faint[:, None], pm, p_entry)


def median_of_pdf(p: np.ndarray, zgrid: np.ndarray) -> np.ndarray:
    c = np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(zgrid), axis=1)
    c = np.hstack([np.zeros((len(p), 1)), c])
    return np.array([np.interp(0.5 * ci[-1], ci, zgrid) if ci[-1] > 0 else np.nan for ci in c])
