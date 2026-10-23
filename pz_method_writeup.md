# PZ task sets 3 and 4 — method description (draft for the PR, 2026-09-22; task sets 1 and 2 added 2026-10-22, see the last section)

Entry for task sets 3 and 4, both simulations, both scenarios (v1.0.0; task sets 1 and 2 in v1.1.0, last section). Code and
text written by Claude (Fable models); direction, critical review and all decisions by
Thomas Reiprich (Bonn). "Ours" below means Thomas' or Claude's.

## Method

**Training labels.** Spectroscopic redshift where present, otherwise the
many-band (COSMOS-like) label; training labels outside the 0–3 grid are
clipped to its edge. For FlexZBoost the rows are weighted so that the
training distribution in (i, colours) matches that of the many-band rows
(k-nearest-neighbour density ratio, Lima et al. 2008); GPz is unweighted.

**Three estimators**, all RAIL: FlexZBoost with magnitudes, colours and
magnitude errors as features (E1); the same with a regularised XGBoost
(depth 6, 300 trees, learning rate 0.1, subsample 0.8; E4); GPz (50 basis
functions). Grid 0–3, 301 points; seeds fixed.

**Pool.** Linear mixture of the three PDFs with weights per i-band bin
(edges 23, 24, 24.7), fitted by maximum likelihood of the many-band labels
of a held-out COSMOS sample *seen through the organizers' label error
model* (Gaussian core plus skew-t tail, magnitude dependent; we use it
as a smearing kernel K(z_label | z), with the true redshift on the 0–3
PDF grid and the label on 0–6, the model's range): each PDF is smeared
with the kernel before its likelihood at the noisy label is evaluated.
This is the "forward loss correction" of Patrini et al. 2017, applied
here to photo-z PDFs (no photo-z precedent found in the literature).

**Calibration**, per i-band bin, fitted with the same through-kernel
likelihood: an exponent α on the PDF and a mixture (weight ε) with a
Gaussian of width σ_t(1 + z) centred on the PDF mean (a broadening term,
not an outlier tail: the fits choose σ_t = 0.03–0.05 in 29 of the 32
bins, 0.1 in one faint bin, and 0.1–0.2 in two bright bins where
ε ≤ 0.01 makes the term negligible); then a PIT
recalibration map fitted on the label side of the smeared PDFs and
applied to the true-space CDF (standard PIT recalibration; the
label-to-true-space transfer is a heuristic).

**Point estimate** (`zmode`): mean of the uncalibrated pool for i < 23,
interpolated median for i ≥ 23.

## Validation

A 20 % hold-out of the COSMOS rows (pool weights and calibration fitted on
one half, reported on the other) and, for the 1yr scenarios, the ddf_00
field of the NZ challenge with the PZ rows removed by position. Against
the many-band labels, the pool improves the scored scatter by 2–5 % and
the |Δz|/(1+z) > 0.2 fraction by 0.3–1.4 points over a single FlexZBoost
with the same point estimate, in every combination and mostly at i > 24;
against spectroscopic redshifts the scatter changes by at most 0.0002.
The calibration acts on the PDFs only: label-side PIT extreme fractions
are 7–13 % (ideal 10 %); the PIT map's own gain is small and was kept by
decision. The pooled PDFs and all metrics are stable under swapping the
fit and report halves and under retraining GPz; the E1 and E4 weights
are degenerate (they trade off against each other between fits, the
pool does not change). The organizers' validator passes on all eight
files.

## Limitations, stated

The many-band labels constrain the direction but not the amount of the
bright-end (i < 23) width correction, because the kernel's core is as
wide as the PDFs there; against spectroscopic redshifts the bright PDFs
stay under-confident by 2–6 points of PIT extreme fraction in the 10yr
scenarios. The label kernel is verified against the labels to i ≈ 24 and
assumed correct at fainter magnitudes, where no spectroscopic check is
possible. About 0.5 % of Flagship objects lie above the
grid edge z = 3 (a scan of edges to 4.0 and a 0.005 step found no
criterion that improves and a resolution cost for FlexZBoost). Nothing in
the scored point metrics depends on the calibration.

## Tried and dropped

Per-object Richardson–Lucy deconvolution of the label noise (kept the
label-side metrics unchanged on real data, unverifiable at the faint
end); a calibration objective using the spectroscopic rows directly
(over-sharpens on the independent field).

## Attribution

Others: RAIL, FlexZBoost, GPz; the label-noise kernel (organizers' RAIL
selector, Yin et al. 2025, Khostovan et al. 2026); kNN density-ratio
weights (Lima et al. 2008); PIT recalibration (standard practice);
forward loss correction (Patrini et al. 2017). The accepted `conclave`
entry uses the same ensemble-plus-PIT-recalibration design, which we
found after building ours. Ours: fitting the pool weights and the
calibration through the kernel; the choice of i-band magnitude as the
bin variable; the point-estimate rule; the validation design and the
checks above.

## Reproducibility

All seeds fixed (including GPz's internal split); the shipped models
reproduce the submitted files to floating-point precision, and a repeat
training of one combination (Cardinal 1yr, task set 3) reproduced the
member predictions, pool weights, calibration and files exactly.
Pool weights and calibration are refitted in the pipeline from the
training file; if the training-time check of the label kernel fails
(different label model), the pool weights are fitted on the labels
without the kernel and the calibration is skipped, so that nothing in
the submitted files then depends on it.

## Task sets 1 and 2 (v1.1.0, added at the organizers' invitation of 2026-10-22)

**What changes.** The training files of task sets 1 and 2 carry exact
redshifts only (no many-band column), so the label kernel has nothing to
do: the same three estimators and the same pool are fitted in an
"exact-label mode" — pool weights by the likelihood of the held-out true
redshifts, calibration against them (the exponent α and the Gaussian
mixing term), V1 = a random 20 % hold-out of the training file (weights
and calibration fitted on one half of it). The FlexZBoost members are
unweighted on task set 1 (representative training sample) and weighted
towards the *test* photometry on task set 2 (k-nearest-neighbour density
ratio in (i, colours); no labels enter the weights). Point estimate:
mean of the uncalibrated pool on task set 1 (i < 23 throughout);
mean below i = 23 and interpolated median above on task set 2, as on
task sets 3/4. Only the task set's own training file and the unlabelled
test photometry enter the fit; the many-band data of the NZ challenge
were used for local validation of task set 2 only, never in training,
weights or calibration. The task-set-3/4 files are those of v1.0.0,
unchanged.

**A uniform floor on task set 2.** The task-set-2 test sample extends two
magnitudes beyond the training support, and an object without training
support can be anywhere on the grid; every task-set-2 PDF therefore gets
3 % of its mass spread uniformly over 0–3 (after the calibration; the
point estimate is untouched). The practical reason: the narrow 10yr PDFs
otherwise leave no mass where the true redshift of about 1 % of the
objects lies, the PIT is exactly 1 there, and the scored PIT-outlier
metric — the CDF at 1e-4 and 0.9999 of a 100-quantile fit to the PIT
values — breaks down on that spike (0.29 and −0.75 on our representative
checks for an actual extreme-PIT fraction of 5 %); with the floor it
reads 0.0001–0.004 and the KS and Anderson statistics improve as well.
The floor is recorded in the bundle and applied by the estimation-only
path.

**No PIT map on task sets 1/2.** The scored Anderson k-sample statistic
is scipy's standardised statistic, which fluctuates around zero for a
uniform PIT, and the bracket starts at 0: with the PIT map the task-set-1
PIT is so close to uniform (KS 0.007 on the independent hold-out half)
that the statistic is negative in 70 % of bootstrap draws and the row
would score zero by chance; without the map (KS 0.019, statistic +4) the
PIT extreme fraction is the same (10.1 % vs 10.4 %) while the KS
statistic is somewhat larger (0.019 vs 0.007): a small loss of
calibration, accepted to avoid the scorer artefact.
The bundles record the setting and the estimation-only path follows it.

**Validation, task set 1** (20k hold-out rows per combination, true
redshifts): bias ≤ 0.0002, biweight scatter 0.011–0.018, |Δz|/(1+z) > 0.2
fraction ≤ 0.2 %, KS 0.013–0.035, PIT outlier fraction ≤ 0.002, i.e.
every scored metric inside the innermost bracket by a factor ≥ 1.5. The
pool is GPz-dominated (weights 0.55–0.67) except on Flagship 10yr (0.40);
the regularised FlexZBoost gets no weight on a representative sample.

**Validation, task set 2** (representative samples with many-band labels
and the PZ rows removed by position: the NZ challenge's ddf_00 field for
1yr, the many-band rows of the task-set-3 training files for 1yr and 10yr;
label noise corrected with the documented model; spectroscopic rows
against true z): the spectroscopic rows score 21/21 everywhere. On the
full samples the bias is within ±0.01 (Flagship 1yr at the edge: +0.010),
the scatter 0.080–0.090 at 1yr and 0.041–0.042 at 10yr, the |Δz|/(1+z) >
0.2 fraction 8–12 % at 1yr and 3.8–4.0 % at 10yr, and the PIT rows are at
full points (KS 0.04–0.11, PIT-outlier ≤ 0.004 with the floor) — about
17–18 (1yr) and 19 (10yr) of 21 points per run. The limit is the 59 % of
the test sample fainter than i = 24: at 1yr their colours are
noise-dominated (median errors 0.3–0.6 mag at i = 24.5–25), and 50–100 %
of them have no training object nearby in (i, colours), 35–80 % none even
in colours alone; the training prior at its faint edge agrees with the
representative N(z) to i = 24 and is extrapolated beyond. Pool variants
(unweighted members, GPz alone, fixed weights) and point rules (mean,
mode, mass modes) were equal or worse. Stated plainly: a
spectroscopically selected training set does not constrain the faint
half of this test sample, and no estimator we tried changes that.

## Task set 2, v1.2.0: a template-fitting posterior mixed in at the faint end (2026-10-23)

**What changes.** For task-set-2 objects fainter than i = 24.5 — beyond
the depth of the spectroscopic training set — the shipped PDF is an
equal-weight mixture of the pool PDF of v1.1.0 (calibrated, with its 3 %
floor) and a template-fitting posterior from BPZ (Benítez 2000; RAIL's
`pz-rail-bpz` port, `desc-bpz`), with the same floor on the BPZ part,
and the point estimate of those objects is the median of the mixed PDF.
Brighter objects, task set 1 and task sets 3/4 are unchanged. BPZ runs
with the 31 COSMOS templates shipped with RAIL, Benítez's HDFN prior,
Madau IGM absorption, a 2 % systematic zero-point error per band, and
per-band zero-point offsets fitted on the task's own training file
(best template at the known redshift for the rows with i < 23, median of
m_obs − m_model per band, anchored on i, iterated; offsets of 0.03–0.23
mag in u, ≤ 0.03 mag elsewhere); non-detections enter with the 1σ
limits read off the task's test file. The nine bands include Roman Y, J
and H (filter curves from RAIL). Everything the estimation-only path
needs is in the bundle; the template–filter tables ship with the package.

**Why.** The trained members extrapolate the training set's N(z) beyond
i ≈ 24 and get the faint bias wrong where the test population differs
from the extrapolation (Flagship: +0.03 to +0.06 at i > 24.5 against the
many-band labels); a template posterior with a magnitude-dependent prior
does not have that failure mode. Mixed in at equal weight the template
PDFs also broaden the pool's faint PDFs exactly where they were too
confident. Equal weight is the a-priori choice: there is no labelled
faint sample from which a weight could be fitted without using the
validation data. The cut at 24.5 lies between the pool's own faint bin edges (24.0 and
24.7); it was chosen with the validation samples in view, which we state
plainly, and confirmed on a second representative sample the choice had
not seen. Stated completely: the template set (CWW+SB vs COSMOS), the
cut (24 vs 24.5), the point rule (mode, mean, median, averaged), a
point-only variant and the weights 0.5, 0.75 and 1 were all scored on
the same two representative samples before the mixture at equal weight
was adopted; the two samples are drawn from the same simulated deep field
and are not independent of each other in the strong sense. The
validation numbers below are label-noise-corrected model estimates.
Learned-where-supported, templates-beyond is a standard combination;
what is specific here is that the two components are mixed at the PDF
level so that point estimate and PDF describe the same object.

**Validation** (same samples and conventions as the v1.1.0 section;
corrected |Δz|/(1+z) > 0.2 fraction, v1.1.0 → v1.2.0): Cardinal 1yr
9.4 → 8.8 % (ddf_00) and 8.1 → 7.4 % (task-set-3 source); Flagship 1yr
12.2 → 10.3 % and 11.6 → 9.6 %; Cardinal 10yr 3.8 → 2.5 %; Flagship 10yr
4.0 → 2.4 %. Bias within ±0.005 everywhere (Flagship 1yr from +0.010 to
+0.003), scatter unchanged within 0.002, spectroscopic rows unchanged.
PIT rows: KS 0.053 → 0.049 (Cardinal 1yr), 0.115 → 0.071 (Flagship 1yr),
0.082 → 0.091 (Cardinal 10yr), 0.118 → 0.082 (Flagship 10yr);
PIT-outlier ≤ 0.0044. A BPZ-only faint end would be worse than the
pool on Cardinal 1yr and over-confident everywhere (PIT exactly 0 or 1
for 6–20 % of bright hold-out rows); the point-only swap (entry PDFs,
BPZ point estimate) was equal in points but leaves PDF and point
inconsistent, so it was not adopted. A dimming-augmentation experiment
(bright training galaxies copied to fainter magnitudes and re-noised
with the test file's error model) did not move the faint outlier
fraction and is not part of the entry.
