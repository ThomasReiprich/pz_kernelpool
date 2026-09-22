# PZ task sets 3 and 4 — method description (draft for the PR, 2026-09-22)

Entry for task sets 3 and 4, both simulations, both scenarios. Code and
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
