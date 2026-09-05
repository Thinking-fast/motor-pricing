# Data Quality Assessment

## Purpose

This document records material data findings, their actuarial implications and
the treatments implemented by the pipeline. Cleaning functions return copies,
log affected-row counts and preserve observations unless removal is required
for a valid rate denominator.

## Policy and claim reconciliation

The SQL extraction starts from all policy rows and left-joins claim totals.
This retains the large no-claim majority of the portfolio. Missing joined claim
amounts and claim-row counts are replaced with zero rather than `NULL`.

The current extraction contains 678,013 policy rows and 14 base columns. The
cleaning pipeline does not change this row count because no nonpositive
exposures are currently present.

## Findings and treatments

### Exposure above one

- **Finding:** 1,224 policies have exposure above one policy-year.
- **Risk:** Values above one are inconsistent with the selected annual policy
  interpretation and distort exposure-weighted rates.
- **Treatment:** Cap exposure at 1.0.
- **Residual limitation:** The source reason for these values is unknown; the
  treatment is a modelling convention rather than a source-data correction.

### Nonpositive exposure

- **Finding:** No policies in the current extracted portfolio have exposure
  less than or equal to zero.
- **Risk:** Nonpositive exposure makes annual rate calculations and the
  logarithmic Poisson offset invalid.
- **Treatment:** The pipeline removes such rows if they appear in future data
  and logs the number removed.

### Extreme bonus-malus values

- **Finding:** 209 policies have bonus-malus values above the configured cap of
  150; source values reach 230.
- **Risk:** Sparse extremes can produce unstable fitted effects and excessive
  extrapolation.
- **Treatment:** Cap `bonus_malus` at 150.
- **Residual limitation:** Capping reduces differentiation among the most
  penalised policyholders.

### Vehicle-fuel formatting

- **Finding:** Imported `veh_gas` values contained surrounding quotation marks;
  all 678,013 rows were normalised during the current run.
- **Risk:** Logically identical levels can be represented as different model
  categories.
- **Treatment:** Strip surrounding whitespace and single or double quotation
  marks. The resulting levels are `Diesel` and `Regular`.

### Claim-count mismatch

- **Finding:** 9,117 policies have `claim_nb != n_claim_rows`.
- **Risk:** `claim_nb` counts reported incidents while `n_claim_rows` counts
  claims with a recorded amount. Using one for frequency and the other for
  pricing severity prevents frequency × severity from reconstructing cost.
- **Treatment:** Retain all policies and add the boolean
  `claim_count_mismatch` flag. Frequency and pricing severity both use
  `claim_nb`. The separate reported-severity study uses `n_claim_rows` and is
  named explicitly.
- **Residual limitation:** The available data does not explain why some
  reported incidents lack corresponding amount-bearing records.

### Driver-age boundary

- **Finding:** Three drivers aged exactly 100 initially fell outside bands whose
  final boundary was 100 with right-open intervals.
- **Risk:** These policies would disappear from grouped age analysis.
- **Treatment:** Use an upper boundary of 101, creating a final interval
  `[75, 101)` labelled `75–100`.
- **Outcome:** No current policy is missing an age band.

### Large claims

- **Finding:** The portfolio includes highly influential claim totals, including
  a policy total above €4 million. At a €100,000 policy-total threshold, 42
  policies are capped and €10.65 million of excess cost is identified in the
  complete portfolio.
- **Risk:** A few large losses dominate one-way severity and destabilise Gamma
  modelling, especially in small cells.
- **Treatment:** Produce both gross and capped experience tables. For modelling,
  estimate capped cost and add a training-derived portfolio large-loss loading.
  On the modelling split, the loading is €31.98 per policy-year of exposure.
- **Residual limitation:** Claims are aggregated to policy level before this
  decision. The €100,000 threshold therefore caps each policy's total, not each
  individual claim.

### Sparse segments

- **Finding:** One region has less than the default credibility threshold of
  1,000 policy-years; other requested one-way factors exceed it.
- **Risk:** Rates in low-exposure cells may reflect random variation rather than
  stable underlying risk.
- **Treatment:** Add a `credible` flag rather than deleting the segment.
- **Residual limitation:** The threshold is a pragmatic screening rule, not a
  formal credibility-theory estimate.

## Controls and reconciliation

- Every cleaning function follows copy → count → change → log → return.
- Policies with no claim are retained with zero claim amount.
- Gross and capped pure premium reconcile to frequency × incident-basis
  severity within one cent.
- Gross portfolio cost reconciles to capped portfolio cost plus the calculated
  large-loss excess loading.
- All configurable thresholds are held in `config.yaml`.
- Unit tests cover exposure and bonus-malus caps, row removal, category
  normalisation, mismatch flags, large-loss capping and decomposition.

## Use limitation

These treatments are appropriate for a learning and portfolio analysis. Before
commercial pricing use, source-system owners should investigate the claim-count
mismatch, validate exposure definitions and replace the policy-total cap with
individual-claim large-loss treatment.
