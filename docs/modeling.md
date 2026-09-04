# Stage 5 — Pricing models

## Scope and data limitations

The source freMTPL2 data contains policy exposure, claims and rating factors,
but no premium charged to policyholders. All technical premiums and
profitability results in this project are constructed analytical estimates.

Claim amounts are available as policy-level aggregates. The €100,000
large-loss threshold is therefore a policy-total cap, not an individual-claim
cap. The excess is spread across the portfolio using a training-derived
loading per unit of exposure.

`claim_nb` and `n_claim_rows` disagree for some policies. Frequency and pricing
severity use the same `claim_nb` denominator so their product reconstructs
capped aggregate cost. The separate Gamma severity study uses amount-bearing
`n_claim_rows`, because a Gamma response must be strictly positive.

## Validation design

The data uses an 80/20 train/test split with seed 42. Frequency model selection
uses five-fold stratified cross-validation within the training sample. The test
sample remains untouched until final evaluation.

The frequency candidates are a constant portfolio rate, a Poisson GLM with
`log(exposure)` offset, and exposure-weighted XGBoost with a Poisson objective.
They are compared using Poisson deviance, normalized Gini, portfolio A/E and
calibration by predicted-risk decile.

## Frequency results

On the held-out test set, XGBoost achieved Poisson deviance 0.5818 and
normalized Gini 0.3311, compared with 0.6081 and 0.2526 for the Poisson GLM.
Five-fold results were consistent with the test comparison. XGBoost is the
selected frequency model; the GLM is retained as the transparent benchmark.

## Severity results

The gross and capped Gamma GLMs were fitted with a log link and claim-count
weights. Capping improved Gamma deviance from 1.7947 to 1.3829, but the capped
Gamma GLM remained slightly worse than its constant-severity baseline of
1.3709. The available rating factors provide limited stable severity
differentiation, so the selected pricing approach uses capped portfolio
severity rather than policy-level Gamma predictions.

## Pure premium and technical premium

The selected annual pure-premium rate is:

```text
XGBoost frequency × capped severity per incident + large-loss loading
```

The training-derived large-loss loading is €31.98 per policy-year of exposure.
On the held-out test set, the selected pure premium achieved Tweedie deviance
82.8245 and portfolio A/E 0.9930. It outperformed both the Poisson-GLM pure
premium (83.7752) and constant baseline (86.4963).

Constructed technical premium applies the configured 25% expense loading and
5% profit loading. The least-profitable credible one-way cohorts on the test
sample were vehicle brand B11 (377.9% loss ratio), vehicle brand B10 (162.3%),
and driver age 75–100 (146.8%). These overlapping one-way cohorts describe
associations, not causal effects.

## Production recommendation

Use XGBoost frequency with capped portfolio severity and a large-loss loading
when predictive performance is the priority. Retain the Poisson GLM for
governance, coefficient relativities and transparent challenge. Before live
use, validate stability over time, review fairness and regulatory constraints,
monitor drift and calibration, and replace constructed premiums with observed
premium data where available.
