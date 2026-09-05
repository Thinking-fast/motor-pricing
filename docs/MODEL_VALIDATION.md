# Model Validation Report

## Executive conclusion

XGBoost is the selected claim-frequency model because it provides the lowest
Poisson deviance and strongest risk ranking on both cross-validation and the
held-out test sample. The Poisson GLM is retained as an interpretable benchmark.

Claim severity is materially less predictable from the available rating
factors. Neither the gross nor capped Gamma GLM beats its corresponding
constant-severity baseline on Gamma deviance. The selected pure-premium method
therefore combines XGBoost frequency with portfolio capped severity and a
training-derived large-loss loading.

This implementation is suitable for portfolio demonstration and model
development. It is not approved for live quoting or customer pricing.

## Intended use

The models estimate:

1. Annual claim frequency per policy-year.
2. Expected capped claim cost.
3. Gross pure premium after restoring a portfolio large-loss loading.
4. An illustrative technical premium after expense and profit loadings.

The dashboard supports portfolio monitoring and communication to management and
technical reviewers. It does not produce legally binding insurance quotes.

## Data and sample design

- Source: freMTPL2 frequency and severity datasets from OpenML.
- Portfolio size: 678,013 policy observations.
- Split: 80% training and 20% held-out testing.
- Random seed: 42, stored in `config.yaml`.
- Cross-validation: five stratified folds within the training sample.
- Stratification: whether a policy has at least one reported claim.
- Final test sample: 135,603 policies, used only after model development.

All model candidates use the same split. Preprocessing for XGBoost is contained
inside a scikit-learn pipeline and fitted on training folds only, preventing
categorical encoding leakage.

## Frequency models

### Constant baseline

Every policy receives the training portfolio's exposure-weighted average claim
frequency. This establishes the minimum benchmark a useful model should beat.

### Poisson GLM

The response is `claim_nb`, with a Poisson family, log link and
`log(exposure)` offset. The offset converts expected claim count into an annual
claim rate while preserving the correct count likelihood. Categorical factors
are one-hot represented through formula terms; density enters as
`log(1 + density)`.

GLM coefficients can be exponentiated into multiplicative relativities, making
the model suitable for transparent challenge and actuarial interpretation.

### XGBoost

XGBoost predicts annual frequency using a Poisson objective. Policy exposure is
the sample weight. Categorical fields are one-hot encoded inside the fitted
pipeline, and numeric fields enter directly.

## Frequency validation results

### Held-out test performance

| Model | Poisson deviance ↓ | Normalized Gini ↑ | A/E |
|---|---:|---:|---:|
| Constant baseline | 0.6277 | 0.0000 | 1.0049 |
| Poisson GLM | 0.6081 | 0.2526 | 1.0059 |
| XGBoost | **0.5818** | **0.3311** | 1.0094 |

XGBoost reduces test Poisson deviance by approximately 4.3% relative to the GLM
and produces materially stronger risk ranking. Portfolio A/E remains close to
one for both models.

### Five-fold cross-validation

| Model | Mean deviance | Deviance SD | Mean Gini | Gini SD |
|---|---:|---:|---:|---:|
| Constant baseline | 0.6261 | 0.0071 | 0.0000 | 0.0000 |
| Poisson GLM | 0.6047 | 0.0074 | 0.2629 | 0.0045 |
| XGBoost | **0.5766** | 0.0081 | **0.3391** | 0.0084 |

Cross-validation and test results agree, supporting the conclusion that the
XGBoost advantage is not caused by one favourable split.

## Frequency calibration

The XGBoost frequency model separates risk meaningfully: observed frequency
rises from 0.0456 in the lowest predicted-risk decile to 0.3594 in the highest,
a ratio of approximately 7.9. Test decile A/E ranges from approximately 0.91 to
1.08, while portfolio A/E is 1.0094.

This supports good overall calibration and discrimination, with normal
cell-level variation remaining across deciles.

## Severity model

The Gamma GLM uses a log link and is fitted only to policies with positive
amount-bearing claims. Average claim amount is weighted by `n_claim_rows`.
Because Gamma responses must be positive, this severity study uses a different
claim-record basis from the frequency response and is evaluated separately.

| Loss basis | Model | Gamma deviance ↓ | Amount A/E |
|---|---|---:|---:|
| Gross | Constant baseline | **1.6341** | 0.9627 |
| Gross | Gamma GLM | 1.7947 | **0.9821** |
| Capped | Constant baseline | **1.3709** | 1.0414 |
| Capped | Gamma GLM | 1.3829 | **1.0391** |

Capping improves Gamma deviance substantially, confirming the influence of
large losses. However, the Gamma GLM does not beat the constant baseline under
either loss basis. Rating factors that successfully predict frequency do not
necessarily explain the size of a claim once it occurs.

## Large-loss treatment

The configured cap is €100,000 per policy aggregate. On the training sample:

- Excess above the cap: €9,171,403.33.
- Training exposure: 286,763.91 policy-years.
- Large-loss loading: €31.98 per policy-year.
- Capped pricing severity: €1,346.79 per reported incident.

The large-loss loading is estimated from training data only. Test-set excess is
recorded for validation but does not enter model fitting or pricing parameters.

## Pure-premium model

The selected annual pure-premium rate is:

```text
XGBoost frequency × capped severity per incident + large-loss loading
```

Frequency and capped severity both use `claim_nb`, ensuring their product
reconstructs capped cost on a consistent claim-count basis.

| Frequency component | Tweedie deviance ↓ | Claim-cost A/E |
|---|---:|---:|
| Constant baseline | 86.4963 | 0.9895 |
| Poisson GLM | 83.7752 | 0.9903 |
| XGBoost | **82.8245** | **0.9930** |

XGBoost produces the best pure-premium deviance and predicts aggregate test
claim cost within approximately 0.7%.

## Technical premium and profitability

The technical premium applies illustrative loadings of 25% for expenses and 5%
for profit/capital:

```text
technical premium = predicted claim cost × (1 + 0.25 + 0.05)
```

The source contains no premium actually charged to policyholders. Resulting
loss ratios measure performance against a constructed benchmark and must not be
described as the insurer's historical profitability.

The least-profitable credible one-way cohorts on the held-out test sample are:

1. Vehicle brand B11: 377.9% loss ratio.
2. Vehicle brand B10: 162.3% loss ratio.
3. Driver age 75–100: 146.8% loss ratio.

These factors overlap and the results are descriptive associations, not three
independent sources of loss or evidence of causation.

## Key limitations

- Premium is constructed rather than observed.
- Claim-count and amount-bearing claim-row counts disagree for 9,117 policies.
- Large-loss treatment is at policy-total rather than individual-claim level.
- The credibility threshold is pragmatic rather than formally estimated.
- The split is random, not time-based, because no suitable policy date is used.
- Hyperparameter tuning is limited; extensive searching may overfit validation
  data and should use nested or carefully controlled validation.
- Feature importance does not identify causal direction.
- The public data and illustrative model do not establish regulatory, fairness,
  operational or conduct compliance.

## Production recommendation

For this portfolio demonstration, use XGBoost frequency with capped portfolio
severity and the training-derived large-loss loading. Retain the Poisson GLM as
the transparent challenger and governance reference.

Before production use:

1. Obtain observed premium and individual-claim transaction data.
2. Resolve claim-count reconciliation issues with source-system owners.
3. Perform temporal and out-of-time validation.
4. Assess fairness, regulatory constraints and prohibited rating factors.
5. Validate data pipelines, access controls and reproducibility independently.
6. Establish monitoring thresholds for data drift, deviance, Gini and A/E.
7. Recalibrate or retrain only through an approved model-change process.

## Monitoring proposal

At least monthly, monitor:

- Policy count, exposure and missing-value rates.
- Rating-factor distributions and unseen categorical levels.
- Claim frequency, severity and pure premium.
- Overall and decile A/E.
- Poisson and Tweedie deviance when outcomes mature.
- Gini or lift stability.
- Large-loss frequency, excess amount and loading adequacy.
- Segment loss ratios and low-exposure credibility flags.

Material threshold breaches should trigger investigation before model
recalibration or replacement.
