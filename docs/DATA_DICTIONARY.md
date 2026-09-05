# Data Dictionary

## Purpose

This document defines the source, analysis and model-output fields used by the
motor pricing platform. The source is the public French motor third-party
liability dataset freMTPL2, downloaded from OpenML.

## Database tables

### `policies`

One row represents one insurance policy observation.

| Field | Type | Definition | Project treatment |
|---|---|---|---|
| `policy_id` | Integer | Unique policy identifier (`IDpol` in the source). | Primary key and join key; not used as a model feature. |
| `exposure` | Float | Fraction of a policy-year during which the policy was at risk. | Capped at 1.0; nonpositive observations are removed. Used as the Poisson offset and metric weight. |
| `claim_nb` | Integer | Number of reported claim incidents for the policy. | Frequency target and denominator for pricing severity. |
| `area` | Category | Population-density area class, coded A–F. | Used as a categorical rating factor. |
| `veh_power` | Integer | Vehicle power category. | Used as a numeric model feature. |
| `veh_age` | Integer | Vehicle age in years. | Used as a numeric model feature. |
| `driv_age` | Integer | Driver age in years. | Converted into configured age bands for analysis and modelling. |
| `bonus_malus` | Integer | French bonus-malus coefficient; values above 100 indicate a penalty. | Capped at 150 and used as a numeric model feature. |
| `veh_brand` | Category | Anonymised vehicle-brand group. | Used as a categorical rating factor. |
| `veh_gas` | Category | Vehicle fuel type (`Diesel` or `Regular`). | Surrounding whitespace and quotation marks are removed. |
| `density` | Integer | Inhabitants per square kilometre in the policyholder's city. | Used directly by XGBoost and as `log(1 + density)` in the GLM. |
| `region` | Category | Anonymised French region code. | Foreign key to `regions` and categorical rating factor. |

### `claims`

One row represents one amount-bearing claim record.

| Field | Type | Definition | Project treatment |
|---|---|---|---|
| `claim_id` | Integer | Surrogate claim-record identifier created during loading. | Primary key; not used as a model feature. |
| `policy_id` | Integer | Policy associated with the claim record. | Foreign key to `policies`. |
| `claim_amount` | Float | Recorded claim cost. | Aggregated to policy level before analysis. |

### `regions`

One row represents one region code.

| Field | Type | Definition | Project treatment |
|---|---|---|---|
| `region` | Category | Region code appearing in the policy data. | Primary key. |
| `region_name` | Text | Human-readable name where a mapping is available. | Unmapped codes retain their source code. |
| `macro_area` | Text | Higher-level geographic grouping. | Currently set to `France`; retained for future extension. |

## Analysis base table

The base table is created with a left join from policies to claim aggregates,
preserving policies with no claim records.

| Field | Type | Definition | Derivation |
|---|---|---|---|
| `total_claim_amount` | Float | Total recorded claim cost for a policy. | Sum of `claim_amount`; replaced with zero when no claim row exists. |
| `n_claim_rows` | Integer | Number of amount-bearing claim rows associated with a policy. | Count of rows in `claims`; replaced with zero when absent. |
| `claim_count_mismatch` | Boolean | Whether `claim_nb` differs from `n_claim_rows`. | Added during cleaning; flagged rather than filtered. |
| `age_band` | Ordered category | Driver-age interval used for one-way analysis and modelling. | Bands: 18–24, 25–34, 35–44, 45–54, 55–64, 65–74 and 75–100. Intervals are left-closed and right-open. |

## Experience-study measures

All ratios are calculated after aggregating their numerators and denominators.

| Measure | Definition |
|---|---|
| `policies` | Number of policy rows in the segment. |
| `exposure` | Sum of policy exposure in the segment. |
| `claim_nb` | Sum of reported claim incidents. |
| `n_claims_with_amount` | Sum of amount-bearing claim rows. |
| `claim_amount` | Sum of policy claim amounts. |
| `frequency` | `claim_nb / exposure`. |
| `severity` | `claim_amount / claim_nb`; uses the incident-count basis. |
| `severity_reported` | `claim_amount / n_claims_with_amount`; average size of amount-bearing claim records. |
| `pure_premium` | `claim_amount / exposure`, before expenses and profit. |
| `credible` | Whether segment exposure is at least 1,000 policy-years. |

## Pricing and model outputs

| Field | Definition |
|---|---|
| `predicted_frequency` | Predicted annual claim incidents per policy-year. |
| `predicted_claim_count` | `predicted_frequency × exposure`. |
| `predicted_severity` | Gamma GLM prediction of average amount-bearing claim size. |
| `predicted_capped_loss_rate` | Predicted frequency multiplied by capped pricing severity. |
| `large_loss_loading` | Training-set claim cost above the policy-total cap, divided by training exposure. |
| `predicted_pure_premium_rate` | `predicted_capped_loss_rate + large_loss_loading`. |
| `predicted_claim_cost` | `predicted_pure_premium_rate × exposure`. |
| `technical_premium_rate` | Pure-premium rate after configured expense and profit loadings. |
| `technical_premium` | `technical_premium_rate × exposure`. This is constructed, not observed. |
| `loss_ratio` | Actual claim cost divided by constructed technical premium. |
| `underwriting_result` | Constructed technical premium minus actual claim cost. |
| `actual_to_expected` | Total actual outcome divided by total model-predicted outcome. |

## Units and missing values

Claim amounts and premiums are presented using the euro symbol because the
portfolio is French. The source does not provide premium actually charged.
Policies without amount-bearing claims retain zero total claim amount; they are
not converted to missing values or removed from frequency and pure-premium
analysis.
