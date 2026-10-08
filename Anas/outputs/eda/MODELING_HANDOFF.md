# EDA Modeling Handoff

## Dataset Dimensions
- Uncertainty cases: 100
- Raw production rows: 6,724,340
- Grid active rows per case: [42512]
- Shared history coverage: 1998-01-01 00:00:00 to 2008-01-01 00:00:00

## Recommended Baseline Sequence
1. Fit input scaling using Cases 1-70 only.
2. Start with the four uncertainty parameters as inputs.
3. Compare regularized regression and GP surrogates for curve coefficients.
4. Add grid summaries only if they improve validation results.
5. Validate temporal forecasting with rolling history cutoffs.
6. Use Cases 71-85 for model selection; evaluate Cases 86-100 once.

## Exported Artifacts
- case_modeling_candidates.parquet
- grid_case_features.parquet
- grid_layer_features.parquet
- quality_summary.json
- tables/*.csv
- figures/*.png and figures/*.pdf
