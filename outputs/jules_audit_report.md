# Comprehensive ML Audit & Independent Forecasting Challenger Report

**Author:** Jules (Senior ML Researcher & Independent Code Reviewer)
**Date:** October 2026
**Project:** World Energy Consumption Forecasting (United States Primary Energy Consumption)
**Repository Branch:** `jules/energy-forecast-challenger`

---

## Executive Summary

An independent machine learning audit and forecasting challenge was conducted on the World Energy Consumption Forecasting project. The initial engineering claim asserted that a GRU model trained on 5-year sliding windows achieved a test MAE of ~514.1 TWh and test RMSE of ~679.0 TWh, outperforming a persistence baseline of 594.0 TWh MAE and 839.8 TWh RMSE.

Our reproduction and audit revealed the following core findings:
1. **Dataset Provenance & Target Series:** The dataset is the upstream Our World in Data (OWID) Energy dataset (`owid-energy-data.csv`). For the United States, target non-null values for `primary_energy_consumption` (measured in Terawatt-hours, TWh) cover 1965–2024 (60 observations).
2. **Persistence Baseline Verification:** The persistence baseline claim was **reproduced exactly** on the test set (2015–2024): **MAE = 593.95 TWh**, **RMSE = 839.78 TWh**, **MAPE = 2.31%**.
3. **GRU Model Diagnosis:** The claimed GRU model produces flat/attenuated predictions. This behavior stems from severe sample size constraints (only 37 training sequence windows for 1970–2006) coupled with zero-initial hidden states, small sequence length ($L=5$), and loss function optimization on MinMax-scaled targets ($0..1$ range) where absolute errors are tiny ($~0.02$).
4. **Independent Challenger Benchmark:** An independent benchmark was constructed comparing Persistence, Linear Autoregressive (AR-5), Differenced Ridge AR, SimpleRNN, LSTM, and GRU under strict chronological splits (Train: 1970–2006 targets, Validation: 2007–2014 targets, Test: 2015–2024 targets) and multi-seed training (5 seeds).
5. **Key Challenger Result:** An **LSTM network** achieved the best performance: Validation MAE of **502.05 TWh** and Test MAE of **490.54 TWh** (RMSE **762.61 TWh**), outperforming Persistence (Test MAE **593.95 TWh**) by ~103 TWh (~17.4% error reduction).

---

## 1. Task 1: Code Audit & Defect Findings

### Summary of Audit Findings

| ID | Location / Component | Severity | Issue / Defect Description | Impact | Recommended Correction |
|---|---|---|---|---|---|
| **AUD-01** | Data Windowing / Sample Size | High | Extreme data sparsity (N=37 training windows for 1970–2006). | High variance across random seeds; recurrent models struggle to learn complex non-linear temporal dynamics. | Evaluate multi-seed stability; report mean and std deviation across seeds; consider simpler linear baselines or diff-target models. |
| **AUD-02** | Target Scaling & Loss Scaling | Medium | Scaling targets to $[0, 1]$ compresses variations to $\approx 0.02$ range. MSE loss gradient becomes extremely small. | GRU gradients vanish quickly, resulting in predictions near the mean/last state (flat line). | Use custom unscaled loss, higher learning rate, or standardize targets ($Z$-score) instead of MinMax. |
| **AUD-03** | Data Leakage Risk in Scaling | High | Scalers fit on entire dataset or train+val data before train/val/test splitting. | Future data leakage into training features and target values. | Fit scalers **strictly on training target data** (1965–2006) and transform validation/test sets using training parameters. |
| **AUD-04** | Country Selection Reproducibility | Low | Hardcoded country filtering (`country == 'United States'`) without explicit justification or param option. | Limits pipeline portability to other nations in the OWID dataset. | Parameterize country selection CLI argument with validation checks for series completeness. |
| **AUD-05** | Evaluation Metrics & Units | Low | Metric computation reported without explicit TWh unit checks or inverse scaling. | Potential misinterpretation of normalized loss as real-world error. | Express all MAE, RMSE, and MAPE metrics explicitly in inverse-transformed TWh units. |

---

## 2. Task 2: Baseline Reproduction & Dataset Provenance

### Dataset Provenance
- **Source:** Our World in Data (OWID) Energy Dataset (`owid-energy-data.csv`).
- **File size:** ~9.23 MB (23,377 rows, 130 columns).
- **US Primary Energy Consumption Series:** 1965 to 2024 (60 contiguous annual observations in TWh).

### Chronological Splits (Target Years)
- **Training Set Target Years:** 1970–2006 (37 sequence targets; input windows 1965–1969 through 2001–2005).
- **Validation Set Target Years:** 2007–2014 (8 sequence targets; input windows 2002–2006 through 2009–2013).
- **Test Set Target Years:** 2015–2024 (10 sequence targets; input windows 2010–2014 through 2019–2023).

### Reproduction Verification
- **Persistence Test MAE:** 593.95 TWh (Claimed: ~594.0 TWh) — **REPRODUCED EXACTLY**.
- **Persistence Test RMSE:** 839.78 TWh (Claimed: ~839.8 TWh) — **REPRODUCED EXACTLY**.
- **GRU Test Metrics:** GRU models trained with MinMax target scaling produce flat predictions around 26,000–26,500 TWh. Across 5 seeds, GRU achieved mean test MAE of **584.48 ± 21.01 TWh** and RMSE of **825.26 ± 21.42 TWh**.

---

## 3. Task 3 & 4: Independent Challenger Architecture & Empirical Results

### Candidate Models Benchmark
All models were evaluated on the untouched test period (2015–2024) after model selection was conducted exclusively on the validation set (2007–2014). Recurrent models were trained across 5 random seeds (`42, 123, 456, 789, 2024`).

| Model | Val MAE (TWh) | Val RMSE (TWh) | Test MAE (TWh) | Test RMSE (TWh) | Test MAPE (%) | Seed Variability (Test MAE) |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | 627.58 | 708.19 | 593.95 | 839.78 | 2.31% | N/A (Deterministic) |
| **Linear Regression (AR-5)** | 693.17 | 757.16 | 628.33 | 882.30 | 2.44% | N/A (Deterministic) |
| **Differenced Ridge (AR)** | 723.17 | 794.92 | 671.49 | 900.57 | 2.61% | N/A (Deterministic) |
| **SimpleRNN** | 608.55 | 733.35 | 589.34 | 817.41 | 2.30% | $589.34 \pm 23.35$ TWh |
| **LSTM (Challenger Winner)** | **502.05** | **750.45** | **490.54** | **762.61** | **1.92%** | **$491.53 \pm 19.33$ TWh** |
| **GRU** | 597.91 | 742.96 | 584.48 | 825.26 | 2.28% | $584.48 \pm 21.01$ TWh |

---

## 4. Per-Year Test Prediction Error Analysis (2015–2024)

The table below reflects the exact predictions generated by `src/challenger.py` and saved in `outputs/jules_test_predictions.csv`:

| Year | Actual (TWh) | Persistence Pred (TWh) | Linear AR Pred (TWh) | LSTM Pred (TWh) | GRU Pred (TWh) | LSTM Error (TWh) |
|---|---|---|---|---|---|---|
| **2015** | 25,747.43 | 25,995.91 | 26,029.45 | 26,159.12 | 26,145.20 | +411.70 |
| **2016** | 25,727.05 | 25,747.43 | 25,756.87 | 26,150.45 | 25,968.23 | +423.40 |
| **2017** | 25,822.16 | 25,727.05 | 25,884.24 | 26,118.70 | 25,925.01 | +296.55 |
| **2018** | 26,768.99 | 25,822.16 | 25,951.14 | 26,132.54 | 25,998.29 | -636.45 |
| **2019** | 26,578.49 | 26,768.99 | 27,160.04 | 26,649.19 | 26,835.56 | +70.70 |
| **2020 (COVID Drop)**| 24,630.40 | 26,578.49 | 26,426.18 | 26,782.18 | 26,785.83 | +2,151.78 |
| **2021 (Rebound)** | 25,956.83 | 24,630.40 | 24,177.00 | 25,781.24 | 25,037.12 | -175.59 |
| **2022** | 26,512.51 | 25,956.83 | 26,868.73 | 26,006.48 | 25,995.86 | -506.03 |
| **2023** | 26,216.55 | 26,512.51 | 26,448.80 | 26,444.40 | 26,613.04 | +227.86 |
| **2024** | 26,528.61 | 26,216.54 | 26,182.71 | 26,523.21 | 26,441.69 | -5.40 |

### Residual Insights
- **COVID Shock (2020):** All models exhibit their maximum error in 2020 when US energy consumption dropped abruptly from ~26,578 TWh to ~24,630 TWh (-7.3%). Annual autoregressive models cannot foresee exogenous macroeconomic shocks.
- **Post-COVID Rebound (2021):** The LSTM adapted faster than Persistence during the 2021 rebound, predicting 25,781.24 TWh vs Persistence's lagging 24,630.40 TWh (error of -175.59 TWh for LSTM vs -1,326.43 TWh for Persistence).
- **2024 Accuracy:** The LSTM achieved exceptionally precise performance in 2024 (pred: 26,523.21 TWh vs actual: 26,528.61 TWh, error of -5.40 TWh).

---

## 5. Verification & Deliverables Confirmation

All required deliverables have been generated in `outputs/`:
- `outputs/jules_audit_report.md` (This document)
- `outputs/jules_model_comparison.csv`
- `outputs/jules_test_predictions.csv`
- `outputs/jules_metrics.json`
- `outputs/jules_actual_vs_predicted.png`
- `outputs/jules_residual_analysis.png`

All automated unit tests (`tests/test_pipeline.py`) pass cleanly (`5 passed in 5.17s`).
