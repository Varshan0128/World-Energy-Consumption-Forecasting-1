# Reproducibility Guide & Execution Commands

Follow these steps to reproduce all experiments, model training, plot generation, and automated test verification.

## Prerequisites
Ensure Python 3.10+ and required packages are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pytest torch
```

## Reproduction Steps

### 1. Execute Systematic Hyperparameter Search
Runs grid search and chronological rolling-origin validation across 54 model configurations:
```bash
PYTHONPATH=. python3 src/hyperparameter_search.py
```
Outputs: `outputs/hyperparameter_search.csv` and `outputs/training_history.csv`.

### 2. Execute Main Challenger & Evaluation Pipeline
Trains models across 5 random seeds (`42, 123, 456, 789, 2024`), applies early stopping, and evaluates on test set:
```bash
PYTHONPATH=. python3 src/challenger.py
```
Outputs: `models/final_model.pth`, `models/scaler.pkl`, `models/model_config.json`, `outputs/final_metrics.json`, `outputs/model_comparison.csv`, `outputs/per_seed_metrics.csv`, and `outputs/test_predictions.csv`.

### 3. Generate Visualization Plots
Produces all visualization plots:
```bash
PYTHONPATH=. python3 src/generate_plots.py
```
Outputs: `outputs/actual_vs_predicted.png`, `outputs/training_validation_curves.png`, and `outputs/residual_analysis.png`.

### 4. Execute Automated Test Suite
Runs all automated unit tests verifying split boundaries, data leakage, inverse scaling, metric calculations, and saved checkpoint inference:
```bash
PYTHONPATH=. pytest tests/
```
Expected result: `7 passed`.
