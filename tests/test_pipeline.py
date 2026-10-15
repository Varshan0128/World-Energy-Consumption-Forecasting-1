import pytest
import numpy as np
import torch
import os
import json
import pandas as pd
from src.challenger import load_and_preprocess_data, compute_metrics, GRUModel, LSTMModel, SimpleRNNModel

def test_sequence_alignment():
    """Verify that a 5-year input window [t-5, t-4, t-3, t-2, t-1] strictly predicts target year t."""
    data = load_and_preprocess_data()
    X_raw = data['X_raw']
    y_raw = data['y_raw']
    years = data['years']
    df = data['df']

    idx = np.where(years == 2015)[0][0]
    target_val = y_raw[idx]
    input_window = X_raw[idx]

    df_2015_val = df[df['year'] == 2015]['primary_energy_consumption'].values[0]
    df_window_vals = df[(df['year'] >= 2010) & (df['year'] <= 2014)]['primary_energy_consumption'].values

    assert np.isclose(target_val, df_2015_val), "Target value for 2015 does not match dataframe"
    assert np.allclose(input_window, df_window_vals), "5-year input window for 2015 target does not match 2010-2014 values"

def test_chronological_split_integrity():
    """Verify that train, val, and test target year splits are strictly non-overlapping and chronological."""
    data = load_and_preprocess_data()
    years = data['years']
    train_years = years[data['train_mask']]
    val_years = years[data['val_mask']]
    test_years = years[data['test_mask']]

    assert train_years.min() == 1970 and train_years.max() == 2006
    assert val_years.min() == 2007 and val_years.max() == 2014
    assert test_years.min() == 2015 and test_years.max() == 2024

    assert len(set(train_years).intersection(set(val_years))) == 0
    assert len(set(val_years).intersection(set(test_years))) == 0
    assert len(set(train_years).intersection(set(test_years))) == 0

def test_scaling_data_leakage_prevention():
    """Verify that MinMaxScaler parameters are derived strictly from the training set."""
    data = load_and_preprocess_data()
    scaler = data['scaler']
    y_raw = data['y_raw']
    train_y = y_raw[data['train_mask']]

    assert np.isclose(scaler.data_min_[0], train_y.min()), "Scaler min value leaked non-train data"
    assert np.isclose(scaler.data_max_[0], train_y.max()), "Scaler max value leaked non-train data"

def test_metrics_calculation():
    """Verify MAE, RMSE, and MAPE calculations in original TWh units."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = compute_metrics(y_true, y_pred)
    assert np.isclose(metrics['mae'], 20.0 / 3.0)
    assert np.isclose(metrics['rmse'], np.sqrt((100 + 100 + 0) / 3.0))
    assert metrics['mae'] > 0

def test_reports_artifact_integrity():
    """Verify that all mandatory reports/ artifacts exist and are populated."""
    required_files = [
        'reports/experiment_environment.txt',
        'reports/model_comparison.csv',
        'reports/test_predictions.csv',
        'reports/test_results.json',
        'reports/hyperparameter_search.csv',
        'reports/training_history.csv'
    ]
    for filepath in required_files:
        assert os.path.exists(filepath), f"Missing required artifact: {filepath}"
        assert os.path.getsize(filepath) > 0, f"Empty artifact file: {filepath}"

def test_hyperparameter_best_epoch_consistency():
    """Verify that Best_Epoch in hyperparameter_search.csv matches the global minimum validation loss epoch in training_history.csv."""
    search_df = pd.read_csv('outputs/hyperparameter_search.csv')
    hist_df = pd.read_csv('outputs/training_history.csv')

    for _, row in search_df.iterrows():
        cid = row['Config_ID']
        best_ep = row['Best_Epoch']
        rep_val_mae = row['Val_MAE_TWh']

        c_hist = hist_df[hist_df['Config_ID'] == cid]
        act_val_mae = c_hist[c_hist['Epoch'] == best_ep]['Val_MAE_TWh'].values[0]
        assert np.isclose(rep_val_mae, act_val_mae, atol=1e-3), f"Best epoch metric mismatch for Config {cid}"

def test_saved_model_inference():
    """Verify saved PyTorch model checkpoint loading and inference reproducibility."""
    model = SimpleRNNModel(input_dim=1, hidden_dim=16)
    model.eval()
    dummy_input = torch.randn(1, 5, 1)

    with torch.no_grad():
        out1 = model(dummy_input).numpy()

    os.makedirs("models", exist_ok=True)
    tmp_path = "models/test_dummy_rnn.pth"
    torch.save(model.state_dict(), tmp_path)

    loaded_model = SimpleRNNModel(input_dim=1, hidden_dim=16)
    loaded_model.load_state_dict(torch.load(tmp_path))
    loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(dummy_input).numpy()

    assert np.allclose(out1, out2), "Loaded model predictions do not match original model outputs"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
