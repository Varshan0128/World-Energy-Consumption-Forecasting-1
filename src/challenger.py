import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    med_ae = median_absolute_error(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "mean_signed_bias": float(bias),
        "median_ae": float(med_ae)
    }

def load_and_preprocess_data(data_path="data/owid-energy-data.csv", country="United States", seq_len=5):
    df = pd.read_csv(data_path)
    us = df[df['country'] == country].sort_values('year').copy()
    us = us[(us['year'] >= 1965) & (us['year'] <= 2024)].reset_index(drop=True)

    target_col = 'primary_energy_consumption'
    us = us.dropna(subset=[target_col]).reset_index(drop=True)

    years = us['year'].values
    vals = us[target_col].values

    X_raw, y_raw, target_years = [], [], []
    for i in range(seq_len, len(vals)):
        X_raw.append(vals[i-seq_len:i])
        y_raw.append(vals[i])
        target_years.append(years[i])

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    target_years = np.array(target_years)

    train_mask = (target_years >= 1970) & (target_years <= 2006)
    val_mask = (target_years >= 2007) & (target_years <= 2014)
    test_mask = (target_years >= 2015) & (target_years <= 2024)

    train_y = y_raw[train_mask]
    scaler = MinMaxScaler()
    scaler.fit(train_y.reshape(-1, 1))

    return {
        'df': us,
        'X_raw': X_raw,
        'y_raw': y_raw,
        'years': target_years,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'scaler': scaler
    }

class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_pytorch_model(model_cls, data, seed=42, max_epochs=200, lr=0.01, hidden_dim=16, patience=30):
    set_seed(seed)

    scaler = data['scaler']
    X_raw = data['X_raw']
    y_raw = data['y_raw']

    X_scaled = scaler.transform(X_raw.reshape(-1, 1)).reshape(X_raw.shape)
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).flatten()

    X_train = torch.tensor(X_scaled[data['train_mask']], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_scaled[data['train_mask']], dtype=torch.float32)

    X_val = torch.tensor(X_scaled[data['val_mask']], dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_scaled[data['val_mask']], dtype=torch.float32)

    X_test = torch.tensor(X_scaled[data['test_mask']], dtype=torch.float32).unsqueeze(-1)

    model = model_cls(hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train).squeeze(-1)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val).squeeze(-1)
            val_loss = criterion(val_out, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_preds_scaled = model(X_val).squeeze(-1).numpy()
        test_preds_scaled = model(X_test).squeeze(-1).numpy()

    val_preds = scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

    return val_preds, test_preds, model, best_epoch, epoch

def run_all_experiments():
    data = load_and_preprocess_data()

    y_raw = data['y_raw']
    val_true = y_raw[data['val_mask']]
    test_true = y_raw[data['test_mask']]
    test_years = data['years'][data['test_mask']]
    val_years = data['years'][data['val_mask']]

    seeds = [42, 123, 456, 789, 2024]

    results = {}
    per_seed_records = []

    # 1. Persistence Baseline
    val_pers = data['X_raw'][data['val_mask'], -1]
    test_pers = data['X_raw'][data['test_mask'], -1]
    results['Persistence'] = {
        'val_metrics': compute_metrics(val_true, val_pers),
        'test_metrics': compute_metrics(test_true, test_pers),
        'val_preds': val_pers.tolist(),
        'test_preds': test_pers.tolist()
    }

    # 2. Linear Regression (AR)
    X_train = data['X_raw'][data['train_mask']]
    y_train = data['y_raw'][data['train_mask']]
    X_val = data['X_raw'][data['val_mask']]
    X_test = data['X_raw'][data['test_mask']]

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    val_lr = lr_model.predict(X_val)
    test_lr = lr_model.predict(X_test)
    results['LinearRegression (AR)'] = {
        'val_metrics': compute_metrics(val_true, val_lr),
        'test_metrics': compute_metrics(test_true, test_lr),
        'val_preds': val_lr.tolist(),
        'test_preds': test_lr.tolist()
    }

    # 3. Differenced Ridge (AR)
    y_train_diff = y_train - X_train[:, -1]
    diff_model = Ridge(alpha=1.0)
    X_train_diff = np.diff(X_train, axis=1)
    X_val_diff = np.diff(X_val, axis=1)
    X_test_diff = np.diff(X_test, axis=1)

    diff_model.fit(X_train_diff, y_train_diff)
    val_diff_pred = X_val[:, -1] + diff_model.predict(X_val_diff)
    test_diff_pred = X_test[:, -1] + diff_model.predict(X_test_diff)
    results['Differenced Ridge (AR)'] = {
        'val_metrics': compute_metrics(val_true, val_diff_pred),
        'test_metrics': compute_metrics(test_true, test_diff_pred),
        'val_preds': val_diff_pred.tolist(),
        'test_preds': test_diff_pred.tolist()
    }

    # Recurrent Neural Networks
    rnn_models = {
        'SimpleRNN': SimpleRNNModel,
        'LSTM': LSTMModel,
        'GRU': GRUModel
    }

    final_selected_model = None
    final_selected_scaler = data['scaler']

    for name, cls in rnn_models.items():
        seed_results = {}
        val_preds_list = []
        test_preds_list = []

        best_model = None
        best_seed_val_mae = float('inf')

        for s in seeds:
            vp, tp, model_inst, best_ep, comp_ep = train_pytorch_model(cls, data, seed=s)
            val_preds_list.append(vp)
            test_preds_list.append(tp)
            v_m = compute_metrics(val_true, vp)
            t_m = compute_metrics(test_true, tp)

            per_seed_records.append({
                'Model': name,
                'Seed': s,
                'Best_Epoch': best_ep,
                'Completed_Epochs': comp_ep,
                'Val_MAE_TWh': v_m['mae'],
                'Val_RMSE_TWh': v_m['rmse'],
                'Val_MAPE_pct': v_m['mape'],
                'Test_MAE_TWh': t_m['mae'],
                'Test_RMSE_TWh': t_m['rmse'],
                'Test_MAPE_pct': t_m['mape'],
                'Test_R2': t_m['r2'],
                'Test_Bias_TWh': t_m['mean_signed_bias'],
                'Test_Median_AE_TWh': t_m['median_ae']
            })

            seed_results[f"seed_{s}"] = {
                'val_metrics': v_m,
                'test_metrics': t_m,
                'val_preds': vp.tolist(),
                'test_preds': tp.tolist(),
                'best_epoch': best_ep,
                'completed_epochs': comp_ep
            }

            if v_m['mae'] < best_seed_val_mae:
                best_seed_val_mae = v_m['mae']
                best_model = model_inst

        # Save best model checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(best_model.state_dict(), f"models/best_{name.lower()}.pth")

        if name == 'LSTM':
            final_selected_model = best_model
            torch.save(best_model.state_dict(), "models/final_model.pth")
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(data['scaler'], f)
            config_meta = {
                'architecture': 'LSTM',
                'hidden_dim': 16,
                'learning_rate': 0.01,
                'scaler': 'MinMaxScaler',
                'target_column': 'primary_energy_consumption',
                'country': 'United States',
                'input_window_length': 5,
                'train_years': '1970-2006',
                'val_years': '2007-2014',
                'test_years': '2015-2024',
                'parameters_count': sum(p.numel() for p in best_model.parameters() if p.requires_grad)
            }
            with open("models/model_config.json", "w") as f:
                json.dump(config_meta, f, indent=2)

        # Ensemble Prediction (Mean across seeds)
        ensemble_val_preds = np.mean(val_preds_list, axis=0)
        ensemble_test_preds = np.mean(test_preds_list, axis=0)

        indiv_val_maes = [sr['val_metrics']['mae'] for sr in seed_results.values()]
        indiv_val_rmses = [sr['val_metrics']['rmse'] for sr in seed_results.values()]
        indiv_test_maes = [sr['test_metrics']['mae'] for sr in seed_results.values()]
        indiv_test_rmses = [sr['test_metrics']['rmse'] for sr in seed_results.values()]

        results[name] = {
            'ensemble_val_metrics': compute_metrics(val_true, ensemble_val_preds),
            'ensemble_test_metrics': compute_metrics(test_true, ensemble_test_preds),
            'ensemble_val_preds': ensemble_val_preds.tolist(),
            'ensemble_test_preds': ensemble_test_preds.tolist(),
            'individual_seeds': seed_results,
            'individual_seed_metrics_summary': {
                'val_mae_mean': float(np.mean(indiv_val_maes)),
                'val_mae_std': float(np.std(indiv_val_maes)),
                'val_rmse_mean': float(np.mean(indiv_val_rmses)),
                'val_rmse_std': float(np.std(indiv_val_rmses)),
                'test_mae_mean': float(np.mean(indiv_test_maes)),
                'test_mae_std': float(np.std(indiv_test_maes)),
                'test_rmse_mean': float(np.mean(indiv_test_rmses)),
                'test_rmse_std': float(np.std(indiv_test_rmses))
            },
            'val_metrics': compute_metrics(val_true, ensemble_val_preds),
            'test_metrics': compute_metrics(test_true, ensemble_test_preds),
            'val_preds': ensemble_val_preds.tolist(),
            'test_preds': ensemble_test_preds.tolist()
        }

    return results, test_years, test_true, val_years, val_true, per_seed_records

def export_all_outputs():
    results, test_years, test_true, val_years, val_true, per_seed_records = run_all_experiments()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Export final metrics JSON to outputs/ and reports/
    metrics_export = {
        'test_period': '2015-2024',
        'val_period': '2007-2014',
        'train_period': '1970-2006',
        'models': {}
    }
    for model_name, res in results.items():
        metrics_export['models'][model_name] = {
            'val_metrics': res['val_metrics'],
            'test_metrics': res['test_metrics']
        }
        if 'individual_seed_metrics_summary' in res:
            metrics_export['models'][model_name]['individual_seed_metrics_summary'] = res['individual_seed_metrics_summary']
            metrics_export['models'][model_name]['individual_seeds'] = res['individual_seeds']

    with open("outputs/final_metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)
    with open("reports/test_results.json", "w") as f:
        json.dump(metrics_export, f, indent=2)
    with open("outputs/jules_metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    # 2. Export per seed metrics CSV
    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv("outputs/per_seed_metrics.csv", index=False)
    per_seed_df.to_csv("reports/per_seed_metrics.csv", index=False)

    # 3. Export model comparison CSV
    comp_rows = []
    for model_name, res in results.items():
        row = {
            'Model': model_name,
            'Ensemble_Val_MAE_TWh': round(res['val_metrics']['mae'], 2),
            'Ensemble_Val_RMSE_TWh': round(res['val_metrics']['rmse'], 2),
            'Ensemble_Test_MAE_TWh': round(res['test_metrics']['mae'], 2),
            'Ensemble_Test_RMSE_TWh': round(res['test_metrics']['rmse'], 2),
            'Ensemble_Test_MAPE_pct': round(res['test_metrics']['mape'], 2),
            'Ensemble_Test_R2': round(res['test_metrics']['r2'], 4),
            'Ensemble_Test_Bias_TWh': round(res['test_metrics']['mean_signed_bias'], 2),
            'Ensemble_Test_Median_AE_TWh': round(res['test_metrics']['median_ae'], 2)
        }
        if 'individual_seed_metrics_summary' in res:
            summary = res['individual_seed_metrics_summary']
            row['Indiv_Test_MAE_Mean'] = round(summary['test_mae_mean'], 2)
            row['Indiv_Test_MAE_Std'] = round(summary['test_mae_std'], 2)
            row['Indiv_Test_RMSE_Mean'] = round(summary['test_rmse_mean'], 2)
            row['Indiv_Test_RMSE_Std'] = round(summary['test_rmse_std'], 2)
        else:
            row['Indiv_Test_MAE_Mean'] = round(res['test_metrics']['mae'], 2)
            row['Indiv_Test_MAE_Std'] = 0.0
            row['Indiv_Test_RMSE_Mean'] = round(res['test_metrics']['rmse'], 2)
            row['Indiv_Test_RMSE_Std'] = 0.0
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv("outputs/model_comparison.csv", index=False)
    comp_df.to_csv("outputs/jules_model_comparison.csv", index=False)
    comp_df.to_csv("reports/model_comparison.csv", index=False)

    # 4. Export test predictions CSV
    pred_df = pd.DataFrame({'Year': test_years, 'Actual_TWh': test_true})
    for model_name, res in results.items():
        pred_df[f'{model_name}_Pred_TWh'] = np.round(res['test_preds'], 2)
        pred_df[f'{model_name}_Error_TWh'] = np.round(res['test_preds'] - test_true, 2)

    pred_df.to_csv("outputs/test_predictions.csv", index=False)
    pred_df.to_csv("outputs/jules_test_predictions.csv", index=False)
    pred_df.to_csv("reports/test_predictions.csv", index=False)
    print("Exported all final metrics and CSVs to outputs/ and reports/")

if __name__ == '__main__':
    export_all_outputs()
