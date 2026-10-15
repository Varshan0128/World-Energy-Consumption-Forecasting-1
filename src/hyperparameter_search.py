import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def load_data(data_path="data/owid-energy-data.csv", country="United States", seq_len=5):
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

    return {
        'df': us,
        'X_raw': np.array(X_raw),
        'y_raw': np.array(y_raw),
        'target_years': np.array(target_years)
    }

def run_grid_search_and_rolling_origin():
    data = load_data()
    X_raw = data['X_raw']
    y_raw = data['y_raw']
    target_years = data['target_years']

    train_mask = (target_years >= 1970) & (target_years <= 2006)
    val_mask = (target_years >= 2007) & (target_years <= 2014)

    architectures = {'SimpleRNN': SimpleRNNModel, 'LSTM': LSTMModel, 'GRU': GRUModel}
    hidden_units_list = [4, 8, 16]
    learning_rates = [0.001, 0.003, 0.01]
    scalers = {'MinMax': MinMaxScaler, 'Standard': StandardScaler}

    results = []
    training_history_records = []

    config_id = 0

    for arch_name, model_cls in architectures.items():
        for hidden_dim in hidden_units_list:
            for lr in learning_rates:
                for scaler_name, ScalerCls in scalers.items():
                    config_id += 1
                    set_seed(42)

                    scaler = ScalerCls()
                    scaler.fit(y_raw[train_mask].reshape(-1, 1))

                    X_scaled = scaler.transform(X_raw.reshape(-1, 1)).reshape(X_raw.shape)
                    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).flatten()

                    X_train = torch.tensor(X_scaled[train_mask], dtype=torch.float32).unsqueeze(-1)
                    y_train = torch.tensor(y_scaled[train_mask], dtype=torch.float32)

                    X_val = torch.tensor(X_scaled[val_mask], dtype=torch.float32).unsqueeze(-1)
                    y_val_raw = y_raw[val_mask]

                    model = model_cls(hidden_dim=hidden_dim)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    patience = 25
                    best_val_loss = float('inf')
                    best_epoch = 0
                    best_val_mae = float('inf')
                    best_val_rmse = float('inf')
                    patience_counter = 0

                    max_epochs = 200

                    for epoch in range(1, max_epochs + 1):
                        model.train()
                        optimizer.zero_grad()
                        out = model(X_train).squeeze(-1)
                        loss = criterion(out, y_train)
                        loss.backward()
                        optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            val_out_scaled = model(X_val).squeeze(-1).numpy()
                            val_loss = criterion(model(X_val).squeeze(-1), torch.tensor(y_scaled[val_mask], dtype=torch.float32)).item()

                            val_preds = scaler.inverse_transform(val_out_scaled.reshape(-1, 1)).flatten()
                            val_mae = mean_absolute_error(y_val_raw, val_preds)
                            val_rmse = np.sqrt(mean_squared_error(y_val_raw, val_preds))

                            train_preds = scaler.inverse_transform(out.detach().numpy().reshape(-1, 1)).flatten()
                            train_mae = mean_absolute_error(y_raw[train_mask], train_preds)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = epoch
                            best_val_mae = val_mae
                            best_val_rmse = val_rmse
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        training_history_records.append({
                            'Config_ID': config_id,
                            'Architecture': arch_name,
                            'Hidden_Dim': hidden_dim,
                            'Learning_Rate': lr,
                            'Scaler': scaler_name,
                            'Epoch': epoch,
                            'Train_Loss': loss.item(),
                            'Val_Loss': val_loss,
                            'Train_MAE_TWh': train_mae,
                            'Val_MAE_TWh': val_mae,
                            'Val_RMSE_TWh': val_rmse
                        })

                        if patience_counter >= patience:
                            break

                    results.append({
                        'Config_ID': config_id,
                        'Architecture': arch_name,
                        'Hidden_Dim': hidden_dim,
                        'Learning_Rate': lr,
                        'Scaler': scaler_name,
                        'Best_Epoch': best_epoch,
                        'Completed_Epochs': epoch,
                        'Best_Val_Loss': best_val_loss,
                        'Val_MAE_TWh': best_val_mae,
                        'Val_RMSE_TWh': best_val_rmse
                    })

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    search_df = pd.DataFrame(results)
    search_df.to_csv("outputs/hyperparameter_search.csv", index=False)
    search_df.to_csv("reports/hyperparameter_search.csv", index=False)

    hist_df = pd.DataFrame(training_history_records)
    hist_df.to_csv("outputs/training_history.csv", index=False)
    hist_df.to_csv("reports/training_history.csv", index=False)

    print(f"Hyperparameter search completed. Evaluated {len(results)} configurations.")
    best_config = search_df.sort_values('Val_MAE_TWh').iloc[0]
    print("Optimal Validation Configuration:")
    print(best_config.to_string())

if __name__ == "__main__":
    run_grid_search_and_rolling_origin()
