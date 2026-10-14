import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from challenger import load_and_preprocess_data, GRUModel, LSTMModel, SimpleRNNModel, set_seed

def run_gradient_experiment():
    data = load_and_preprocess_data()
    X_raw = data['X_raw']
    y_raw = data['y_raw']
    train_mask = data['train_mask']

    X_train_raw = X_raw[train_mask]
    y_train_raw = y_raw[train_mask]

    # 1. MinMax Scaler
    minmax = MinMaxScaler()
    minmax.fit(y_train_raw.reshape(-1, 1))
    X_train_minmax = minmax.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
    y_train_minmax = minmax.transform(y_train_raw.reshape(-1, 1)).flatten()

    # 2. Standard Scaler
    std_scaler = StandardScaler()
    std_scaler.fit(y_train_raw.reshape(-1, 1))
    X_train_std = std_scaler.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
    y_train_std = std_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()

    # 3. Unscaled
    X_train_unscaled = X_train_raw
    y_train_unscaled = y_train_raw

    configs = [
        ("MinMax Scaled [0, 1]", X_train_minmax, y_train_minmax),
        ("Standard Scaled (Z-score)", X_train_std, y_train_std),
        ("Unscaled (TWh)", X_train_unscaled, y_train_unscaled)
    ]

    epochs = 200
    results = {}

    for label, X_tr, y_tr in configs:
        set_seed(42)
        model = GRUModel(input_dim=1, hidden_dim=16)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        X_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y_tr, dtype=torch.float32)

        grad_norms = []
        losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_t).squeeze(-1)
            loss = criterion(out, y_t)
            loss.backward()

            # Measure gradient norm across all parameters
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms.append(total_norm)
            losses.append(loss.item())
            optimizer.step()

        results[label] = {
            'first_10_avg_grad_norm': float(np.mean(grad_norms[:10])),
            'last_10_avg_grad_norm': float(np.mean(grad_norms[-10:])),
            'final_loss': float(losses[-1]),
            'grad_norms_epoch_1_10_100_200': [grad_norms[0], grad_norms[9], grad_norms[99], grad_norms[199]]
        }

    print("=== Gradient Norms & Loss Scaling Comparison ===")
    for k, v in results.items():
        print(f"\n{k}:")
        print(f"  First 10 Epochs Avg Grad Norm: {v['first_10_avg_grad_norm']:.6f}")
        print(f"  Last 10 Epochs Avg Grad Norm:  {v['last_10_avg_grad_norm']:.6f}")
        print(f"  Final Epoch Loss:             {v['final_loss']:.6f}")
        print(f"  Grad Norms [Epoch 1, 10, 100, 200]: {[round(x, 6) for x in v['grad_norms_epoch_1_10_100_200']]}")

    return results

if __name__ == "__main__":
    run_gradient_experiment()
