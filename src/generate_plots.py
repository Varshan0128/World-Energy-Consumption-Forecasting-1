import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os
from src.challenger import load_and_preprocess_data, SimpleRNNModel, LSTMModel, GRUModel

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def create_all_visualizations():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    data = load_and_preprocess_data()
    df = data['df']
    scaler = data['scaler']
    test_df = pd.read_csv("outputs/test_predictions.csv")

    with open("outputs/final_metrics.json", "r") as f:
        metrics_json = json.load(f)

    rnn_metrics = metrics_json['models']['SimpleRNN']['test_metrics']

    # 1. Historical Energy Consumption Graph (1965-2024)
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df['primary_energy_consumption'], 'b-', linewidth=2, label='US Historical Primary Energy Consumption')
    latest_2024_actual = df[df['year'] == 2024]['primary_energy_consumption'].values[0]
    plt.axvline(2024, color='red', linestyle='--', alpha=0.7, label=f'Latest Actual Data Indicator (2024: {latest_2024_actual:.1f} TWh)')
    plt.title("US Historical Primary Energy Consumption (1965–2024)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Primary Energy Consumption (TWh)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
    plt.savefig("outputs/jules_historical_energy_graph.png", dpi=300)
    plt.savefig("reports/historical_energy_graph.png", dpi=300)
    plt.close()

    # 2. Training vs Validation Loss Curves (Training History)
    hist_df = pd.read_csv("outputs/training_history.csv")
    # Config ID 12 was selected as optimal in hyperparameter search
    opt_hist = hist_df[hist_df['Config_ID'] == 12]

    plt.figure(figsize=(10, 6))
    plt.plot(opt_hist['Epoch'], opt_hist['Train_MAE_TWh'], 'b-', linewidth=2, label='Training MAE (TWh)')
    plt.plot(opt_hist['Epoch'], opt_hist['Val_MAE_TWh'], 'r--', linewidth=2, label='Validation MAE (TWh)')
    plt.axvline(25, color='green', linestyle=':', label='Best Validation Epoch (Epoch 25)')
    plt.title("Training vs Validation MAE Curves (Selected Model: SimpleRNN)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Absolute Error (TWh)", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/training_validation_curves.png", dpi=300)
    plt.savefig("reports/training_validation_curves.png", dpi=300)
    plt.close()

    # 3. Residual Analysis Plot
    plt.figure(figsize=(10, 6))
    for model_col in ['Persistence', 'LinearRegression (AR)', 'SimpleRNN', 'LSTM', 'GRU']:
        err_col = f"{model_col}_Error_TWh"
        if err_col in test_df.columns:
            plt.plot(test_df['Year'], test_df[err_col], '-o', label=f"{model_col} Error", alpha=0.85)

    plt.axhline(0, color='black', linestyle=':', linewidth=1.5)
    plt.title("Residual Analysis on Test Set (Predicted - Actual in TWh)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Forecast Residual (TWh)", fontsize=12)
    plt.xticks(test_df['Year'])
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/residual_analysis.png", dpi=300)
    plt.savefig("outputs/jules_residual_analysis.png", dpi=300)
    plt.savefig("reports/residual_analysis.png", dpi=300)
    plt.close()

    print("Successfully regenerated all visualization plots.")

if __name__ == '__main__':
    create_all_visualizations()
