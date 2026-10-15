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

    sr_metrics = metrics_json['models']['SimpleRNN']['test_metrics']
    latest_2024_actual = df[df['year'] == 2024]['primary_energy_consumption'].values[0]

    # 1. Historical Energy Consumption & Test Actual vs Predicted Graph (1965-2024)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(df['year'], df['primary_energy_consumption'], 'k-o', linewidth=2, markersize=5, label='Actual US Primary Energy')
    ax.plot(test_df['Year'], test_df['Persistence_Pred_TWh'], '--s', label='Persistence Baseline', alpha=0.8)
    ax.plot(test_df['Year'], test_df['SimpleRNN_Pred_TWh'], '-d', label='SimpleRNN (Selected)', color='green', linewidth=2.2, markersize=7)

    ax.axvline(2024, color='red', linestyle='--', alpha=0.7, label=f'Latest 2024 Actual Indicator ({latest_2024_actual:.1f} TWh)')
    ax.set_title("US Primary Energy Consumption: Actual vs Predictions (2015–2024)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Primary Energy Consumption (TWh)", fontsize=12)
    ax.set_xticks(test_df['Year'])
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Overlay metric cards
    card_text = (
        f"SimpleRNN Test Metrics (2015–2024):\n"
        f"  • Test MAE:   {sr_metrics['mae']:.2f} TWh\n"
        f"  • Test RMSE:  {sr_metrics['rmse']:.2f} TWh\n"
        f"  • Test MAPE:  {sr_metrics['mape']:.2f}%\n"
        f"  • Test R²:    {sr_metrics['r2']:.4f}\n"
        f"  • Bias:       {sr_metrics['mean_signed_bias']:.2f} TWh"
    )
    ax.text(0.58, 0.18, card_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
    plt.savefig("outputs/jules_actual_vs_predicted.png", dpi=300)
    plt.savefig("reports/actual_vs_predicted.png", dpi=300)
    plt.close()

    # 2. Training vs Validation Loss Curves (Training History)
    hist_df = pd.read_csv("outputs/training_history.csv")
    # Config ID 11 was selected as optimal (SimpleRNN, Hidden Dim 8, LR 0.01, MinMax)
    opt_hist = hist_df[hist_df['Config_ID'] == 11]

    plt.figure(figsize=(10, 6))
    plt.plot(opt_hist['Epoch'], opt_hist['Train_MAE_TWh'], 'b-', linewidth=2, label='Training MAE (TWh)')
    plt.plot(opt_hist['Epoch'], opt_hist['Val_MAE_TWh'], 'r--', linewidth=2, label='Validation MAE (TWh)')
    plt.axvline(27, color='green', linestyle=':', linewidth=2, label='Best Validation Epoch (Epoch 27)')
    plt.title("Training vs Validation MAE Curves (Selected Model: SimpleRNN Config 11)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Absolute Error (TWh)", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/training_validation_curves.png", dpi=300)
    plt.savefig("reports/training_validation_curves.png", dpi=300)
    plt.close()

    # 3. Residual Analysis Plot (Residual = Predicted - Actual)
    plt.figure(figsize=(10, 6))
    for model_col in ['Persistence', 'LinearRegression (AR)', 'SimpleRNN', 'LSTM', 'GRU']:
        err_col = f"{model_col}_Error_TWh"
        if err_col in test_df.columns:
            plt.plot(test_df['Year'], test_df[err_col], '-o', label=f"{model_col} Residual", alpha=0.85)

    plt.axhline(0, color='black', linestyle=':', linewidth=1.5)
    plt.title("Residual Analysis on Test Set (Residual = Predicted - Actual in TWh)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Forecast Residual (TWh)", fontsize=12)
    plt.xticks(test_df['Year'])
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/residual_analysis.png", dpi=300)
    plt.savefig("outputs/jules_residual_analysis.png", dpi=300)
    plt.savefig("reports/residual_analysis.png", dpi=300)
    plt.close()

    print("Successfully regenerated all visualization plots with authoritative metrics and residual = predicted - actual.")

if __name__ == '__main__':
    create_all_visualizations()
