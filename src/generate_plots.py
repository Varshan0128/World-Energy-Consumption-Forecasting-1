import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def create_plots():
    os.makedirs("outputs", exist_ok=True)

    # Load test predictions
    df = pd.read_csv("outputs/jules_test_predictions.csv")

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Actual_TWh'], 'k-o', label='Actual Primary Energy Consumption', linewidth=2.5, markersize=8)
    plt.plot(df['Year'], df['Persistence_Pred_TWh'], '--s', label='Persistence Baseline', alpha=0.8)
    plt.plot(df['Year'], df['LinearRegression (AR)_Pred_TWh'], '--^', label='Linear Regression (AR)', alpha=0.8)
    plt.plot(df['Year'], df['LSTM_Pred_TWh'], '-d', label='LSTM (Selected Challenger)', color='red', linewidth=2, markersize=7)
    plt.plot(df['Year'], df['GRU_Pred_TWh'], '-v', label='GRU', color='green', alpha=0.8)

    plt.title("US Primary Energy Consumption: Actual vs Predictions (Test Period 2015–2024)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12, labelpad=8)
    plt.ylabel("Primary Energy Consumption (TWh)", fontsize=12, labelpad=8)
    plt.xticks(df['Year'])
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/jules_actual_vs_predicted.png", dpi=300)
    plt.close()

    # 2. Residual Analysis Plot
    plt.figure(figsize=(10, 6))
    models = ['Persistence', 'LinearRegression (AR)', 'SimpleRNN', 'LSTM', 'GRU']
    for model in models:
        err_col = f"{model}_Error_TWh" if f"{model}_Error_TWh" in df.columns else f"{model} (AR)_Error_TWh"
        if err_col not in df.columns and model == 'LinearRegression (AR)':
            err_col = 'LinearRegression (AR)_Error_TWh'
        plt.plot(df['Year'], df[err_col], '-o', label=f"{model} Error", alpha=0.85)

    plt.axhline(0, color='black', linestyle=':', linewidth=1.5)
    plt.title("Residual Analysis on Test Set (Predicted - Actual in TWh)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12, labelpad=8)
    plt.ylabel("Forecast Residual (TWh)", fontsize=12, labelpad=8)
    plt.xticks(df['Year'])
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/jules_residual_analysis.png", dpi=300)
    plt.close()

    print("Generated actual_vs_predicted.png and residual_analysis.png in outputs/")

if __name__ == '__main__':
    create_plots()
