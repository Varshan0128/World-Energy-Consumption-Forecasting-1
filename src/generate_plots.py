import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os
from src.challenger import load_and_preprocess_data, LSTMModel

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def create_all_visualizations():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load dataset & model prediction artifacts
    data = load_and_preprocess_data()
    df = data['df']
    scaler = data['scaler']
    test_df = pd.read_csv("outputs/jules_test_predictions.csv")

    with open("reports/test_results.json", "r") as f:
        metrics_json = json.load(f)
    lstm_metrics = metrics_json['models']['LSTM']['test_metrics']

    # Load trained LSTM model
    model = LSTMModel(hidden_dim=16)
    model.load_state_dict(torch.load('models/best_lstm.pth'))
    model.eval()

    # Generate 5-year future forecast (2025-2029) using autoregressive inference
    recent_5_years = list(df[(df['year'] >= 2020) & (df['year'] <= 2024)]['primary_energy_consumption'].values)
    latest_2024_actual = recent_5_years[-1]

    future_years = [2025, 2026, 2027, 2028, 2029]
    future_forecasts = []
    current_window = list(recent_5_years)

    for yr in future_years:
        scaled_w = scaler.transform(np.array(current_window).reshape(-1, 1)).reshape(1, 5, 1)
        x_tensor = torch.tensor(scaled_w, dtype=torch.float32)
        with torch.no_grad():
            pred_s = model(x_tensor).item()
        pred_twh = scaler.inverse_transform(np.array([[pred_s]]))[0, 0]
        future_forecasts.append(pred_twh)
        current_window = current_window[1:] + [pred_twh]

    # Save future forecasts CSV
    future_df = pd.DataFrame({'Year': future_years, 'Forecast_TWh': np.round(future_forecasts, 2)})
    future_df.to_csv("outputs/jules_future_forecast.csv", index=False)
    future_df.to_csv("reports/future_forecast.csv", index=False)

    # -------------------------------------------------------------
    # 1. Historical Energy Consumption Graph (1965-2024)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df['primary_energy_consumption'], 'b-', linewidth=2, label='US Historical Primary Energy Consumption')
    plt.axvline(2024, color='red', linestyle='--', alpha=0.7, label=f'Latest Actual Data Indicator (2024: {latest_2024_actual:.1f} TWh)')
    plt.title("US Historical Primary Energy Consumption (1965–2024)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Primary Energy Consumption (TWh)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/jules_historical_energy_graph.png", dpi=300)
    plt.savefig("reports/historical_energy_graph.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. 5-Year Future Forecast Graph (2025-2029)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    hist_tail = df[df['year'] >= 2018]
    plt.plot(hist_tail['year'], hist_tail['primary_energy_consumption'], 'k-o', label='Actual Primary Energy (2018–2024)', linewidth=2)
    plt.plot(future_years, future_forecasts, 'r--s', label='LSTM 5-Year Future Forecast (2025–2029)', linewidth=2.5, markersize=8)

    for y, f in zip(future_years, future_forecasts):
        plt.annotate(f"{f:.0f} TWh", (y, f), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.title("US Primary Energy Consumption: 5-Year Future Forecast (2025–2029)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Primary Energy Consumption (TWh)", fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/jules_future_forecast_graph.png", dpi=300)
    plt.savefig("reports/future_forecast_graph.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Actual vs Predicted Test Graph with Metric Cards (2015-2024)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(test_df['Year'], test_df['Actual_TWh'], 'k-o', label='Actual Energy Consumption', linewidth=2.5, markersize=8)
    ax.plot(test_df['Year'], test_df['Persistence_Pred_TWh'], '--s', label='Persistence Baseline', alpha=0.8)
    ax.plot(test_df['Year'], test_df['LSTM_Pred_TWh'], '-d', label='LSTM Challenger', color='red', linewidth=2.2, markersize=7)

    ax.set_title("Actual vs Predicted Energy Consumption (Test Period 2015–2024)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Primary Energy Consumption (TWh)", fontsize=12)
    ax.set_xticks(test_df['Year'])
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add Metric Cards Overlay
    card_text = (
        f"LSTM Evaluation Metrics (2015–2024):\n"
        f"  • Test MAE:   {lstm_metrics['mae']:.2f} TWh\n"
        f"  • Test RMSE:  {lstm_metrics['rmse']:.2f} TWh\n"
        f"  • Test MAPE:  {lstm_metrics['mape']:.2f}%\n"
        f"  • 2024 Indicator: {latest_2024_actual:.1f} TWh"
    )
    ax.text(0.62, 0.18, card_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    plt.savefig("outputs/jules_actual_vs_predicted.png", dpi=300)
    plt.savefig("reports/actual_vs_predicted.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Residual Analysis Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for model_col in ['Persistence', 'LinearRegression (AR)', 'SimpleRNN', 'LSTM', 'GRU']:
        err_col = f"{model_col}_Error_TWh"
        plt.plot(test_df['Year'], test_df[err_col], '-o', label=f"{model_col} Error", alpha=0.85)

    plt.axhline(0, color='black', linestyle=':', linewidth=1.5)
    plt.title("Residual Analysis on Test Set (Predicted - Actual in TWh)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Forecast Residual (TWh)", fontsize=12)
    plt.xticks(test_df['Year'])
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/jules_residual_analysis.png", dpi=300)
    plt.savefig("reports/residual_analysis.png", dpi=300)
    plt.close()

    print("Successfully generated all graphs, metric cards, and 5-year future forecasts!")

if __name__ == '__main__':
    create_all_visualizations()
