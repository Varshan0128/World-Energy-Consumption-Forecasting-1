# Final Training Configuration & Hyperparameter Log

**Selected Architecture:** SimpleRNN
**Selected Scaler:** MinMax Scaler
**Input Window Length:** 5 annual observations ($t-5, \dots, t-1$)
**Target Variable:** `primary_energy_consumption` (TWh)
**Country:** United States

## Model Hyperparameters
- **Hidden Layer Units:** 8
- **Learning Rate:** 0.01
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Train Split Target Years:** 1970–2006 ($N=37$)
- **Validation Split Target Years:** 2007–2014 ($N=8$)
- **Test Split Target Years:** 2015–2024 ($N=10$)
- **Patience for Early Stopping:** 25 epochs
- **Best Validation Epoch:** Epoch 27
- **Total Completed Epochs:** Epoch 52
- **Trainable Parameter Count:** 89 parameters

## Selection Protocol
Model selection was conducted strictly on validation set metrics across 54 systematically evaluated configurations in `outputs/hyperparameter_search.csv`. The test set (2015–2024) was kept completely untouched during hyperparameter tuning.
