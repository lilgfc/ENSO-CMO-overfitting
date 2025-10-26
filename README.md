# ENSO-CMO-overfitting
A case study on model complexity vs. data size, testing LSTM, TCN, and TFT models for forecasting ENSO-driven coffee prices.
Coffee Price Forecasting: A Case Study in Model Complexity and Overfitting
This repository contains the complete code and analysis for the paper: "Model Complexity vs. Data Scarcity: Overfitting in Advanced Time Series Models for ENSO-Driven Coffee Price Forecasting."

The project investigates the impact of El Niño-Southern Oscillation (ENSO) on monthly coffee prices by performing a comparative "bake-off" between four forecasting models:

ARIMAX (Statistical Baseline)

LSTM-Small (Recurrent Neural Network)

TCN-Small (Temporal Convolutional Network)

TFT-Small (Temporal Fusion Transformer)

Key Findings
The central finding is a clear demonstration of catastrophic overfitting when applying complex, data-hungry deep learning models (TCN and TFT) to a real-world, small-scale dataset (n=600 training samples). Despite attempts at regularization, the most advanced models performed worst, yielding negative R² values (TFT R² = -13.47).

This repository serves as a cautionary case study on the limitations of applying state-of-the-art models directly to data-scarce problems, highlighting the critical trade-off between model complexity and data availability.
