# Project-Time-Series
Advanced Time Series Forecasting with LSTM and Transformer Models (with Ablation Study)

📌 1. Project Overview

This project focuses on building an advanced time series forecasting system using both traditional and modern deep learning approaches.
The goal is to:

Automatically generate a realistic multivariate time-series dataset

Perform full data preprocessing & windowing

Train two forecasting models:

Baseline LSTM

Transformer-based model with self-attention

Evaluate them using:

RMSE (Root Mean Squared Error)

MASE (Mean Absolute Scaled Error)

Conduct a full Ablation Study:

With different attention heads

With different encoder layers

This project demonstrates how deep learning and attention mechanisms improve forecasting accuracy over traditional sequence models.

📌 2. Features of This Project

✔ Synthetic multivariate dataset (trend + daily & weekly seasonality + noise)
✔ Fully clean & imputed time-series
✔ MinMax scaling
✔ Sliding-window supervised dataset
✔ LSTM baseline forecasting
✔ Transformer encoder forecasting (self-attention)
✔ RMSE & MASE evaluation
✔ Graphs: prediction comparison
✔ Ablation study for Transformer architecture
✔ Results stored as CSV
✔ Fully modular & well-documented code

📌 3. Architecture Diagram (Simple)
Dataset → Preprocessing → Window Creation → Models
                                      ↙         ↘
                                   LSTM       Transformer
                                      ↘         ↙
                                 Evaluation (RMSE, MASE)
                                             ↓
                                      Ablation Study

📌 4. Dataset Description

The dataset is generated programmatically using:

Linear Trend

Daily Seasonality

Weekly Seasonality

Noise

Correlated Exogenous Features

Random Missing Values → cleaned via forward/backward fill

Columns:

Column	Description
feature_1	Daily seasonality + noise
feature_2	Slow trend + noise
target	Weighted mix of trend, seasonality, exogenous effects

Dataset length default = 1500 timestamps with hourly frequency.

📌 5. Data Preprocessing Pipeline

Forward-fill & backward-fill missing values

Train-val-test split (70% / 15% / 15%)

MinMax scaling on all features

Sliding window creation

X = past 48 values of all features  
y = next value of target  


Conversion to TensorFlow tf.data.Dataset objects

📌 6. Models Used
🔹 LSTM Baseline

Multi-layer LSTM

Dense regression head

Adam optimizer

Early stopping

Predicts 1-step ahead

🔹 Transformer Encoder Model

Implemented from scratch:

Feature projection → d_model

Sinusoidal positional encoding

Multi-head self-attention

Feed-forward network

Layer norm + residual connections

GlobalAveragePooling1D

Dense(1) output

This is the highlight of the project.

📌 7. Evaluation Metrics
Metric	Meaning
RMSE	Measures typical size of errors
MASE	Scaled error compared to naive forecast

Both are computed on inverse-transformed predictions to ensure fair comparison.

📌 8. Ablation Study

We vary:

Number of attention heads → (2, 4)

Number of encoder layers → (1, 2)

For each combination, we:

Train a new Transformer model

Evaluate RMSE & MASE

Append results into results/transformer_ablation.csv

Example output:

Model	Heads	Layers	RMSE	MASE
Transformer_h2_L1	2	1	X	X
Transformer_h4_L2	4	2	X	X
📌 9. Project File Structure
project/
│── Timeseries.py
│── README.md
│── results/
│    └── transformer_ablation.csv
│── figures/
│    └── ltsm_vs_transformer_plot.png

📌 10. How to Run the Project
Install Dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib

Run the full project
python Timeseries.py


All results will be printed + saved automatically.

📌 11. Outputs Generated

✔ Model summaries (LSTM + Transformer)
✔ Graph comparing predictions
✔ RMSE + MASE values
✔ Ablation study CSV
✔ Synthetic dataset plotted (target series)

📌 12. Key Findings (Example Summary)

Transformer model gave lower RMSE than LSTM

Ablation study showed:

Increasing attention heads improves performance

More encoder layers help but may overfit for small datasets

Self-attention captured long-term dependencies better than LSTM

📌 13. Conclusion

This project demonstrates how attention-based architectures outperform classical LSTM models for time series forecasting. The Transformer effectively learns complex temporal dependencies using self-attention and provides better generalization on unseen data.

The ablation study provides insights into how architecture choices (heads, layers) influence performance.
