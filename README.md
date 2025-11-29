# 📈 Profit Prediction Using ARIMA, SARIMA and LSTM Models in Time Series Forecasting

This project presents a comprehensive comparative study of three time-series forecasting models — **ARIMA**, **SARIMA**, and **LSTM** — for predicting monthly business profit using historical retail sales data.  
The goal is to evaluate which forecasting technique performs best in accuracy, stability, and trend capturing ability for real-world financial prediction.

📄 **Full project documentation available in the repository (PDF Report included)**  
🧠 **Dataset source:** Excel BI Analytics Sales & Financial Transactions Dataset  (https://excelbianalytics.com/wp/wp-content/uploads/2020/09/2m-Sales-Records.zip)
📍 **Implementation Platform:** Google Colab

---

## 🎯 Project Objectives
- Preprocess financial transaction data and convert daily profit to a monthly time-series.
- Perform EDA and stationarity analysis using **ADF** and **KPSS** tests.
- Build and evaluate **ARIMA**, **SARIMA**, and **LSTM** models.
- Compare model performance using real forecasting error metrics.
- Identify the best-performing model for profit forecasting.

---

## 🧪 Exploratory Data Analysis (EDA)
- Converted daily transaction records into **monthly aggregated profit**.
- Handled missing and invalid values and removed anomalies (e.g., September 2020 extreme spike).
- Visualized **time-series trend & seasonality** using line plots.
- Applied **ADF (Augmented Dickey–Fuller)** & **KPSS** tests to check stationarity.
- Performed **ACF & PACF** plot analysis to identify ARIMA/SARIMA parameters.
- Identified clear **seasonal patterns**, confirming suitability for SARIMA.
---

## 🧪 Models Implemented

| Model | Description | Best Use Case |
|-------|-------------|----------------|
| **ARIMA** | Non-seasonal statistical model using autoregression, differencing & moving averages | Simple linear patterns |
| **SARIMA** | Seasonal ARIMA model with seasonal trend capturing ability | Strong seasonal yearly repetition |
| **LSTM** | Deep learning neural network for sequential data | Non-linear & long-dependency learning |

---

## 📊 Performance Comparison Results

Based on evaluation metrics such as **RMSE, MAE, MAPE, Min-Max Error, and Accuracy**, SARIMA clearly outperformed ARIMA and LSTM.

| Metric | ARIMA | SARIMA | LSTM |
|--------|--------|--------|--------|
| RMSE | 1.3875E8 | **1.0360E8** | 1.4289E8 |
| MAE | 9.77E7 | **7.98E7** | 1.128E8 |
| MAPE | 0.016226 | **0.013102** | 0.018568 |
| Min-Max | 0.185884 | **0.151914** | 0.214715 |
| Accuracy | 98.38% | **98.69%** | 98.14% |

📌 **Conclusion:** *SARIMA is the best performing model* due to strong yearly seasonality in the dataset. LSTM performed well but requires larger data volume and more hyperparameter tuning.


---

## 🧠 Methodology Overview

### ✔ Steps followed:
1. Data cleaning & anomaly correction (e.g., September 2020 outlier corrected)
3. Monthly resampling using `resample('M').sum()`
4. ADF & KPSS stationarity tests
5. Log transformation & differencing
6. ACF / PACF analysis for ARIMA & SARIMA parameter tuning
7. LSTM sequence generation using sliding windows (12 months)
8. Train–test split (80/20)
9. Performance evaluation and visualization

### Technologies Used
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Statsmodels (ARIMA & SARIMA)
- TensorFlow / Keras (LSTM)
- Scikit-learn (Scaling & Metrics)

---

## 🔮 Future Enhancements
- Deploy interactive prediction UI using Streamlit / Flask
- Transformer-based forecasting
- Larger dataset for DL improvement

---

