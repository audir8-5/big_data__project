# 🌦️ Smart Weather Prediction & Analytics System
Deployed on : https://weather-vision.onrender.com/ 
Note: May take few minutes render

An end-to-end machine learning system for short-term weather forecasting using live environmental data.  
The project focuses on model selection, error analysis, and real-time inference via a REST API.

---

## 🚀 Features
- End-to-end ML pipeline using real-world weather data
- Exploratory Data Analysis (EDA) on multivariate features
- Ridge Regression for improved generalization
- Error analysis across varying weather conditions
- Flask-based REST API for real-time predictions
- Interactive analytics-ready architecture

---

## 🧠 Machine Learning Approach
- **Input Features:** Temperature, humidity, pressure, wind speed  
- **Target:** Short-term temperature prediction  
- **Model:** Ridge Regression  
- **Why Ridge?**
  - Handles multicollinearity better than baseline linear regression
  - Achieved ~10–15% improvement in validation accuracy

---

## 🧪 Evaluation
- Validation-based performance comparison against baseline models
- Error analysis across different weather regimes
- Identified reliability limits and edge cases

---

## 🛠️ Tech Stack
- Python
- scikit-learn
- Flask
- OpenWeather API
- Pandas, NumPy

---

## 📡 API Overview
| Endpoint        | Description |
|-----------------|------------|
| `/api/train`    | Trains the ML model |
| `/api/predict`  | Returns real-time weather prediction |

---

## 📌 Status
Actively maintained and iterated to improve prediction reliability and robustness.
