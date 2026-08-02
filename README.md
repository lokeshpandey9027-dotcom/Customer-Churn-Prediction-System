# Customer-Churn-Prediction-System
## Telecom Churn Prediction & Intelligence Portal

An end-to-end Machine Learning web application and Power BI-styled analytics dashboard built with **FastAPI**, **MongoDB Atlas**, and **Scikit-Learn**. This system evaluates customer attrition risk in real-time, generates automated AI retention strategies, handles bulk batch predictions, and tracks user activity logs.

---

## 🚀 Key Features & Capabilities

- **Executive BI Dashboard:** Visual overview of key business metrics, customer tenure, monthly charges, and churn distribution charts (powered by Chart.js).
- **Real-Time Predictive & Retention Engine:** Interactive input form for customer demographics and service metrics to predict churn risk (`Churn` vs. `No Churn`) with probability scores.
- **Smart Retention Actionable Insights:** Automatically suggests targeted business interventions (e.g., contract upgrades, security bundles, onboarding outreach) based on customer risk profiles.
- **Bulk CSV Data Upload:** Upload batch CSV files to run mass predictions and log historical records automatically.
- **Activity Tracking & Audit Log:** Automatically saves and displays recent evaluation history with timestamps (recorded in Indian Standard Time - IST) backed by **MongoDB Atlas**.
- **Secure Authentication & Management:** Protected workspace secured via environment variables (`.env`) alongside a secure user login and registration system.

---

## 🛠️ Tech Stack

- **Backend & API:** FastAPI, Uvicorn, Motor (Async MongoDB Driver)
- **Frontend / UI:** HTML5, CSS3, JavaScript, Jinja2 Templates, Bootstrap 5, Chart.js, Google Fonts (*Plus Jakarta Sans*)
- **Machine Learning & Data Science:** Scikit-Learn, Random Forest Classifier, Pandas, NumPy, Pickle, Jupyter Notebook
- **Database:** MongoDB Atlas (`customer_churn_database`)

---

## 📂 Project Structure

```text
📦 Customer Churn Intelligence Portal
│
├── fastapi_app.py                 # Main FastAPI application, routes & MongoDB integration
├── CustomerChurnPrediction.ipynb  # Jupyter Notebook for EDA, preprocessing, and model training
├── best_model.pkl                 # Trained Machine Learning model (Random Forest)
├── encoder.pkl                    # Pre-fitted categorical feature encoders
├── scaler.pkl                     # Feature standard scaler
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original Telco Customer Churn dataset
├── .env                           # Environment variables (Credentials & MongoDB URI)
├── requirements.txt               # Python package dependencies
│
└── templates/
    ├── index.html                 # Main BI dashboard, prediction form, and tracking table UI
    └── login.html                 # Secure authentication & user registration UI
