# 🏦 Credit Spectrum AI

**Intelligent Credit Eligibility Prediction Platform**

🔗 **Live Demo:** https://creditspectrumai.streamlit.app/

---

## 📖 Overview

**Credit Spectrum AI** is an interactive, AI-powered loan eligibility prediction platform built with **Streamlit** and **XGBoost**. It analyzes an applicant's financial and credit profile — income, FICO score, debt-to-income ratio, credit history, and more — to predict whether they meet a lender's credit policy, delivering real-time, explainable decisions through a modern, visually rich interface.

The platform goes beyond a simple prediction script — it's a full end-to-end credit intelligence dashboard, combining exploratory data analysis, model performance analytics, and an interactive prediction engine into one seamless experience.

---

## ✨ Features

- 🧠 **AI-Powered Predictions** — Real-time credit eligibility assessment powered by a tuned XGBoost classifier
- 📊 **Interactive Dashboard** — Explore applicant data through dynamic Plotly visualizations (distributions, correlations, approval trends)
- 📈 **Model Analytics** — Accuracy, ROC AUC, ROC curve, confusion matrix, and feature importance, all visualized live
- 🔮 **AI Credit Oracle** — A guided, multi-section prediction form for entering applicant details
- 🎨 **Dynamic Theming** — The interface visually reacts to the prediction outcome (approval / decline)
- 📁 **Dataset Explorer** — Searchable, downloadable view of the underlying loan dataset
- 📥 **Download Center** — Export the trained model and dataset directly from the app
- 🧭 **Model Transparency** — A full walkthrough of the model development pipeline, from data cleaning to deployment

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App Framework | Streamlit |
| Machine Learning Model | XGBoost |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly |
| Language | Python |

---

## 📂 Project Structure

```
Credit-Spectrum-AI/
│
├── app.py                              # Main Streamlit application
├── model.pkl                           # Trained XGBoost model
├── loan_data.csv                       # Loan/credit dataset
├── requirements.txt                    # Project dependencies
├── Loan_Eligibility_Prediction.ipynb   # Model training & experimentation notebook
└── README.md                           # Project documentation
```

---

## 📊 Dataset

The model is trained on a loan/credit dataset containing borrower demographic and financial attributes, including:

- Loan purpose
- Interest rate & monthly installment
- Annual income (log-transformed)
- Debt-to-income ratio
- FICO credit score
- Credit line history & revolving balance/utilization
- Recent credit inquiries, delinquencies, and public records

The target variable indicates whether an applicant meets the lender's credit policy.

---

## ⚙️ Model Pipeline

1. Data collection & cleaning
2. Categorical encoding
3. Feature scaling
4. Class balancing (SMOTE)
5. Train/test split
6. Hyperparameter tuning
7. Model training (Logistic Regression, Decision Tree, Random Forest, XGBoost — compared)
8. Evaluation (Accuracy, ROC AUC, Confusion Matrix)
9. Deployment of the best-performing model (XGBoost)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Ruthwik24/Credit-Spectrum-AI.git
cd Credit-Spectrum-AI
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📈 Model Performance

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | ~85.7% | ~93.4% |
| Decision Tree | ~98.8% | ~98.3% |
| Random Forest | ~98.4% | ~99.7% |
| **XGBoost (Deployed)** | **~98.9%** | **~99.8%** |

---

## 🔮 Future Enhancements

- SHAP-based per-prediction explainability
- User authentication & prediction history storage
- REST API endpoint for external integrations
- Multi-language support
- Mobile-responsive UI improvements

---

## 👨‍💻 Author

Built and maintained by B Ruthwik Reddy

🔗 Live Website: https://creditspectrumai.streamlit.app/
📧 Contact: ruthwikbhoompally@gmail.com

---

## 📄 License

This project currently has no license. All rights are reserved by the author — please contact before reusing or distributing.
