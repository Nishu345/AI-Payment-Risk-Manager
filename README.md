# 💳 AI Payment Risk Detection & Fraud Alert System

An AI-powered FinTech prototype that analyzes digital payment transactions, calculates transaction risk, identifies suspicious behavior, and provides explainable fraud-risk alerts.

> **Project Type:** Academic / Portfolio Prototype  
> **Domain:** FinTech, Digital Payments, Machine Learning, Risk Analytics

---

## 📌 Project Overview

Digital payment platforms process a large number of transactions every day. Detecting suspicious transactions quickly is important for reducing financial risk and protecting customers.

This project demonstrates a machine-learning-based payment risk analysis system.

The system analyzes transaction characteristics such as:

- Transaction amount
- Transaction time
- Payment method
- Device type
- New device status
- Location mismatch
- Previous failed transactions
- Transaction frequency
- Account age
- Customer's average transaction amount
- Weekend activity

Based on these signals, the system generates:

- Risk Score
- Risk Level
- Fraud Probability
- Explainable Risk Factors
- Recommended Action

---

# 🎯 Problem Statement

Digital payment systems can be exposed to suspicious transaction behavior such as unusual transaction amounts, unfamiliar devices, location mismatches, repeated failed transactions, and unusually high transaction frequency.

A risk analysis system can help identify potentially suspicious transactions and prioritize them for additional verification or review.

---

# 💡 Solution

The project combines:

1. Synthetic transaction data generation
2. Data preprocessing
3. Exploratory data analysis
4. Machine learning classification
5. Explainable risk scoring
6. Interactive Streamlit dashboard

### Overall Workflow

```text
Synthetic Transaction Data
            ↓
Data Preprocessing
            ↓
Exploratory Data Analysis
            ↓
Machine Learning Model
            ↓
Fraud Probability
            ↓
Risk Scoring Engine
            ↓
Risk Level
            ↓
Risk Factors
            ↓
Recommended Action
            ↓
Streamlit Dashboard


🚀 Key Features
1. Synthetic Transaction Dataset

The project generates approximately 10,000 realistic synthetic payment transactions.

The dataset contains behavioral and transaction-level features that can influence payment risk.

2. Data Analytics

The system analyzes:

Total transactions
Fraud transactions
Genuine transactions
Fraud percentage
Average transaction amount
Payment method distribution
Fraud by payment method
Fraud by device type
Fraud by location
3. Machine Learning

A Random Forest classification model is used to estimate the probability that a transaction belongs to the suspicious/fraud class.

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC where appropriate
4. Explainable Risk Scoring

The system combines machine-learning output with interpretable transaction risk signals.

The final risk score ranges from:

0 – 100

Risk levels:

0 – 30     → LOW RISK
31 – 70    → MEDIUM RISK
71 – 100   → HIGH RISK
5. Risk Factors

The system can identify signals such as:

High transaction amount
Transaction amount significantly above customer average
New device
Location mismatch
Multiple failed transactions
High transaction frequency
New customer account
Unusual transaction hour
6. Recommended Action

Depending on the risk level, the system recommends actions such as:

LOW RISK

Allow the transaction and continue normal monitoring.

MEDIUM RISK

Apply additional verification and enhanced monitoring.

HIGH RISK

Require additional authentication and consider manual review.

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard containing:

KPI Metrics
Total Transactions
Fraud Detected
Fraud Rate
Average Risk Score
Interactive Analytics
Genuine vs Fraud Transactions
Transactions by Payment Method
Risk Distribution
Transaction Amount Distribution
Fraud by Device Type
Fraud by Location
Average Risk Score by Transaction Hour
Real-Time Transaction Analyzer

Users can enter a new transaction and receive:

Risk Score
Risk Level
Fraud Probability
Risk Factors
Recommended Action
🧠 Machine Learning Approach
Model

The primary model used in this project is:

Random Forest Classifier

Random Forest was selected because it:

Works well with tabular data
Can model non-linear relationships
Handles multiple transaction features
Provides a strong baseline for classification
Can be used with mixed feature types after preprocessing
📊 Why Accuracy Is Not Enough

Fraud detection is often an imbalanced classification problem.

For example, if most transactions are genuine, a model could achieve high accuracy while still missing many fraudulent transactions.

Therefore, this project also considers:

Precision

How many transactions predicted as fraud were actually fraud.

Recall

How many actual fraudulent transactions were successfully identified.

F1-score

A combined measure of precision and recall.

For fraud detection, recall can be particularly important because missing a suspicious transaction can create financial and customer-protection risks.

However, precision is also important because excessive false alerts can negatively affect genuine customers.

📁 Project Structure
AI-Payment-Risk-Manager/
│
├── data/
│   ├── raw/
│   │   └── transactions.csv
│   │
│   └── processed/
│       └── processed_transactions.csv
│
├── model/
│   └── fraud_model.pkl
│
├── src/
│   ├── data_generator.py
│   ├── data_preprocessing.py
│   ├── explore_data.py
│   ├── train_model.py
│   └── risk_engine.py
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🛠️ Technology Stack
Programming
Python
Data Analytics
Pandas
NumPy
Machine Learning
Scikit-learn
Random Forest Classifier
Visualization
Plotly
Matplotlib
Web Application
Streamlit
Model Persistence
Joblib
Development
VS Code
Version Control
Git
GitHub
📦 Dataset

This project uses synthetic/demo transaction data generated specifically for this portfolio project.

No real customer payment information is used.

Example transaction attributes include:

transaction_id
customer_id
amount
transaction_hour
payment_method
location
device_type
is_new_device
location_mismatch
previous_failed_transactions
transactions_last_24h
account_age_days
customer_avg_transaction_amount
is_weekend
fraud_label

The synthetic dataset is designed to contain relationships between transaction behavior and suspicious activity rather than assigning labels completely randomly.


⚙️ Installation
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd AI-Payment-Risk-Manager
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment
Windows PowerShell
.\venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
▶️ How to Run
Step 1 — Generate Dataset
python src/data_generator.py
Step 2 — Preprocess Data
python src/data_preprocessing.py
Step 3 — Explore Dataset
python src/explore_data.py
Step 4 — Train Model
python src/train_model.py
Step 5 — Run Risk Engine Test
python src/risk_engine.py
Step 6 — Start Streamlit Dashboard
streamlit run app.py

The application will be available locally at:

http://localhost:8501
🔎 Example Risk Analysis

Example transaction:

Amount: ₹85,000
New Device: Yes
Location Mismatch: Yes
Previous Failed Transactions: 3
Transactions Last 24 Hours: 9
Transaction Hour: 2

The system may identify several risk signals:

New device detected
Location mismatch detected
Very high transaction amount
Previous failed transactions
High transaction frequency
Unusual late-night transaction

The final risk score and fraud probability are generated by the trained model and risk engine.

📸 Screenshots

Screenshots demonstrating the project will be added here.

Dashboard
screenshots/dashboard.png
Analytics
screenshots/analytics.png
High Risk Transaction
screenshots/high-risk.png
Medium / Low Risk Transaction
screenshots/low-medium-risk.png
Model Evaluation
screenshots/model-evaluation.png
🔐 FinTech Relevance

This project demonstrates concepts relevant to digital payment platforms:

Payment risk analysis
Transaction monitoring
Fraud detection
Customer protection
Risk-based authentication
Behavioral analysis
Data-driven decision making

The project is inspired by real-world payment risk management concepts but is not an official system of Razorpay or any other payment company.

🏦 Razorpay Internship Relevance

A payment platform needs systems capable of analyzing transactions and identifying potentially suspicious behavior.

This project demonstrates my understanding of:

Digital payment transaction data
Machine learning classification
Risk scoring
Explainable risk signals
Transaction monitoring
Interactive data analytics
Python-based FinTech application development

It is intended as a portfolio demonstration and not as a claim of implementing any proprietary payment company's internal system.

⚠️ Limitations

This project has several limitations:

The dataset is synthetic.
The model has not been trained on confidential real-world payment data.
The risk scoring mechanism is a prototype.
Real payment systems require much larger and continuously updated datasets.
Production fraud detection requires real-time infrastructure.
Fraud patterns can change over time.
False positives and false negatives require continuous monitoring.
Production systems require strong security, privacy, monitoring, and compliance controls.
🔮 Future Improvements

Possible future improvements include:

Real-time transaction streaming
REST API integration
Real payment gateway integration in a controlled sandbox
Advanced anomaly detection
XGBoost/LightGBM comparison
Model explainability using SHAP
Real-time alert notifications
Customer behavioral profiles
Model drift monitoring
Feature engineering from transaction history
Database integration
Cloud deployment
Authentication and role-based access
Fraud analyst review workflow
🧪 Testing

The application can be tested using multiple transaction scenarios:

Scenario 1

Normal low-value transaction.

Scenario 2

Moderately unusual transaction.

Scenario 3

High-value transaction with a new device.

Scenario 4

Location mismatch with previous failed transactions.

Scenario 5

Multiple suspicious signals occurring together.

The system is not designed to force predetermined outputs. The final prediction depends on the trained model and risk signals.

📈 Project Impact

The prototype demonstrates how machine learning and explainable risk signals can be combined to support transaction monitoring.

Potential benefits of such a system include:

Prioritizing suspicious transactions
Supporting additional authentication
Helping fraud analysts investigate alerts
Improving transaction monitoring
Supporting customer protection

👩‍💻 Skills Demonstrated
Python
Pandas
NumPy
Scikit-learn
Machine Learning
Classification
Random Forest
Risk Scoring
Fraud Detection Concepts
Data Analytics
Data Visualization
Streamlit
Plotly
Git
GitHub
FinTech

📌 Disclaimer

This project is an academic and portfolio prototype created for educational and demonstration purposes.

It uses synthetic transaction data and should not be considered a production fraud detection system.

The predictions and risk scores are not guaranteed to identify actual fraudulent transactions.

No real customer payment information is used in this project.

👤 Author:Nishu Shakya

B.Tech — Computer Science & Engineering

⭐ Project Summary

AI Payment Risk Detection & Fraud Alert System is a Python and machine-learning-based FinTech prototype that analyzes synthetic digital payment transactions, estimates fraud probability, generates an explainable 0–100 risk score, classifies transactions into risk levels, and provides recommended actions through an interactive Streamlit dashboard.
