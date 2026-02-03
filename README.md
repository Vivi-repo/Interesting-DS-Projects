Bank Fraud Detection Dashboard
----
An end-to-end system that simulates and detects dormant account reactivation risk and potential fraud in banking transactions. This project demonstrates full-stack data science, including data generation, preprocessing, machine learning, model interpretability, and an interactive dashboard.

Project Overview
---
This project simulates real-world banking scenarios where dormant accounts can become active and exhibit suspicious behaviors. The system is designed to:

Generate large, realistic synthetic datasets with features such as income, account history, device behavior, transaction velocity, and more.

Train interpretable machine learning models to detect potential fraud.

Identify key predictive patterns using feature importance and SHAP values.

Provide an interactive interface for data exploration, model training, insights visualization, and prediction of fraud probabilities.

The project emphasizes modular programming, maintainable code, and scalable system design while applying advanced data science techniques.

Features
---
1. Data Generation

Produce synthetic banking datasets with flexible size

Simulate realistic features and fraud scenarios

Export data for reuse in training and testing

2. Machine Learning Pipeline

Train Decision Tree and Random Forest classifiers

Handle class imbalance using weighted classes

Evaluate models using precision, recall, F1-score, accuracy, and confusion matrices

3. Fraud Insights

Identify the most predictive features

Visualize feature importance and fraud patterns with SHAP

Understand model decisions and highlight high-risk behaviors

4. Prediction Tool

Predict the probability of fraud for new account entries

Provide interpretable outputs to explain the prediction

5. Interactive Dashboard

Streamlit-based interface

Tabs for Data Explorer, Model Training, Fraud Insights, and Prediction Tool

Real-time visualizations and interactivity for technical and non-technical users

File Structure
----
bank_fraud_detection/
├─ app.py                  # Main Streamlit dashboard orchestrating the project
├─ generate_data.py        # Synthetic data generation script
├─ train_models.py         # Machine learning training and evaluation pipeline
├─ insights.py             # Feature importance and SHAP analysis
├─ predict.py              # Fraud prediction module
├─ notebooks/
│   └─ exploration.ipynb   # Jupyter notebook for experimentation and testing
├─ data/
│   └─ synthetic_data.csv  # Optional pre-generated synthetic dataset
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation


This structure separates data generation, modeling, insights, predictions, and presentation, following software engineering principles for modularity and scalability.

Tech Stack
---
Python – data processing, ML, and dashboard scripting

Pandas, NumPy – data manipulation and preprocessing

Scikit-learn, XGBoost – machine learning

Matplotlib, Seaborn, Plotly – visualizations

SHAP – model interpretability

Streamlit – interactive dashboard

Getting Started
---
1. Clone the repository

git clone https://github.com/Vivi-repo/Interesting-DS-Projects.git
cd bank_fraud_detection


2. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


3. Install dependencies

pip install -r requirements.txt


4. Run the dashboard

streamlit run app.py

Pipeline Summary
---
Data Layer: Generates synthetic banking datasets and preprocesses features

ML Layer: Trains multiple models, evaluates them, and stores trained artifacts

Insights Layer: Extracts feature importance, interprets predictions with SHAP, visualizes fraud patterns

Presentation Layer: Streamlit dashboard provides interactive tabs for exploration, training, insights, and predictions

Key Learnings
---
Designing an end-to-end data science system from raw data to dashboard

Handling highly imbalanced classification problems

Making ML models interpretable and actionable

Writing modular, maintainable code suitable for scaling

Building interactive dashboards for data visualization and decision support

Next Steps
----
Integrate real-world banking datasets for model validation

Include temporal patterns for better fraud detection

Extend dashboard for automated alerts and monitoring

Contact
---
Developed by Vethavarnaa Sundaraamoorthy as a demonstration of applied machine learning, systems programming, and interactive data visualization.

GitHub: https://github.com/Vivi-repo
