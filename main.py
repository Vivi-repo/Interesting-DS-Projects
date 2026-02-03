# main.py
import pandas as pd
from data_simulation import generate_synthetic_data
from feature_engineering import engineer_features
from train_models import train_evaluate_models
from explainability import shap_analysis
from predict_pipeline import predict_fraud

# 1. Generate or load data
df = generate_synthetic_data()
df_encoded, numeric_features = engineer_features(df)
X = df_encoded[numeric_features]
y = df_encoded['fraud_bool']

# 2. Train models
models, X_train, X_val, y_train, y_val = train_evaluate_models(X, y)

# 3. SHAP analysis on Random Forest
rf_model = models['Random Forest']
shap_analysis(rf_model, X_val)

# 4. Test prediction function
sample_new = df.sample(5).drop(columns='fraud_bool')
predictions = predict_fraud(sample_new, rf_model, X_train.columns)
print(predictions)
