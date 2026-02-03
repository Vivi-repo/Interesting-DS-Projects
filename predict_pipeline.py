# predict_pipeline.py
import pandas as pd
from feature_engineering import engineer_features, categorical_features

def predict_fraud(new_accounts, model, X_train_columns):
    # Feature engineering
    new_accounts['velocity_ratio'] = new_accounts['velocity_24h'] / (new_accounts['velocity_4w'] + 1)
    new_accounts['address_change_ratio'] = new_accounts['current_address_months_count'] / (new_accounts['prev_address_months_count'] + 1)
    new_accounts['session_per_bank_month'] = new_accounts['session_length_in_minutes'] / (new_accounts['bank_months_count'] + 1)
    
    # One-hot encode
    new_accounts_encoded = pd.get_dummies(new_accounts, columns=categorical_features, drop_first=True)
    
    # Align with training features
    new_accounts_encoded = new_accounts_encoded.reindex(columns=X_train_columns, fill_value=0)
    
    # Predict
    new_accounts['fraud_probability'] = model.predict_proba(new_accounts_encoded)[:,1]
    return new_accounts[['fraud_probability']]
