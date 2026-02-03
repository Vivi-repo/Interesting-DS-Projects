# feature_engineering.py
import pandas as pd

categorical_features = ["employment_status", "payment_type", "housing_status", "device_os", "email_is_free", "foreign_request"]

def engineer_features(df):
    # Behavioral ratios
    df['velocity_ratio'] = df['velocity_24h'] / (df['velocity_4w'] + 1)
    df['address_change_ratio'] = df['current_address_months_count'] / (df['prev_address_months_count'] + 1)
    df['session_per_bank_month'] = df['session_length_in_minutes'] / (df['bank_months_count'] + 1)
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    numeric_features = [col for col in df_encoded.columns if col != "fraud_bool"]
    
    return df_encoded, numeric_features
