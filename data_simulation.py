# data_simulation.py
import pandas as pd
import numpy as np

def generate_synthetic_data(n=250000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        # Numeric features
        "income": np.random.normal(60000, 15000, n).clip(5000, 150000),
        "customer_age": np.random.randint(18, 80, n),
        "bank_months_count": np.random.randint(1, 240, n),
        "device_distinct_emails_8w": np.random.poisson(1.5, n),
        "zip_count_4w": np.random.poisson(3, n),
        "credit_risk_score": np.random.normal(650, 80, n).clip(300, 850),
        "session_length_in_minutes": np.random.exponential(8, n),
        "velocity_24h": np.random.exponential(2, n),
        "velocity_4w": np.random.exponential(5, n),
        "prev_address_months_count": np.random.randint(0, 200, n),
        "current_address_months_count": np.random.randint(0, 200, n),
        # Categorical features
        "employment_status": np.random.choice(["Employed", "Unemployed", "Student", "Retired"], n),
        "payment_type": np.random.choice(["Credit", "Debit", "ACH"], n),
        "housing_status": np.random.choice(["Own", "Rent", "Unknown"], n),
        "device_os": np.random.choice(["iOS", "Android", "Windows", "Other"], n),
        "email_is_free": np.random.choice(["Yes", "No"], n),
        "foreign_request": np.random.choice(["Yes", "No"], n),
        # Target variable
        "fraud_bool": np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("synthetic_fraud_data.csv", index=False)
    print("Synthetic data saved to synthetic_fraud_data.csv")
