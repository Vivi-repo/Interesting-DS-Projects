# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import plotly.express as px

from data_simulation import generate_synthetic_data
from feature_engineering import engineer_features
from train_models import train_evaluate_models
from predict_pipeline import predict_fraud

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Bank Fraud Dashboard", layout="wide")
st.title("Bank Fraud Detection Dashboard")
st.markdown("""
Explore dormant account reactivation risk and fraud detection.  
Use the tabs to view data, train models, visualize insights, and predict fraud probabilities.
""")

# -----------------------------
# Initialize session state
# -----------------------------
if 'df' not in st.session_state:
    st.session_state['df'] = generate_synthetic_data(n=5000)
if 'df_encoded' not in st.session_state:
    st.session_state['df_encoded'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None

df = st.session_state['df']

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Data Explorer", "Model Training", "Fraud Insights", "Prediction Tool"])

# -----------------------------
# Tab 1: Data Explorer
# -----------------------------
with tabs[0]:
    st.header("Data Explorer")
    st.subheader("Sample of Generated Data")
    st.dataframe(st.session_state['df'].sample(10))

    st.subheader("Fraud Distribution")
    fraud_counts = st.session_state['df']['fraud_bool'].value_counts()
    fig = px.bar(
        x=fraud_counts.index,
        y=fraud_counts.values,
        labels={'x':'Fraud Label','y':'Count'},
        title="Fraud vs Non-Fraud Accounts"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Generate More Data")
    n_rows = st.slider("Number of rows to generate", 5000, 100000, 50000, step=5000)
    
    if st.button("Generate Data"):
        # Generate and store new dataset
        st.session_state['df'] = generate_synthetic_data(n=n_rows)
        
        # Show a success message instead of rerunning
        st.success(f"Generated {n_rows} rows of data! Scroll up to view the updated dataset.")

# -----------------------------
# Tab 2: Model Training
# -----------------------------
with tabs[1]:
    st.header("Model Training")
    if st.button("Prepare Data and Train Models"):
        df_encoded, numeric_features = engineer_features(st.session_state['df'])
        st.session_state['df_encoded'] = df_encoded

        X = df_encoded[numeric_features]
        y = df_encoded['fraud_bool']

        models, X_train, X_val, y_train, y_val = train_evaluate_models(X, y)
        st.session_state['models'] = models
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.success("Models trained successfully!")

    elif st.session_state['models']:
        st.info("Models already trained. You can go to Fraud Insights or Prediction Tool.")

# -----------------------------
# Tab 3: Fraud Insights
# -----------------------------
with tabs[2]:
    st.header("Fraud Insights")
    
    if st.session_state['models'] and st.session_state['X_train'] is not None:
        rf_model = st.session_state['models']['Random Forest']
        X_train = st.session_state['X_train']

        st.subheader("Top Features (SHAP Summary)")

        try:
            # Use a small sample to prevent freezing
            sample = X_train.sample(min(len(X_train), 2000), random_state=42)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer(sample)  # shap.Explanation object

            # Plot top 10 features safely
            plt.figure(figsize=(10,6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(plt.gcf())

            st.success("SHAP feature importance generated successfully!")

        except Exception as e:
            st.error(f"SHAP plot could not render: {e}")
            st.info("Make sure models are trained and feature columns match exactly.")

    else:
        st.warning("Train models first to view fraud insights.")

# -----------------------------
# Tab 4: Prediction Tool
# -----------------------------
with tabs[3]:
    st.header("Prediction Tool")
    if st.session_state['models'] and st.session_state['X_train'] is not None:
        rf_model = st.session_state['models']['Random Forest']
        X_train_cols = st.session_state['X_train'].columns

        st.subheader("Enter Account Details")
        with st.form("prediction_form"):
            # Numeric Inputs
            income = st.number_input("Income", value=60000)
            customer_age = st.number_input("Customer Age", value=35)
            bank_months_count = st.number_input("Bank Months Count", value=24)
            session_length = st.number_input("Session Length in Minutes", value=30)
            velocity_24h = st.number_input("Velocity 24h", value=1)
            velocity_4w = st.number_input("Velocity 4w", value=5)
            prev_address_months = st.number_input("Previous Address Months Count", value=60)
            current_address_months = st.number_input("Current Address Months Count", value=60)

            # Categorical Inputs
            employment_status = st.selectbox("Employment Status", ["Employed","Unemployed","Student","Retired"])
            payment_type = st.selectbox("Payment Type", ["Credit","Debit","ACH"])
            housing_status = st.selectbox("Housing Status", ["Own","Rent","Unknown"])
            device_os = st.selectbox("Device OS", ["iOS","Android","Windows","Other"])
            email_is_free = st.selectbox("Email Is Free", ["Yes","No"])
            foreign_request = st.selectbox("Foreign Request", ["Yes","No"])
            submit = st.form_submit_button("Predict Fraud Probability")

            if submit:
                input_df = pd.DataFrame([{
                    "income": income,
                    "customer_age": customer_age,
                    "bank_months_count": bank_months_count,
                    "session_length_in_minutes": session_length,
                    "velocity_24h": velocity_24h,
                    "velocity_4w": velocity_4w,
                    "prev_address_months_count": prev_address_months,
                    "current_address_months_count": current_address_months,
                    "employment_status": employment_status,
                    "payment_type": payment_type,
                    "housing_status": housing_status,
                    "device_os": device_os,
                    "email_is_free": email_is_free,
                    "foreign_request": foreign_request
                }])

                # Prediction
                prediction = predict_fraud(input_df, rf_model, X_train_cols)
                st.subheader("Predicted Fraud Probability")
                st.write(prediction)
    else:
        st.warning("Train models first to use the prediction tool.")
