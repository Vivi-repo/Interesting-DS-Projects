# train_models.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_evaluate_models(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Trains Decision Tree and Random Forest on input features X and target y.
    Returns trained models and train/validation splits.
    """

    # Ensure y is a 1D numpy array
    y = np.ravel(y)

    # -----------------------------
    # Train / Validation / Test Split
    # -----------------------------
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: train vs validation from remaining
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_temp
    )

    # -----------------------------
    # Class weights for imbalanced data
    # -----------------------------
    # Random Forest and Decision Tree can handle 'balanced' directly
    # But if you want manual weights:
    # classes = np.array([0,1])
    # from sklearn.utils.class_weight import compute_class_weight
    # class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    # class_weights = dict(zip(classes, class_weights_array))

    # -----------------------------
    # Initialize models
    # -----------------------------
    dt_model = DecisionTreeClassifier(
        max_depth=7,
        random_state=random_state,
        class_weight='balanced'
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=random_state,
        class_weight='balanced'
    )

    models = {
        'Decision Tree': dt_model,
        'Random Forest': rf_model
    }

    # -----------------------------
    # Train models
    # -----------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate on validation set
    # -----------------------------
    for name, model in models.items():
        y_pred = model.predict(X_val)
        print(f"--- {name} Validation Report ---")
        print(classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("\n")

    return models, X_train, X_val, y_train, y_val

