# explainability.py
import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values[1], X_val, plot_type="bar", max_display=20)
