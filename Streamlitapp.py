import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import joblib

# === Load your trained SVC model ===
model = joblib.load("SVC.h5")

# === Define features and their types from your dataset ===
expected_features = {
    "category": "categorical",
    "gender": "categorical",
    "year": "numeric",
    "month": "numeric",
    "day": "numeric",
    "amount": "numeric",
    "city_population": "numeric",
    "destiance_c_to_m": "numeric",
    "year_birth": "numeric"
}

# === Streamlit Page Config ===
st.set_page_config(page_title="🔍 Smart Fraud Detection App", layout="wide")
st.title("🔐 Smart Fraud Detection with Support Vector Classifier")
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 24px;
    }
    .feature-label {
        font-weight: 600;
        margin-bottom: 4px;
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# === Sidebar: Upload or Manual Input ===
st.sidebar.header("📂 Upload Data or Use Manual Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
manual_mode = st.sidebar.checkbox("🛠️ Use Manual Input (No File)")

# === If user uploads a file ===
if uploaded_file and not manual_mode:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    st.sidebar.subheader("⚙️ Select Columns")
    target_col = st.sidebar.selectbox("Target column (fraud class)", df.columns)
    feature_cols = st.sidebar.multiselect("Feature columns", [col for col in df.columns if col != target_col])

    if st.sidebar.button("🚀 Predict and Evaluate"):
        X = df[feature_cols]
        y = df[target_col]

        st.subheader("📊 Evaluation Metrics")
        y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits=5))

        report = classification_report(y, y_pred, target_names=["Not Fraud", "Fraud"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
        disp.plot(ax=ax, cmap="Purples", values_format="d")
        st.pyplot(fig)

        st.success("✅ Evaluation Complete!")

        if st.checkbox("💾 Save Predictions"):
            df['prediction'] = y_pred
            st.download_button("Download as CSV", df.to_csv(index=False), "fraud_predictions.csv")

# === Manual Input Form ===
if manual_mode:
    st.markdown("---")
    st.subheader("🧮 Manually Predict a Transaction")
    st.markdown("Enter the values for the required features:")

    user_input = {}
    for feature, ftype in expected_features.items():
        st.markdown(f"<div class='feature-label'>{feature} ({ftype})</div>", unsafe_allow_html=True)
        if ftype == "numeric":
            user_input[feature] = st.number_input("", key=f"{feature}_num")
        else:
            user_input[feature] = st.text_input("", key=f"{feature}_cat")

    if st.button("🔎 Predict Now"):
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            label = "🚨 Fraud Detected!" if prediction == 1 else "✅ Not Fraudulent"
            color = "red" if prediction == 1 else "green"
            st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Prediction failed. Check inputs and feature types.\n\nError: {e}")

# === If neither mode selected ===
if not uploaded_file and not manual_mode:
    st.info("👈 Please upload a dataset or enable manual input to get started!")
