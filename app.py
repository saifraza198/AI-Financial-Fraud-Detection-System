import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load Model
# -------------------------
model = joblib.load("fraud_model.pkl")

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("creditcard.csv")   # Change path if needed

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Financial Fraud Detection System")
st.write("Predict whether a transaction is fraudulent or legitimate.")

st.markdown("---")

# -------------------------
# Sample Selection
# -------------------------
sample_index = st.number_input(
    "Choose Transaction Index",
    min_value=0,
    max_value=len(df)-1,
    value=0,
    step=1
)

sample = df.iloc[sample_index]

st.subheader("Sample Transaction")

st.dataframe(sample)

# Remove Class column before prediction
input_data = sample.drop("Class")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Sample Transaction"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability : {probability[1]*100:.2f}%")
    st.write(f"Legitimate Probability : {probability[0]*100:.2f}%")

    st.write("Actual Class :", int(sample["Class"]))
