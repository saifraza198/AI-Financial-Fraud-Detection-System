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
# Dashboard Metrics
total_transactions = len(df)
fraud_transactions = len(df[df["Class"] == 1])
legitimate_transactions = len(df[df["Class"] == 0])
accuracy = 99.96

st.header("📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    st.metric("Fraud Transactions", fraud_transactions)

with col3:
    st.metric("Legitimate Transactions", legitimate_transactions)

with col4:
    st.metric("Model Accuracy", f"{accuracy}%")

st.markdown("---")
# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("💳 AI Fraud Detection")

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.write("""
This application uses a Machine Learning model to detect fraudulent credit card transactions.
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Model Details")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Accuracy:** 99.96%")
st.sidebar.write("**Dataset:** Credit Card Fraud Detection")

st.sidebar.markdown("---")

st.sidebar.markdown("### 👨‍💻 Developed By")

st.sidebar.write("SAIF RAZA")
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
import matplotlib.pyplot as plt

st.subheader("📊 Fraud vs Legitimate Transactions")

fig, ax = plt.subplots(figsize=(6,4))

counts = df["Class"].value_counts()

ax.bar(
    ["Legitimate", "Fraud"],
    [counts[0], counts[1]]
)

ax.set_ylabel("Number of Transactions")
ax.set_title("Fraud Distribution")

st.pyplot(fig)
st.subheader("🥧 Fraud Percentage")

fig2, ax2 = plt.subplots(figsize=(5,5))

ax2.pie(
    counts,
    labels=["Legitimate", "Fraud"],
    autopct="%1.2f%%",
    startangle=90
)

ax2.axis("equal")

st.pyplot(fig2)
# Remove Class column before prediction
input_data = sample.drop("Class")

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict Sample Transaction"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"### Fraud Probability: {fraud_prob:.2f}%")
    st.progress(fraud_prob / 100)

    st.write(f"### Legitimate Probability: {legit_prob:.2f}%")

    st.info(f"Actual Class: {int(sample['Class'])}")

    if prediction == sample["Class"]:
        st.success("✔ Prediction Matched Actual Class")
    else:
        st.warning("❌ Prediction Did Not Match Actual Class")
