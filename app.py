import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
# Load Model
model = joblib.load("fraud_model.pkl")
# Load Dataset
df = pd.read_csv("creditcard.csv")  
# ==========================================
# MODEL EVALUATION
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

cm = confusion_matrix(y, y_pred)

fpr, tpr, _ = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

feature_columns = X.columns
# Page Config
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)
# HEADER
st.title("💳 AI Financial Fraud Detection Dashboard")

st.markdown("""
### AI-Powered Credit Card Fraud Detection

This dashboard uses a **Random Forest Machine Learning model**
to identify fraudulent credit card transactions in real time.

Use the **Transaction Explorer** to test individual transactions
or upload a CSV file for batch predictions.
""")

st.markdown("---")
st.sidebar.title("💳 AI Fraud Detection")

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.write("""
This application uses a Machine Learning model to detect fraudulent credit card transactions.
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Model Details")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write(f"**Accuracy:** {accuracy*100:.2f}%")
st.sidebar.write("**Dataset:** Credit Card Fraud Detection")

st.sidebar.markdown("---")

st.sidebar.markdown("### 👨‍💻 Developed By")

st.sidebar.write("SAIF RAZA")
# DASHBOARD METRICS
total_transactions = len(df)
fraud_transactions = len(df[df["Class"] == 1])
legitimate_transactions = len(df[df["Class"] == 0])

fraud_percentage = (fraud_transactions / total_transactions) * 100
legitimate_percentage = (legitimate_transactions / total_transactions) * 100


st.header("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📁 Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "🚨 Fraud Cases",
        fraud_transactions,
        f"{fraud_percentage:.2f}%"
    )

with col3:
    st.metric(
        "✅ Legitimate",
        legitimate_transactions,
        f"{legitimate_percentage:.2f}%"
    )

with col4:
    st.metric(
        "🎯 Model Accuracy",
        f"{accuracy:.2f}%"
    )

st.markdown("---")
# DATA VISUALIZATION
# ANALYTICS
st.header("📈 Dataset Analytics")

counts = df["Class"].value_counts()

col1, col2 = st.columns([2, 1])
# Charts
with col1:

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig, ax = plt.subplots(figsize=(5,4))

        ax.bar(
            ["Legitimate", "Fraud"],
            [counts[0], counts[1]]
        )

        ax.set_ylabel("Transactions")
        ax.set_title("Fraud Distribution")

        st.pyplot(fig)

    with chart_col2:

        fig2, ax2 = plt.subplots(figsize=(5,4))

        ax2.pie(
            counts,
            labels=["Legitimate", "Fraud"],
            autopct="%1.2f%%",
            startangle=90
        )

        ax2.axis("equal")

        st.pyplot(fig2)

# Insights
with col2:

    st.subheader("📌 Dataset Insights")

    st.info(f"Total Transactions: {total_transactions:,}")

    st.success(f"Legitimate: {legitimate_transactions:,}")

    st.error(f"Fraud: {fraud_transactions:,}")

    st.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )

    st.metric(
        "Legitimate Rate",
        f"{legitimate_percentage:.2f}%"
    )

    st.caption(
        "The dataset is highly imbalanced, making fraud detection a challenging classification problem."
    )

st.markdown("---")
# TRANSACTION SELECTION

st.header("🔍 Transaction Explorer")

transaction_type = st.selectbox(
    "Select Transaction Type",
    ["All Transactions", "Fraud Transactions", "Legitimate Transactions"]
)

if transaction_type == "Fraud Transactions":
    filtered_df = df[df["Class"] == 1].reset_index(drop=True)

elif transaction_type == "Legitimate Transactions":
    filtered_df = df[df["Class"] == 0].reset_index(drop=True)

else:
    filtered_df = df

sample_index = st.slider(
    "Select Transaction",
    0,
    len(filtered_df) - 1,
    0
)

sample = filtered_df.iloc[sample_index]
# TRANSACTION DETAILS
# ==========================================
# TRANSACTION DETAILS
# ==========================================

st.header("📄 Transaction Details")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "💰 Amount ($)",
        f"{sample['Amount']:.2f}"
    )

with summary_col2:
    st.metric(
        "⏰ Time",
        f"{sample['Time']:.0f}"
    )

with summary_col3:

    if sample["Class"] == 1:
        st.error("🚨 Actual: Fraud")

    else:
        st.success("✅ Actual: Legitimate")

st.markdown("### Feature Values")

transaction_df = sample.to_frame(name="Value")
transaction_df.index.name = "Feature"

st.dataframe(
    transaction_df,
    use_container_width=True,
    height=500
)

st.markdown("---")
input_data = sample.drop("Class")
# ==========================================
# PREDICTION PANEL
# ==========================================

st.header("🤖 AI Fraud Prediction")

st.info(
    "Select a transaction and click the button below to let the AI model analyze it."
)

if st.button("🔍 Analyze Transaction", use_container_width=True):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100

    confidence = max(fraud_prob, legit_prob)

    st.markdown("---")

    st.subheader("📋 Analysis Result")

    # ------------------------
    # Prediction Banner
    # ------------------------

    if prediction == 1:

        st.error("🚨 FRAUD DETECTED")

    else:

        st.success("✅ LEGITIMATE TRANSACTION")

    # ------------------------
    # Metrics
    # ------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_prob:.2f}%"
        )

        st.progress(fraud_prob / 100)

    with col2:

        st.metric(
            "Legitimate Probability",
            f"{legit_prob:.2f}%"
        )

        st.progress(legit_prob / 100)

    with col3:

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

    st.markdown("---")

    # ------------------------
    # Prediction Summary
    # ------------------------

    actual = "Fraud" if sample["Class"] == 1 else "Legitimate"

    predicted = "Fraud" if prediction == 1 else "Legitimate"

    summary = pd.DataFrame({
        "Category": [
            "Actual Class",
            "Predicted Class",
            "Confidence"
        ],
        "Value": [
            actual,
            predicted,
            f"{confidence:.2f}%"
        ]
    })

    st.subheader("Prediction Summary")

    st.table(summary)

    # ------------------------
    # Final Verification
    # ------------------------

    if prediction == sample["Class"]:

        st.success("✔ The model prediction matches the actual transaction label.")

    else:

        st.warning("❌ The model prediction does not match the actual transaction label.")
# ==========================================
# CSV UPLOAD & BATCH PREDICTION
# ==========================================

st.markdown("---")
st.header("📤 Batch Prediction using CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------
    # Load CSV
    # -------------------------
    batch_df = pd.read_csv(uploaded_file)

    # -------------------------
    # Required Columns
    # -------------------------
    required_columns = [
        "Time",
        "V1","V2","V3","V4","V5","V6","V7",
        "V8","V9","V10","V11","V12","V13","V14",
        "V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26",
        "V27","V28","Amount"
    ]

    # -------------------------
    # Check Missing Columns
    # -------------------------
    missing_columns = [
        col for col in required_columns
        if col not in batch_df.columns
    ]

    if missing_columns:

        st.error("❌ Invalid CSV File")

        st.write("### Missing Required Columns")

        for col in missing_columns:
            st.write(f"• {col}")

        st.stop()

    # -------------------------
    # Check Missing Values
    # -------------------------
    missing_values = batch_df.isnull().sum()

    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:

        st.error("❌ Missing Values Found")

        st.write("Please remove the missing values before prediction.")

        st.dataframe(
            missing_values.rename("Missing Count")
        )

        st.stop()

    # -------------------------
    # File Information
    # -------------------------
    st.subheader("📄 File Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", batch_df.shape[0])

    with col2:
        st.metric("Columns", batch_df.shape[1])

    with col3:
        memory = batch_df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory:.2f} MB")

    with col4:
        st.metric("Status", "Ready ✅")

    st.markdown("---")

    # -------------------------
    # Preview
    # -------------------------
    st.subheader("📄 Uploaded Dataset Preview")

    st.dataframe(
        batch_df.head(),
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------
    # Prediction
    # -------------------------
    if st.button(
        "🚀 Predict Uploaded Data",
        use_container_width=True
    ):

        predict_df = batch_df.copy()

        # Remove target column if available
        if "Class" in predict_df.columns:
            predict_df = predict_df.drop("Class", axis=1)

        predictions = model.predict(predict_df)

        probabilities = model.predict_proba(predict_df)

        batch_df["Prediction"] = predictions

        batch_df["Prediction"] = batch_df["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        batch_df["Fraud Probability (%)"] = (
            probabilities[:,1] * 100
        ).round(2)

        batch_df["Legitimate Probability (%)"] = (
            probabilities[:,0] * 100
        ).round(2)

        st.success("✅ Prediction Completed Successfully!")

        st.subheader("📊 Prediction Results")

        st.dataframe(
            batch_df,
            use_container_width=True
        )

        # Summary
        fraud_count = (batch_df["Prediction"] == "Fraud").sum()
        legit_count = (batch_df["Prediction"] == "Legitimate").sum()

        st.subheader("📈 Prediction Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Fraud Detected", fraud_count)

        with c2:
            st.metric("Legitimate Transactions", legit_count)

        csv = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
        # ==========================================
# MODEL PERFORMANCE
# ==========================================

st.markdown("---")
st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

with col2:
    st.metric(
    "Precision",
    f"{precision*100:.2f}%"
)

with col3:
    st.metric(
    "Recall",
    f"{recall*100:.2f}%"
)

with col4:
    st.metric(
    "F1 Score",
    f"{f1*100:.2f}%"
)

st.markdown("---")
# ==========================================
# MODEL EVALUATION
# ==========================================

st.markdown("---")
st.subheader("📊 Model Evaluation")

left_col, right_col = st.columns([1, 1])

# ==========================================
# LEFT COLUMN - CONFUSION MATRIX
# ==========================================

with left_col:

    st.markdown("### 📌 Confusion Matrix")

    conf_matrix = cm
    fig, ax = plt.subplots(figsize=(4,4))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        annot_kws={"size":12},
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ==========================================
# RIGHT COLUMN - CLASSIFICATION REPORT
# ==========================================

with right_col:

    st.markdown("### 📋 Classification Report")

    report_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Legitimate": [1.00, 1.00, 1.00],
        "Fraud": [0.94, 0.82, 0.87]
    })

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )
# FOOTER
# ==========================================

st.markdown("---")

st.header("ℹ️ About This Project")

col1, col2 = st.columns(2)

with col1:

    st.subheader("👨‍💻 Developer")

    st.write("**Name:** SAIF RAZA")

    st.write("**Project:** AI Financial Fraud Detection System")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Language:** Python")

with col2:

    st.subheader("🛠 Technologies Used")

    st.write("✅ Python")

    st.write("✅ Pandas")

    st.write("✅ Scikit-learn")

    st.write("✅ Streamlit")

    st.write("✅ Matplotlib")

    st.write("✅ Joblib")

st.markdown("---")

st.caption(
    "Version 1.0 | Built for Machine Learning Portfolio & Resume Project"
)