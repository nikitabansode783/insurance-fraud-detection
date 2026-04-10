import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("fraud_model.pkl")
columns = joblib.load("columns.pkl")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🚀 Navigation")
option = st.sidebar.selectbox(
    "Select Option",
    ["Fraud Prediction", "Dataset View", "Graphs & Analysis"]
)

# ---------------- TITLE ---------------- #
st.title("🚗 Insurance Fraud Detection System")

# =========================================================
# 🔴 1. FRAUD PREDICTION
# =========================================================
if option == "Fraud Prediction":

    st.header("🔍 Enter Customer Details")

    age = st.number_input("Age", 18, 100, 30)
    deductible = st.number_input("Deductible", 300, 1000, 400)
    driver_rating = st.slider("Driver Rating", 1, 5, 3)

    st.markdown("---")

    if st.button("Predict Fraud"):

        # Create input dataframe
        input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

        if 'Age' in input_data.columns:
            input_data.at[0, 'Age'] = age

        if 'Deductible' in input_data.columns:
            input_data.at[0, 'Deductible'] = deductible

        if 'DriverRating' in input_data.columns:
            input_data.at[0, 'DriverRating'] = driver_rating

        # Prediction
        prediction = model.predict(input_data)[0]

        st.subheader("Result:")

        if prediction == 1:
            st.error("⚠️ Fraudulent Claim Detected")
        else:
            st.success("✅ Genuine Claim")

    st.markdown("---")

    # Risk Segmentation
    st.subheader("📊 Risk Level")

    if driver_rating <= 2 and deductible < 400:
        st.error("🔴 High Risk Customer")

    elif driver_rating == 3:
        st.warning("🟡 Medium Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")

# =========================================================
# 📂 2. DATASET VIEW
# =========================================================
elif option == "Dataset View":

    st.header("📂 Insurance Dataset")

    df = pd.read_csv("data/carclaims.csv")

    st.write("### Preview of Dataset")
    st.dataframe(df.head(20))

    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### Column Names")
    st.write(df.columns)

# =========================================================
# 📊 3. GRAPHS & ANALYSIS
# =========================================================
elif option == "Graphs & Analysis":

    st.header("📊 Data Visualization")

    # Fraud distribution
    st.subheader("Fraud vs Non-Fraud")

    fraud_counts = [14497, 923]
    labels = ['Not Fraud', 'Fraud']

    fig1, ax1 = plt.subplots()
    ax1.pie(fraud_counts, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig1)

    # Driver rating graph
    st.subheader("Driver Rating Distribution")

    rating_counts = [2000, 3000, 4000, 3000, 1420]

    fig2, ax2 = plt.subplots()
    ax2.bar([1,2,3,4,5], rating_counts)
    ax2.set_xlabel("Driver Rating")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("AI/ML Project: Insurance Fraud Detection & Risk Segmentation")