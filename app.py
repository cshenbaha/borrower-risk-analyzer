import streamlit as st
import joblib
import numpy as np

# -------------------------------
# 🚀 LOAD MODEL (CACHED - FAST)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_compressed.pkl")

model = load_model()

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------------
# 🎯 TITLE
# -------------------------------
st.title("💳 Credit Risk Prediction Dashboard")

# -------------------------------
# 📌 SIDEBAR (INPUT GUIDE)
# -------------------------------
st.sidebar.header("📌 Input Guide")

st.sidebar.markdown("""
**Revolving Utilization**: 0–2 (Ideal < 0.5)  
**Age**: 18–100  
**30-59 Days Past Due**: 0–10 (Ideal 0)  
**Debt Ratio**: 0–5 (Ideal < 0.5)  
**Monthly Income**: Higher is better  
**Open Credit Lines**: 0–20  
**90 Days Late**: 0–10 (Ideal 0)  
**Dependents**: 0–10  
""")

# -------------------------------
# 📊 DATASET SUMMARY
# -------------------------------
st.subheader("📊 Dataset Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", "150,000")
col2.metric("Features", "10")
col3.metric("Target", "Default Risk")

st.write("---")

# -------------------------------
# 🧾 INPUT FORM (NO LAG)
# -------------------------------
st.subheader("🧾 Enter Borrower Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        revolving = st.slider("Revolving Utilization", 0.0, 2.0, 0.5)
        age = st.slider("Age", 18, 100, 30)
        past_due_30 = st.slider("30-59 Days Past Due", 0, 10, 0)
        debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.5)

    with col2:
        income = st.number_input("Monthly Income (₹)", 0, 100000, 5000)
        open_credit = st.slider("Open Credit Lines", 0, 20, 5)
        late_90 = st.slider("90 Days Late", 0, 10, 0)
        dependents = st.slider("Dependents", 0, 10, 0)

    submit = st.form_submit_button("🔍 Predict Risk")

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
if submit:

    with st.spinner("Analyzing borrower risk..."):

        input_data = np.array([[
            revolving,
            age,
            past_due_30,
            debt_ratio,
            income,
            open_credit,
            late_90,
            0,  # placeholder
            0,  # placeholder
            dependents
        ]])

        prediction = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

    # -------------------------------
    # 📊 RESULT
    # -------------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🔴 High Risk of Default")
    else:
        st.success("🟢 Low Risk (Safe Borrower)")

    if prob is not None:
        st.write(f"**Default Probability:** {round(prob * 100, 2)}%")
        st.progress(int(prob * 100))

    st.write("---")

    # -------------------------------
    # 📋 SUMMARY
    # -------------------------------
    st.subheader("📋 Customer Summary")

    st.write(f"""
    - Age: {age}
    - Monthly Income: ₹{income}
    - Revolving Utilization: {revolving}
    - Debt Ratio: {debt_ratio}
    - Late Payments (30-59 days): {past_due_30}
    - Late Payments (90 days): {late_90}
    - Open Credit Lines: {open_credit}
    - Dependents: {dependents}
    """)

    # -------------------------------
    # 💡 INSIGHTS
    # -------------------------------
    st.subheader("💡 Insights")

    if past_due_30 > 2 or late_90 > 0:
        st.warning("⚠️ Frequent late payments increase default risk")

    if debt_ratio > 1:
        st.warning("⚠️ High debt compared to income")

    if income < 10000:
        st.warning("⚠️ Low income may affect repayment ability")

    if prediction == 0:
        st.success("✅ Customer is likely safe for loan approval")
    else:
        st.error("🚨 High caution: borrower may default")

# -------------------------------
# 📌 FOOTER
# -------------------------------
st.write("---")
st.caption("Built with Streamlit | Credit Risk ML Project 🚀")
