import streamlit as st
import joblib
import numpy as np

# -------------------------------
# 🚀 LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_compressed.pkl")

model = load_model()

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("💳 Credit Risk Prediction Dashboard")

# -------------------------------
# 📌 SIDEBAR GUIDE
# -------------------------------
st.sidebar.header("📌 Input Guide")
st.sidebar.write("""
Lower late payments + higher income = lower risk
""")

# -------------------------------
# 📊 SUMMARY
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Rows", "150,000")
col2.metric("Features", "10")
col3.metric("Target", "Default Risk")

st.write("---")

# -------------------------------
# 🧾 FORM (NO LAG)
# -------------------------------
with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        revolving = st.slider("Revolving Utilization", 0.0, 2.0, 0.5)
        age = st.slider("Age", 18, 100, 30)
        past_due_30 = st.slider("30-59 Days Late", 0, 10, 0)
        debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.5)

    with col2:
        income = st.number_input("Monthly Income", 0, 100000, 5000)
        open_credit = st.slider("Open Credit Lines", 0, 20, 5)
        late_90 = st.slider("90 Days Late", 0, 10, 0)
        dependents = st.slider("Dependents", 0, 10, 0)

    submit = st.form_submit_button("Predict Risk")

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
if submit:

    with st.spinner("Analyzing..."):

        input_data = np.array([[
            revolving,
            age,
            past_due_30,
            debt_ratio,
            income,
            open_credit,
            late_90,
            0,
            0,
            dependents
        ]])

        prediction = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

    st.subheader("📊 Result")

    if prediction == 1:
        st.error("🔴 High Risk")
    else:
        st.success("🟢 Low Risk")

    if prob:
        st.progress(int(prob * 100))
        st.write(f"Probability: {round(prob*100,2)}%")

    st.write("---")

    st.subheader("📋 Summary")
    st.write(f"""
    Age: {age}  
    Income: ₹{income}  
    Late Payments: {past_due_30 + late_90}  
    Debt Ratio: {debt_ratio}
    """)
