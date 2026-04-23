import streamlit as st
import pickle
import numpy as np

# 📌 Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# 📌 Title
st.title("💳 Credit Risk Prediction Dashboard")

st.sidebar.header("📌 Input Guide")

st.sidebar.markdown("""
### 🧾 How to Fill Inputs

**1. Revolving Utilization**
- Range: 0.0 – 2.0  
- Ideal: Below 0.5  
- Meaning: Credit usage ratio (lower is better)

**2. Age**
- Range: 18 – 100  
- Ideal: 25+  

**3. 30-59 Days Past Due**
- Range: 0 – 10  
- Ideal: 0  
- Meaning: Late payments (recent)

**4. Debt Ratio**
- Range: 0.0 – 5.0  
- Ideal: Below 0.5  
- Meaning: Debt vs income

**5. Monthly Income**
- Range: ₹0 – ₹100000  
- Ideal: Higher income = lower risk  

**6. Open Credit Lines**
- Range: 0 – 20  
- Ideal: 3 – 10  

**7. 90 Days Late**
- Range: 0 – 10  
- Ideal: 0  
- Meaning: Serious late payments  

**8. Dependents**
- Range: 0 – 10  
- Meaning: Number of dependents  
""")

# 📌 Summary Section
st.subheader("📌 Dataset Summary")
col1, col2, col3 = st.columns(3)

col1.metric("Rows", "150,000")
col2.metric("Features", "10")
col3.metric("Target", "Default Risk")

st.write("---")

# 📌 User Input Section
st.subheader("🧾 Enter Borrower Details")

col1, col2 = st.columns(2)

with col1:
    revolving = st.slider("Revolving Utilization", 0.0, 2.0, 0.5)
    age = st.slider("Age", 18, 100, 30)
    past_due_30 = st.slider("30-59 Days Past Due", 0, 10, 0)
    debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.5)

with col2:
    income = st.number_input("Monthly Income", 0, 100000, 5000)
    open_credit = st.slider("Open Credit Lines", 0, 20, 5)
    late_90 = st.slider("90 Days Late", 0, 10, 0)
    dependents = st.slider("Dependents", 0, 10, 0)

# 📌 Prediction Button
if st.button("🔍 Predict Risk"):

    input_data = np.array([[
        revolving,
        age,
        past_due_30,
        debt_ratio,
        income,
        open_credit,
        late_90,
        0,  # placeholder for real estate loans (optional)
        0,  # placeholder for 60-89 days late
        dependents
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk (Safe Borrower)")

    # 📊 Probability display
    st.write(f"**Default Probability:** {prob:.2f}")

    # 📊 Progress bar (visual)
    st.progress(int(prob * 100))

st.write("---")

# 📌 Footer
st.caption("Built with Streamlit | Credit Risk ML Project")