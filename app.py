import json, joblib, pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Machine learning project · Python, Pandas, Scikit-learn, Plotly")

@st.cache_resource
def load():
    return joblib.load("model.joblib"), json.load(open("metrics.json"))
@st.cache_data
def data():
    d = pd.read_csv("data/telco_churn.csv"); d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce"); return d
model, metrics = load(); df = data()
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model comparison", "🔎 Data insights"])

with tab1:
    st.subheader("Will this customer leave?")
    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c1.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
    contract = c2.selectbox("Contract", df.Contract.unique())
    internet = c2.selectbox("Internet service", df.InternetService.unique())
    payment = c3.selectbox("Payment method", df.PaymentMethod.unique())
    tech = c3.selectbox("Tech support", ["No", "Yes", "No internet service"])
    row = df.drop(columns=["customerID", "Churn"]).mode().iloc[[0]].copy()   # sensible defaults
    for k, v in dict(tenure=tenure, MonthlyCharges=monthly, Contract=contract, InternetService=internet,
                     PaymentMethod=payment, TechSupport=tech, TotalCharges=tenure * monthly).items():
        row[k] = v
    p = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{p:.0%}")
    st.progress(float(p))
    (st.error if p >= .5 else st.success)("High risk of churn: consider a retention offer." if p >= .5 else "Low risk of churn.")

with tab2:
    st.subheader(f"Best model: {metrics['best']}")
    res = pd.DataFrame(metrics["results"]).drop(columns="confusion").set_index("model")
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#cdeeee"), width="stretch")
    imp = pd.DataFrame(metrics["importance"], columns=["feature", "importance"])
    st.plotly_chart(px.bar(imp[::-1], x="importance", y="feature", orientation="h",
                           title="Top features (permutation importance, ROC-AUC drop)"), width="stretch")

with tab3:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="Contract", color="Churn", barmode="group", title="Churn by contract type"), width="stretch")
    b.plotly_chart(px.box(df, x="Churn", y="tenure", title="Tenure vs churn"), width="stretch")

st.caption("Dataset: IBM Telco Customer Churn (public sample data).")
