import streamlit as st, pandas as pd
from pathlib import Path
df = pd.read_csv(Path(__file__).parent / "scored_customers.csv")
st.title("Churn ROI Simulator")
st.title("Churn ROI Simulator")
discount = st.slider("Discount (%)", 0, 50, 10)
boost = st.slider("Churn reduction due to offer (%)", 0, 50, 15)
top_n = st.slider("Target top N risky customers", 100, 2000, 500)

t = df.sort_values("churn_prob", ascending=False).head(top_n)
cost = (t["MonthlyCharges"] * discount / 100 * 12).sum()
saved = (t["CLV"] * t["churn_prob"] * boost / 100).sum()
c1, c2, c3 = st.columns(3)
c1.metric("Offer cost", f"${cost:,.0f}")
c2.metric("Expected savings", f"${saved:,.0f}")
c3.metric("Net ROI", f"${saved - cost:,.0f}")
st.dataframe(t)
