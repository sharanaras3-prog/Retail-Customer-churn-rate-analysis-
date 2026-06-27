import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_data
from plot_style import PALETTE, apply_layout

st.set_page_config(page_title="Descriptive Analysis", layout="wide")

st.markdown("""
<style>
.section-header {
    font-size:18px;font-weight:700;color:#1A3C5E;
    border-bottom:2px solid #E05C2A;padding-bottom:6px;margin:20px 0 14px;
}
.insight-box {
    background:#EBF3FB;border-left:4px solid #1A3C5E;
    padding:12px 16px;border-radius:6px;margin:8px 0;font-size:13px;
}
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("📋 Descriptive Analysis")
st.caption("Cross-tabulations, distributions, value counts, and summary statistics across all survey variables.")

# ── 1. Summary Statistics ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>1.1 Summary Statistics — Numeric Variables</div>", unsafe_allow_html=True)
num_cols = [
    "Q1_Tenure","Q2_PurchaseFrequency","Q3_SpendTier",
    "Q7_RepeatContactCount","Q8_AIUnderstanding","Q9_TrueResolution",
    "Q10_Personalization","Q11_CustomerEffortScore","Q12_PostAIBrandSentiment",
    "Q13_OverallSatisfaction","Q14_NPSScore","Q15_PurchaseIntent30d","RiskScore"
]
stats = df[num_cols].describe().T.round(2)
stats.index.name = "Variable"
st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["mean","std"]), use_container_width=True)

# ── 2. Value Counts ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>1.2 Categorical Variable Value Counts</div>", unsafe_allow_html=True)
cat_cols = {
    "Q4_IssueType": "Issue Type",
    "Q5_Channel": "Contact Channel",
    "Q18_ResolutionVsGaveUp": "Resolution Outcome",
    "AgeGroup": "Age Group",
    "IncomeBand": "Income Band",
    "SupportTeam": "Support Team",
    "DERIVED_NPSSegment": "NPS Segment",
}
tab_labels = list(cat_cols.values())
tabs = st.tabs(tab_labels)
for tab, (col, label) in zip(tabs, cat_cols.items()):
    with tab:
        vc = df[col].value_counts().reset_index()
        vc.columns = [label, "Count"]
        vc["Percentage"] = (vc["Count"] / len(df) * 100).round(1)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(vc, use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(vc, x=label, y="Count", text="Percentage",
                         color="Count", color_continuous_scale=["#D6E4F0","#1A3C5E"])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            apply_layout(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)

# ── 3. Churn Distribution by Key Variables ────────────────────────────────────
st.markdown("<div class='section-header'>1.3 Churn Rate by Key Variables</div>", unsafe_allow_html=True)

def churn_rate_bar(group_col, label, order=None):
    cr = df.groupby(group_col)["DERIVED_ChurnBinary"].mean().reset_index()
    cr.columns = [label, "Churn Rate"]
    cr["Churn Rate"] = (cr["Churn Rate"] * 100).round(1)
    if order:
        cr[label] = pd.Categorical(cr[label], categories=order, ordered=True)
        cr = cr.sort_values(label)
    fig = px.bar(cr, x=label, y="Churn Rate", text="Churn Rate",
                 color="Churn Rate", color_continuous_scale=["#2D7D46","#E05C2A"])
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn Rate (%)")
    apply_layout(fig, height=300)
    return fig

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.markdown("**By Tenure**")
    age_order = ["< 3 months","3–12 months","1–3 years","3+ years"]
    df_t = df.copy()
    df_t["Tenure_Label2"] = pd.Categorical(df_t["Tenure_Label"], categories=age_order, ordered=True)
    cr_t = df_t.groupby("Tenure_Label2")["DERIVED_ChurnBinary"].mean().reset_index()
    cr_t.columns = ["Tenure","Churn Rate"]
    cr_t["Churn Rate"] = (cr_t["Churn Rate"]*100).round(1)
    fig_t = px.bar(cr_t, x="Tenure", y="Churn Rate", text="Churn Rate",
                   color="Churn Rate", color_continuous_scale=["#2D7D46","#E05C2A"])
    fig_t.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_t.update_layout(coloraxis_showscale=False)
    apply_layout(fig_t, height=300)
    st.plotly_chart(fig_t, use_container_width=True)

with r1c2:
    st.markdown("**By Spend Tier**")
    spend_order = ["< $50","$50–$150","$150–$500","$500+"]
    df_s = df.copy()
    df_s["Spend_Label2"] = pd.Categorical(df_s["Spend_Label"], categories=spend_order, ordered=True)
    cr_s = df_s.groupby("Spend_Label2")["DERIVED_ChurnBinary"].mean().reset_index()
    cr_s.columns = ["Spend","Churn Rate"]
    cr_s["Churn Rate"] = (cr_s["Churn Rate"]*100).round(1)
    fig_s = px.bar(cr_s, x="Spend", y="Churn Rate", text="Churn Rate",
                   color="Churn Rate", color_continuous_scale=["#2D7D46","#E05C2A"])
    fig_s.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_s.update_layout(coloraxis_showscale=False)
    apply_layout(fig_s, height=300)
    st.plotly_chart(fig_s, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.markdown("**By Issue Type**")
    st.plotly_chart(churn_rate_bar("Q4_IssueType", "Issue Type"), use_container_width=True)
with r2c2:
    st.markdown("**By Support Team**")
    st.plotly_chart(churn_rate_bar("SupportTeam", "Support Team"), use_container_width=True)

# ── 4. Cross-tabulations ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>1.4 Cross-Tabulations</div>", unsafe_allow_html=True)

def render_crosstab(row_var, col_var, row_label, col_label):
    ct = pd.crosstab(df[row_var], df[col_var], margins=True)
    ct_pct = pd.crosstab(df[row_var], df[col_var], normalize="index").round(3) * 100
    with st.expander(f"Cross-tab: {row_label} × {col_label}", expanded=True):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.caption("Count")
            st.dataframe(ct, use_container_width=True)
        with tc2:
            st.caption("Row % (excluding margins)")
            st.dataframe(ct_pct.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

render_crosstab("AgeGroup","DERIVED_ChurnBinary","Age Group","Churn")
render_crosstab("IncomeBand","DERIVED_ChurnBinary","Income Band","Churn")
render_crosstab("SupportTeam","DERIVED_ChurnBinary","Support Team","Churn")
render_crosstab("Q18_ResolutionVsGaveUp","DERIVED_ChurnBinary","Resolution Outcome","Churn")

# ── 5. Score Distributions ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>1.5 Satisfaction Score Distributions by Churn Status</div>", unsafe_allow_html=True)
score_cols = ["Q8_AIUnderstanding","Q9_TrueResolution","Q11_CustomerEffortScore",
              "Q13_OverallSatisfaction","Q14_NPSScore","RiskScore"]
score_labels = ["AI Understanding","True Resolution","Customer Effort","Overall Satisfaction","NPS Score","Risk Score"]

for i in range(0, len(score_cols), 3):
    cols_row = st.columns(3)
    for j, col in enumerate(cols_row):
        idx = i + j
        if idx >= len(score_cols):
            break
        sc = score_cols[idx]
        lb = score_labels[idx]
        with col:
            fig = px.histogram(df, x=sc, color="DERIVED_ChurnBinary",
                               barmode="overlay",
                               color_discrete_map={0: PALETTE["green"], 1: PALETTE["accent"]},
                               labels={"DERIVED_ChurnBinary": "Churn", sc: lb},
                               nbins=10, opacity=0.75)
            fig.for_each_trace(lambda t: t.update(
                name="Retained" if t.name == "0" else "Churned"))
            apply_layout(fig, title=lb, height=260)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='insight-box'>
📌 <b>Key takeaways:</b> Churned customers consistently score lower on True Resolution (Q9) and
Overall Satisfaction (Q13), and higher on Customer Effort Score (Q11) — confirming that
<i>perceived</i> resolution quality diverges significantly from the AI's "Resolved" label.
</div>
""", unsafe_allow_html=True)

# ── 6. Raw data explorer ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>1.6 Raw Data Explorer</div>", unsafe_allow_html=True)
churn_filter = st.selectbox("Filter by Churn Status", ["All", "Churned", "Retained"])
team_filter  = st.multiselect("Filter by Support Team", df["SupportTeam"].unique().tolist(), default=df["SupportTeam"].unique().tolist())
df_view = df.copy()
if churn_filter == "Churned":   df_view = df_view[df_view["DERIVED_ChurnBinary"] == 1]
elif churn_filter == "Retained": df_view = df_view[df_view["DERIVED_ChurnBinary"] == 0]
df_view = df_view[df_view["SupportTeam"].isin(team_filter)]
display_cols = ["ResponseID","AgeGroup","IncomeBand","SupportTeam","Q4_IssueType",
                "Q9_TrueResolution","Q11_CustomerEffortScore","Q14_NPSScore",
                "Q18_ResolutionVsGaveUp","DERIVED_ChurnBinary","SilentChurner","RiskScore"]
st.dataframe(df_view[display_cols].reset_index(drop=True), use_container_width=True, height=350)
st.caption(f"Showing {len(df_view)} records")
