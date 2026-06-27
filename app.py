import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_data
from plot_style import PALETTE, apply_layout

st.set_page_config(
    page_title="CX Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1A3C5E; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }
.metric-card {
    background: white; border-radius: 10px; padding: 18px 20px;
    border-left: 4px solid #E05C2A; box-shadow: 0 1px 6px rgba(0,0,0,.08);
}
.metric-val { font-size: 28px; font-weight: 700; color: #1A3C5E; }
.metric-lbl { font-size: 12px; color: #7A7066; margin-top: 2px; }
.metric-delta-bad  { font-size: 12px; color: #c0392b; font-weight: 600; }
.metric-delta-good { font-size: 12px; color: #2D7D46; font-weight: 600; }
.section-header {
    font-size: 20px; font-weight: 700; color: #1A3C5E;
    border-bottom: 2px solid #E05C2A; padding-bottom: 6px; margin-bottom: 16px;
}
.insight-box {
    background: #EBF3FB; border-left: 4px solid #1A3C5E;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 13px;
}
.warning-box {
    background: #FEF3E2; border-left: 4px solid #E05C2A;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CX Churn Intelligence")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("app.py",                                label="🏠 Overview",             icon="🏠")
    st.page_link("pages/1_Descriptive_Analysis.py",       page_title="1 · Descriptive Analysis",  icon="📋")
    st.page_link("pages/2_Bias_Detection.py",             page_title="2 · Bias Detection",        icon="🔍")
    st.page_link("pages/3_ML_Classification.py",          page_title="3 · ML Classification",     icon="🤖")
    st.page_link("pages/4_Model_Evaluation.py",           page_title="4 · Model Evaluation",      icon="📈")
    st.page_link("pages/5_Findings.py",                   page_title="5 · Findings & Recommendations", icon="💡")
    st.markdown("---")
    st.markdown("""
    <small>
    <b>Dataset:</b> 500 post-AI interaction survey responses<br>
    <b>Target:</b> Churn 30–60 days post-resolution<br>
    <b>Focus:</b> Silent Churner detection & bias audit
    </small>
    """, unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1A3C5E,#2a5c8e);
            padding:28px 32px;border-radius:12px;margin-bottom:24px;color:white'>
  <div style='font-size:11px;letter-spacing:2px;opacity:.7;text-transform:uppercase'>
    Customer Experience Analytics · Silent Churn Intelligence Platform
  </div>
  <div style='font-size:26px;font-weight:700;margin:8px 0 6px'>
    AI Resolution Bias & Churn Prediction Dashboard
  </div>
  <div style='font-size:13px;opacity:.8;max-width:620px'>
    Investigating why customers marked as "Successfully Resolved" churn 30–60 days later.
    This platform covers descriptive analysis, bias detection, ML classification,
    model evaluation, and actionable recommendations.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────────────────
total        = len(df)
churned      = df["DERIVED_ChurnBinary"].sum()
churn_rate   = churned / total * 100
silent       = df["SilentChurner"].sum()
silent_pct   = silent / churned * 100
false_res    = df[(df["DERIVED_ResolutionBinary"] == 1) & (df["DERIVED_ChurnBinary"] == 1)]
false_res_rt = len(false_res) / df["DERIVED_ResolutionBinary"].sum() * 100
avg_risk     = df["RiskScore"].mean()

cols = st.columns(5)
kpis = [
    ("500", "Total Responses", None, None),
    (f"{churned}", "Churned Customers", f"{churn_rate:.1f}% churn rate", "bad"),
    (f"{silent}", "Silent Churners", f"{silent_pct:.1f}% of churned", "bad"),
    (f"{false_res_rt:.1f}%", "False Resolution Rate", "Resolved → then churned", "bad"),
    (f"{avg_risk:.2f}", "Avg Risk Score", "Composite (0–10 scale)", None),
]
for col, (val, lbl, delta, dtype) in zip(cols, kpis):
    delta_cls = f"metric-delta-{dtype}" if dtype else "metric-lbl"
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-val'>{val}</div>
      <div class='metric-lbl'>{lbl}</div>
      {'<div class="' + delta_cls + '">' + delta + '</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Overview charts ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='section-header'>Churn Distribution</div>", unsafe_allow_html=True)
    pie = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[total - churned, churned],
        hole=.55,
        marker_colors=[PALETTE["green"], PALETTE["accent"]],
        textinfo="label+percent",
    ))
    apply_layout(pie, height=280)
    st.plotly_chart(pie, use_container_width=True)

with c2:
    st.markdown("<div class='section-header'>Churn by Resolution Status</div>", unsafe_allow_html=True)
    ct = df.groupby(["DERIVED_ResolutionBinary", "DERIVED_ChurnBinary"]).size().reset_index(name="n")
    ct["Resolution"] = ct["DERIVED_ResolutionBinary"].map({0: "Unresolved", 1: "Resolved"})
    ct["Churn"]      = ct["DERIVED_ChurnBinary"].map({0: "Retained", 1: "Churned"})
    bar = px.bar(ct, x="Resolution", y="n", color="Churn",
                 color_discrete_map={"Retained": PALETTE["green"], "Churned": PALETTE["accent"]},
                 barmode="group")
    apply_layout(bar, height=280)
    st.plotly_chart(bar, use_container_width=True)

with c3:
    st.markdown("<div class='section-header'>NPS Segment vs Churn</div>", unsafe_allow_html=True)
    nps_churn = df.groupby(["DERIVED_NPSSegment", "DERIVED_ChurnBinary"]).size().unstack(fill_value=0)
    nps_churn["ChurnRate"] = nps_churn[1] / (nps_churn[0] + nps_churn[1]) * 100
    nps_churn = nps_churn.reset_index()
    bar2 = px.bar(nps_churn, x="DERIVED_NPSSegment", y="ChurnRate",
                  color="DERIVED_NPSSegment",
                  color_discrete_map={
                      "Detractor": PALETTE["accent"],
                      "Passive": PALETTE["orange"],
                      "Promoter": PALETTE["green"],
                  })
    bar2.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
    apply_layout(bar2, height=280)
    st.plotly_chart(bar2, use_container_width=True)

# ── Silent Churner spotlight ─────────────────────────────────────────────────
st.markdown("<div class='section-header'>🚨 Silent Churner Spotlight</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='warning-box'>
  <b>⚠️ {silent} customers ({silent_pct:.1f}% of all churners) are Silent Churners</b> — their AI ticket
  was marked <i>Resolved</i> yet they left within 30–60 days. These customers are invisible to
  standard resolution-rate reporting and represent the core problem this platform investigates.
</div>
""", unsafe_allow_html=True)

sc1, sc2 = st.columns(2)
with sc1:
    sc_by_team = df.groupby("SupportTeam")["SilentChurner"].mean().reset_index()
    sc_by_team.columns = ["Team", "Silent Churn Rate"]
    sc_by_team["Silent Churn Rate"] *= 100
    fig_t = px.bar(sc_by_team, x="Team", y="Silent Churn Rate",
                   color="Silent Churn Rate", color_continuous_scale=["#2D7D46", "#E05C2A"],
                   title="Silent Churn Rate by Support Team")
    apply_layout(fig_t, height=280)
    st.plotly_chart(fig_t, use_container_width=True)

with sc2:
    sc_by_age = df.groupby("AgeGroup")["SilentChurner"].mean().reset_index()
    sc_by_age.columns = ["Age Group", "Silent Churn Rate"]
    sc_by_age["Silent Churn Rate"] *= 100
    age_order = ["18–24","25–34","35–44","45–54","55–64","65+"]
    sc_by_age["Age Group"] = pd.Categorical(sc_by_age["Age Group"], categories=age_order, ordered=True)
    sc_by_age = sc_by_age.sort_values("Age Group")
    fig_a = px.bar(sc_by_age, x="Age Group", y="Silent Churn Rate",
                   color="Silent Churn Rate", color_continuous_scale=["#2D7D46", "#E05C2A"],
                   title="Silent Churn Rate by Age Group")
    apply_layout(fig_a, height=280)
    st.plotly_chart(fig_a, use_container_width=True)

st.markdown("""
<div class='insight-box'>
  📌 Use the sidebar to navigate through the full analysis pipeline:
  Descriptive → Bias Detection → ML Models → Evaluation → Recommendations.
</div>
""", unsafe_allow_html=True)
