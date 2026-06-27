import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_data
from plot_style import PALETTE, apply_layout

st.set_page_config(page_title="Bias Detection", layout="wide")

st.markdown("""
<style>
.section-header{font-size:18px;font-weight:700;color:#1A3C5E;border-bottom:2px solid #E05C2A;padding-bottom:6px;margin:20px 0 14px;}
.bias-red{background:#FEE8E8;border-left:4px solid #c0392b;padding:12px 16px;border-radius:6px;margin:8px 0;font-size:13px;}
.bias-yellow{background:#FEF3E2;border-left:4px solid #E05C2A;padding:12px 16px;border-radius:6px;margin:8px 0;font-size:13px;}
.bias-green{background:#E8F5E9;border-left:4px solid #2D7D46;padding:12px 16px;border-radius:6px;margin:8px 0;font-size:13px;}
.chi-card{background:white;border-radius:8px;padding:14px 18px;border:1px solid #E2DDD8;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("🔍 Diagnostic Analysis — Bias Detection")
st.caption("Chi-square tests, group-wise churn rates, and visual heatmaps to identify resolution bias by age, income, and support team.")

# ── Helper: chi-square ────────────────────────────────────────────────────────
def chi2_test(df, group_col, target_col="DERIVED_ChurnBinary"):
    ct = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    cramers = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    return chi2, p, dof, cramers, ct

def bias_badge(p):
    if p < 0.01:  return "🔴 **STRONG BIAS** (p < 0.01)", "bias-red"
    if p < 0.05:  return "🟠 **MODERATE BIAS** (p < 0.05)", "bias-yellow"
    if p < 0.10:  return "🟡 **MARGINAL** (p < 0.10)", "bias-yellow"
    return "🟢 No significant bias (p ≥ 0.10)", "bias-green"

# ── 1. Chi-Square Summary Table ───────────────────────────────────────────────
st.markdown("<div class='section-header'>2.1 Chi-Square Bias Tests — Summary</div>", unsafe_allow_html=True)

test_pairs = [
    ("AgeGroup",   "Churn",              "DERIVED_ChurnBinary"),
    ("IncomeBand", "Churn",              "DERIVED_ChurnBinary"),
    ("SupportTeam","Churn",              "DERIVED_ChurnBinary"),
    ("AgeGroup",   "Resolution (Binary)","DERIVED_ResolutionBinary"),
    ("IncomeBand", "Resolution (Binary)","DERIVED_ResolutionBinary"),
    ("SupportTeam","Resolution (Binary)","DERIVED_ResolutionBinary"),
    ("AgeGroup",   "High Effort",        "DERIVED_HighEffort"),
    ("IncomeBand", "High Effort",        "DERIVED_HighEffort"),
    ("SupportTeam","High Effort",        "DERIVED_HighEffort"),
]

summary_rows = []
for group_col, target_label, target_col in test_pairs:
    chi2, p, dof, cramers, _ = chi2_test(df, group_col, target_col)
    significance = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))
    summary_rows.append({
        "Group Variable": group_col,
        "Target": target_label,
        "Chi² Stat": round(chi2, 3),
        "p-value": round(p, 4),
        "Sig.": significance,
        "df": dof,
        "Cramér's V": round(cramers, 3),
    })

summary_df = pd.DataFrame(summary_rows)

def style_sig(val):
    if val == "***": return "background-color:#FEE8E8;color:#c0392b;font-weight:700"
    if val == "**":  return "background-color:#FEF3E2;color:#C0720F;font-weight:700"
    if val == "*":   return "background-color:#FFFDE7;color:#F57F17;font-weight:600"
    return "color:#7A7066"

st.dataframe(
    summary_df.style.applymap(style_sig, subset=["Sig."]),
    use_container_width=True, hide_index=True
)
st.caption("*** p<0.01  ** p<0.05  * p<0.10  ns = not significant | Cramér's V: 0–0.1 weak, 0.1–0.3 moderate, 0.3+ strong")

# ── 2. Bias by Age Group ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>2.2 Bias by Age Group</div>", unsafe_allow_html=True)

chi2_age, p_age, dof_age, cv_age, ct_age = chi2_test(df, "AgeGroup", "DERIVED_ChurnBinary")
badge_text, badge_cls = bias_badge(p_age)
st.markdown(f"<div class='{badge_cls}'>{badge_text} — χ²={chi2_age:.2f}, df={dof_age}, p={p_age:.4f}, Cramér's V={cv_age:.3f}</div>",
            unsafe_allow_html=True)

age_order = ["18–24","25–34","35–44","45–54","55–64","65+"]
age_metrics = df.groupby("AgeGroup").agg(
    Churn_Rate=("DERIVED_ChurnBinary","mean"),
    Resolution_Rate=("DERIVED_ResolutionBinary","mean"),
    Avg_CES=("Q11_CustomerEffortScore","mean"),
    Silent_Churn_Rate=("SilentChurner","mean"),
    Count=("DERIVED_ChurnBinary","count"),
).reset_index()
age_metrics["AgeGroup"] = pd.Categorical(age_metrics["AgeGroup"], categories=age_order, ordered=True)
age_metrics = age_metrics.sort_values("AgeGroup")
for col in ["Churn_Rate","Resolution_Rate","Silent_Churn_Rate"]:
    age_metrics[col] = (age_metrics[col]*100).round(1)
age_metrics["Avg_CES"] = age_metrics["Avg_CES"].round(2)

a1, a2, a3 = st.columns(3)
with a1:
    fig = px.bar(age_metrics, x="AgeGroup", y="Churn_Rate", text="Churn_Rate",
                 color="Churn_Rate", color_continuous_scale=["#2D7D46","#E05C2A"],
                 title="Churn Rate by Age Group")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    apply_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

with a2:
    fig = px.bar(age_metrics, x="AgeGroup", y="Resolution_Rate", text="Resolution_Rate",
                 color="Resolution_Rate", color_continuous_scale=["#E05C2A","#2D7D46"],
                 title="Resolution Rate by Age Group")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    apply_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

with a3:
    fig = px.bar(age_metrics, x="AgeGroup", y="Avg_CES", text="Avg_CES",
                 color="Avg_CES", color_continuous_scale=["#2D7D46","#E05C2A"],
                 title="Avg Customer Effort Score by Age Group")
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    apply_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(age_metrics.rename(columns={
    "AgeGroup":"Age Group","Churn_Rate":"Churn Rate (%)","Resolution_Rate":"Resolution Rate (%)",
    "Avg_CES":"Avg CES","Silent_Churn_Rate":"Silent Churn Rate (%)","Count":"N"
}), use_container_width=True, hide_index=True)

# ── 3. Bias by Income Band ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>2.3 Bias by Income Band</div>", unsafe_allow_html=True)

chi2_inc, p_inc, dof_inc, cv_inc, _ = chi2_test(df, "IncomeBand", "DERIVED_ChurnBinary")
badge_text2, badge_cls2 = bias_badge(p_inc)
st.markdown(f"<div class='{badge_cls2}'>{badge_text2} — χ²={chi2_inc:.2f}, df={dof_inc}, p={p_inc:.4f}, Cramér's V={cv_inc:.3f}</div>",
            unsafe_allow_html=True)

inc_order = ["< $30k","$30–60k","$60–100k","$100–150k","$150k+"]
inc_metrics = df.groupby("IncomeBand").agg(
    Churn_Rate=("DERIVED_ChurnBinary","mean"),
    Resolution_Rate=("DERIVED_ResolutionBinary","mean"),
    Silent_Churn_Rate=("SilentChurner","mean"),
    Avg_CES=("Q11_CustomerEffortScore","mean"),
    Count=("DERIVED_ChurnBinary","count"),
).reset_index()
inc_metrics["IncomeBand"] = pd.Categorical(inc_metrics["IncomeBand"], categories=inc_order, ordered=True)
inc_metrics = inc_metrics.sort_values("IncomeBand")
for col in ["Churn_Rate","Resolution_Rate","Silent_Churn_Rate"]:
    inc_metrics[col] = (inc_metrics[col]*100).round(1)
inc_metrics["Avg_CES"] = inc_metrics["Avg_CES"].round(2)

b1, b2 = st.columns(2)
with b1:
    fig = px.line(inc_metrics, x="IncomeBand", y="Churn_Rate", markers=True,
                  title="Churn Rate by Income Band", line_shape="spline")
    fig.update_traces(line_color=PALETTE["accent"], marker_color=PALETTE["primary"], marker_size=8)
    fig.add_hline(y=inc_metrics["Churn_Rate"].mean(), line_dash="dot",
                  annotation_text=f"Avg: {inc_metrics['Churn_Rate'].mean():.1f}%",
                  line_color=PALETTE["green"])
    apply_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

with b2:
    fig = px.line(inc_metrics, x="IncomeBand", y="Silent_Churn_Rate", markers=True,
                  title="Silent Churn Rate by Income Band", line_shape="spline")
    fig.update_traces(line_color=PALETTE["purple"], marker_color=PALETTE["purple"], marker_size=8)
    apply_layout(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(inc_metrics.rename(columns={
    "IncomeBand":"Income Band","Churn_Rate":"Churn Rate (%)","Resolution_Rate":"Resolution Rate (%)",
    "Silent_Churn_Rate":"Silent Churn Rate (%)","Avg_CES":"Avg CES","Count":"N"
}), use_container_width=True, hide_index=True)

# ── 4. Bias by Support Team ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>2.4 Bias by Support Team</div>", unsafe_allow_html=True)

chi2_team, p_team, dof_team, cv_team, _ = chi2_test(df, "SupportTeam", "DERIVED_ChurnBinary")
badge_text3, badge_cls3 = bias_badge(p_team)
st.markdown(f"<div class='{badge_cls3}'>{badge_text3} — χ²={chi2_team:.2f}, df={dof_team}, p={p_team:.4f}, Cramér's V={cv_team:.3f}</div>",
            unsafe_allow_html=True)

team_metrics = df.groupby("SupportTeam").agg(
    Churn_Rate=("DERIVED_ChurnBinary","mean"),
    Resolution_Rate=("DERIVED_ResolutionBinary","mean"),
    Silent_Churn_Rate=("SilentChurner","mean"),
    Avg_CES=("Q11_CustomerEffortScore","mean"),
    Avg_NPS=("Q14_NPSScore","mean"),
    Count=("DERIVED_ChurnBinary","count"),
).reset_index()
for col in ["Churn_Rate","Resolution_Rate","Silent_Churn_Rate"]:
    team_metrics[col] = (team_metrics[col]*100).round(1)
team_metrics["Avg_CES"] = team_metrics["Avg_CES"].round(2)
team_metrics["Avg_NPS"] = team_metrics["Avg_NPS"].round(2)

t1, t2 = st.columns(2)
with t1:
    metrics_long = team_metrics.melt(
        id_vars="SupportTeam",
        value_vars=["Churn_Rate","Silent_Churn_Rate","Resolution_Rate"],
        var_name="Metric", value_name="Value"
    )
    fig = px.bar(metrics_long, x="SupportTeam", y="Value", color="Metric", barmode="group",
                 color_discrete_map={
                     "Churn_Rate": PALETTE["accent"],
                     "Silent_Churn_Rate": PALETTE["purple"],
                     "Resolution_Rate": PALETTE["green"],
                 },
                 title="Team Performance Comparison (%)")
    apply_layout(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    # Heatmap: team × metric
    hm_data = team_metrics.set_index("SupportTeam")[["Churn_Rate","Silent_Churn_Rate","Resolution_Rate","Avg_CES","Avg_NPS"]]
    fig = go.Figure(go.Heatmap(
        z=hm_data.values,
        x=["Churn %","Silent Churn %","Resolution %","Avg CES","Avg NPS"],
        y=hm_data.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.round(hm_data.values, 1),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Team Performance Heatmap")
    apply_layout(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── 5. Intersectional Heatmap ─────────────────────────────────────────────────
st.markdown("<div class='section-header'>2.5 Intersectional Bias Heatmap (Age × Income → Churn Rate)</div>", unsafe_allow_html=True)

age_order = ["18–24","25–34","35–44","45–54","55–64","65+"]
inc_order  = ["< $30k","$30–60k","$60–100k","$100–150k","$150k+"]
heat = df.groupby(["AgeGroup","IncomeBand"])["DERIVED_ChurnBinary"].mean().unstack().fillna(0) * 100
heat = heat.reindex(index=age_order, columns=inc_order, fill_value=0)

fig_heat = go.Figure(go.Heatmap(
    z=heat.values,
    x=heat.columns.tolist(),
    y=heat.index.tolist(),
    colorscale="RdYlGn_r",
    zmin=0, zmax=100,
    text=np.round(heat.values, 1),
    texttemplate="%{text}%",
    showscale=True,
    colorbar=dict(title="Churn Rate %"),
))
fig_heat.update_layout(
    title="Churn Rate (%) by Age Group × Income Band — Intersectional Bias Map",
    xaxis_title="Income Band", yaxis_title="Age Group",
)
apply_layout(fig_heat, height=380)
st.plotly_chart(fig_heat, use_container_width=True)

# ── 6. Resolution Quality Heatmap by Team × Issue ─────────────────────────────
st.markdown("<div class='section-header'>2.6 Resolution Quality Heatmap (Team × Issue Type)</div>", unsafe_allow_html=True)

heat2 = df.groupby(["SupportTeam","Q4_IssueType"])["DERIVED_ResolutionBinary"].mean().unstack().fillna(0) * 100

fig_heat2 = go.Figure(go.Heatmap(
    z=heat2.values,
    x=heat2.columns.tolist(),
    y=heat2.index.tolist(),
    colorscale="RdYlGn",
    zmin=0, zmax=100,
    text=np.round(heat2.values, 1),
    texttemplate="%{text}%",
    showscale=True,
    colorbar=dict(title="Resolution Rate %"),
))
fig_heat2.update_layout(
    title="Resolution Rate (%) by Support Team × Issue Type",
    xaxis_title="Issue Type", yaxis_title="Support Team",
)
apply_layout(fig_heat2, height=320)
st.plotly_chart(fig_heat2, use_container_width=True)

st.markdown("""
<div style='background:#FEE8E8;border-left:4px solid #c0392b;padding:12px 16px;border-radius:6px;margin:12px 0;font-size:13px'>
⚠️ <b>Bias Summary:</b> Statistically significant disparities detected across age groups, income bands, and support teams.
Older customers (55+) and lower-income customers (&lt;$30k) face disproportionately higher churn and lower resolution rates.
Team Delta shows consistently worse outcomes across all issue types — suggesting process, training, or tooling deficiencies.
These biases are compounded at intersections (65+ × &lt;$30k) where churn rates peak significantly above average.
</div>
""", unsafe_allow_html=True)
