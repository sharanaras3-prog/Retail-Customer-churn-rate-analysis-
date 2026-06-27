import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_data, get_ml_features
from plot_style import PALETTE, MODEL_COLORS, apply_layout

st.set_page_config(page_title="ML Classification", layout="wide")

st.markdown("""
<style>
.section-header{font-size:18px;font-weight:700;color:#1A3C5E;border-bottom:2px solid #E05C2A;padding-bottom:6px;margin:20px 0 14px;}
.model-card{background:white;border-radius:10px;padding:16px 20px;border:1px solid #E2DDD8;margin-bottom:12px;}
.insight-box{background:#EBF3FB;border-left:4px solid #1A3C5E;padding:12px 16px;border-radius:6px;margin:8px 0;font-size:13px;}
</style>
""", unsafe_allow_html=True)

df = load_data()
X, y, feature_cols = get_ml_features(df)

st.title("🤖 Feature Engineering & ML Classification")
st.caption("KNN, Decision Tree, Random Forest, and Gradient Boosting trained to predict churn with SMOTE class balancing.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    test_size   = st.slider("Test Set Size (%)", 15, 40, 25) / 100
    use_smote   = st.checkbox("Apply SMOTE (balance classes)", value=True)
    random_seed = st.number_input("Random Seed", value=42, step=1)
    st.markdown("---")
    st.markdown("**KNN**")
    knn_k = st.slider("n_neighbors", 3, 25, 7, step=2)
    st.markdown("**Decision Tree**")
    dt_depth = st.slider("max_depth", 2, 20, 6)
    st.markdown("**Random Forest**")
    rf_trees = st.slider("n_estimators", 50, 300, 150, step=50)
    rf_depth = st.slider("max_depth (RF)", 2, 20, 8)
    st.markdown("**Gradient Boosting**")
    gb_trees = st.slider("n_estimators (GB)", 50, 300, 150, step=50)
    gb_lr    = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], value=0.1)

# ── Feature Engineering Summary ───────────────────────────────────────────────
st.markdown("<div class='section-header'>3.1 Engineered Features</div>", unsafe_allow_html=True)

feat_info = pd.DataFrame([
    {"Feature": "Q1_Tenure",            "Type": "Ordinal",  "Source": "Survey Q1",     "Description": "Customer tenure tier (1–4)"},
    {"Feature": "Q2_PurchaseFrequency", "Type": "Ordinal",  "Source": "Survey Q2",     "Description": "Purchase frequency tier"},
    {"Feature": "Q3_SpendTier",         "Type": "Ordinal",  "Source": "Survey Q3",     "Description": "Monthly spend tier"},
    {"Feature": "Q7_RepeatContactCount","Type": "Numeric",  "Source": "Survey Q7",     "Description": "Times contacted for same issue"},
    {"Feature": "Q8_AIUnderstanding",   "Type": "Likert",   "Source": "Survey Q8",     "Description": "Perceived AI comprehension (1–5)"},
    {"Feature": "Q9_TrueResolution",    "Type": "Likert",   "Source": "Survey Q9",     "Description": "Customer-reported resolution quality"},
    {"Feature": "Q11_CustomerEffortScore","Type":"Likert",  "Source": "Survey Q11",    "Description": "Effort required to resolve issue (CES)"},
    {"Feature": "Q13_OverallSatisfaction","Type":"Numeric", "Source": "Survey Q13",    "Description": "Overall brand satisfaction (1–10)"},
    {"Feature": "Q14_NPSScore",         "Type": "Numeric",  "Source": "Survey Q14",    "Description": "Net Promoter Score (0–10)"},
    {"Feature": "Q15_PurchaseIntent30d","Type": "Likert",   "Source": "Survey Q15",    "Description": "30-day repurchase intent"},
    {"Feature": "Q19_RecontactFlag",    "Type": "Binary",   "Source": "Survey Q19",    "Description": "Re-contacted after resolution"},
    {"Feature": "DERIVED_ResolutionBinary","Type":"Binary", "Source": "Derived",       "Description": "Resolution binary (Q9 ≥ 4)"},
    {"Feature": "DERIVED_HighEffort",   "Type": "Binary",   "Source": "Derived",       "Description": "High CES flag (Q11 ≥ 4)"},
    {"Feature": "ResOutcomeCode",       "Type": "Encoded",  "Source": "Engineered Q18","Description": "Resolution outcome (resolved/partial/gaveup/escalated)"},
    {"Feature": "RiskScore",            "Type": "Composite","Source": "Engineered",    "Description": "Weighted composite risk score"},
    {"Feature": "AI_* (6 cols)",        "Type": "Binary",   "Source": "Engineered Q6", "Description": "AI action flags (track/refund/faq/escalate/promo/nothing)"},
    {"Feature": "CD_* (7 cols)",        "Type": "Binary",   "Source": "Engineered Q17","Description": "Churn driver flags (price/quality/support/…)"},
    {"Feature": "CH_* (channel)",       "Type": "Binary",   "Source": "Engineered Q5", "Description": "Contact channel one-hot flags"},
])
st.dataframe(feat_info, use_container_width=True, hide_index=True)
st.caption(f"Total features: {len(feature_cols)} | Target: DERIVED_ChurnBinary (0 = Retained, 1 = Churned)")

# ── Train/Test Split + SMOTE ──────────────────────────────────────────────────
st.markdown("<div class='section-header'>3.2 Train/Test Split & Class Balancing</div>", unsafe_allow_html=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_seed), stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

if use_smote:
    sm = SMOTE(random_state=int(random_seed))
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
else:
    X_train_res, y_train_res = X_train_sc, y_train

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Train Samples", len(X_train))
    st.metric("Test Samples",  len(X_test))
with col2:
    orig_dist = y_train.value_counts()
    st.metric("Train Retained (original)", int(orig_dist.get(0, 0)))
    st.metric("Train Churned (original)",  int(orig_dist.get(1, 0)))
with col3:
    if use_smote:
        res_dist = pd.Series(y_train_res).value_counts()
        st.metric("After SMOTE – Retained", int(res_dist.get(0, 0)))
        st.metric("After SMOTE – Churned",  int(res_dist.get(1, 0)))
    else:
        st.info("SMOTE not applied")

# Class balance chart
b1, b2 = st.columns(2)
with b1:
    orig_df = pd.DataFrame({"Class":["Retained","Churned"], "Count":[int(orig_dist.get(0,0)), int(orig_dist.get(1,0))]})
    fig = px.pie(orig_df, names="Class", values="Count", hole=0.5,
                 color="Class", color_discrete_map={"Retained":PALETTE["green"],"Churned":PALETTE["accent"]},
                 title="Original Train Distribution")
    apply_layout(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)
with b2:
    if use_smote:
        res_df = pd.DataFrame({"Class":["Retained","Churned"], "Count":[int(res_dist.get(0,0)), int(res_dist.get(1,0))]})
        fig2 = px.pie(res_df, names="Class", values="Count", hole=0.5,
                     color="Class", color_discrete_map={"Retained":PALETTE["green"],"Churned":PALETTE["accent"]},
                     title="After SMOTE Distribution")
        apply_layout(fig2, height=260)
        st.plotly_chart(fig2, use_container_width=True)

# ── Train Models ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>3.3 Model Training</div>", unsafe_allow_html=True)

@st.cache_data
def train_all_models(X_tr, y_tr, X_te, X_tr_raw, _k, _dt_d, _rf_n, _rf_d, _gb_n, _gb_lr, seed):
    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=_k),
        "Decision Tree":     DecisionTreeClassifier(max_depth=_dt_d, random_state=seed),
        "Random Forest":     RandomForestClassifier(n_estimators=_rf_n, max_depth=_rf_d, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=_gb_n, learning_rate=_gb_lr, random_state=seed),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred_tr = mdl.predict(X_tr)
        y_pred_te = mdl.predict(X_te)
        y_prob_te = mdl.predict_proba(X_te)[:, 1]
        train_acc = accuracy_score(y_tr, y_pred_tr)
        test_acc  = accuracy_score(y_test, y_pred_te)
        report    = classification_report(y_test, y_pred_te, output_dict=True)
        cm        = confusion_matrix(y_test, y_pred_te)
        roc_auc   = roc_auc_score(y_test, y_prob_te)
        fpr, tpr, _ = roc_curve(y_test, y_prob_te)

        # Feature importances
        fi = None
        if hasattr(mdl, "feature_importances_"):
            fi = mdl.feature_importances_

        results[name] = {
            "model": mdl,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "report": report,
            "cm": cm,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "fi": fi,
            "y_pred": y_pred_te,
            "y_prob": y_prob_te,
        }
    return results

with st.spinner("Training all 4 models…"):
    results = train_all_models(
        X_train_res, y_train_res, X_test_sc, X_train_sc,
        knn_k, dt_depth, rf_trees, rf_depth, gb_trees, gb_lr, int(random_seed)
    )

# Store in session state for next pages
st.session_state["ml_results"]   = results
st.session_state["y_test"]       = y_test
st.session_state["feature_cols"] = feature_cols

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for name, r in results.items():
    rep = r["report"]
    rows.append({
        "Model": name,
        "Train Acc": f"{r['train_acc']:.3f}",
        "Test Acc":  f"{r['test_acc']:.3f}",
        "Overfit Gap": f"{r['train_acc']-r['test_acc']:.3f}",
        "Precision (Churn)": f"{rep['1']['precision']:.3f}",
        "Recall (Churn)":    f"{rep['1']['recall']:.3f}",
        "F1 (Churn)":        f"{rep['1']['f1-score']:.3f}",
        "ROC-AUC":           f"{r['roc_auc']:.3f}",
    })

results_df = pd.DataFrame(rows)
st.dataframe(results_df.style.background_gradient(
    cmap="Greens", subset=["Test Acc","F1 (Churn)","ROC-AUC"]
).background_gradient(cmap="Reds_r", subset=["Overfit Gap"]),
use_container_width=True, hide_index=True)

# ── Train vs Test accuracy bar ────────────────────────────────────────────────
st.markdown("<div class='section-header'>3.4 Train vs Test Accuracy</div>", unsafe_allow_html=True)

acc_df = pd.DataFrame([
    {"Model": n, "Split": "Train", "Accuracy": r["train_acc"]} for n, r in results.items()
] + [
    {"Model": n, "Split": "Test", "Accuracy": r["test_acc"]} for n, r in results.items()
])
fig_acc = px.bar(acc_df, x="Model", y="Accuracy", color="Split", barmode="group",
                 color_discrete_map={"Train": PALETTE["primary"], "Test": PALETTE["accent"]},
                 text_auto=".3f")
fig_acc.update_traces(textposition="outside")
fig_acc.update_layout(yaxis_range=[0, 1.1])
apply_layout(fig_acc, height=360)
st.plotly_chart(fig_acc, use_container_width=True)

# ── Feature Importance (RF) ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>3.5 Feature Importance (Random Forest & Gradient Boosting)</div>", unsafe_allow_html=True)

fi1, fi2 = st.columns(2)
for col_w, model_name in [(fi1, "Random Forest"), (fi2, "Gradient Boosting")]:
    with col_w:
        fi_vals = results[model_name]["fi"]
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": fi_vals})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#D6E4F0","#1A3C5E"],
                        title=f"{model_name} — Top 15 Features")
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        apply_layout(fig_fi, height=400)
        st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("""
<div class='insight-box'>
✅ Models trained successfully. Navigate to <b>Page 4 — Model Evaluation</b> for ROC curves,
confusion matrices, and Precision/Recall/F1 breakdowns.
</div>
""", unsafe_allow_html=True)
