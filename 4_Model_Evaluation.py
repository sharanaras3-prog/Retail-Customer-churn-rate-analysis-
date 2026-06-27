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
                              confusion_matrix, roc_auc_score, roc_curve,
                              precision_recall_curve)
from imblearn.over_sampling import SMOTE
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import load_data, get_ml_features
from plot_style import PALETTE, MODEL_COLORS, apply_layout

st.set_page_config(page_title="Model Evaluation", layout="wide")

st.markdown("""
<style>
.section-header{font-size:18px;font-weight:700;color:#1A3C5E;border-bottom:2px solid #E05C2A;padding-bottom:6px;margin:20px 0 14px;}
.model-tab-title{font-size:15px;font-weight:700;color:#1A3C5E;margin-bottom:8px;}
.metric-small{background:white;border-radius:8px;padding:12px 16px;border:1px solid #E2DDD8;text-align:center;}
.metric-small .val{font-size:22px;font-weight:700;color:#1A3C5E;}
.metric-small .lbl{font-size:11px;color:#7A7066;}
</style>
""", unsafe_allow_html=True)

# ── Retrain models so this page is self-contained ─────────────────────────────
df = load_data()
X, y, feature_cols = get_ml_features(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

@st.cache_data
def train_models(X_tr, y_tr, X_te, y_te):
    configs = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    }
    results = {}
    for name, mdl in configs.items():
        mdl.fit(X_tr, y_tr)
        y_pred_tr = mdl.predict(X_tr)
        y_pred_te = mdl.predict(X_te)
        y_prob_te = mdl.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob_te)
        prec, rec, _ = precision_recall_curve(y_te, y_prob_te)
        results[name] = {
            "model":     mdl,
            "train_acc": accuracy_score(y_tr, y_pred_tr),
            "test_acc":  accuracy_score(y_te, y_pred_te),
            "report":    classification_report(y_te, y_pred_te, output_dict=True),
            "cm":        confusion_matrix(y_te, y_pred_te),
            "roc_auc":   roc_auc_score(y_te, y_prob_te),
            "fpr": fpr, "tpr": tpr,
            "prec": prec, "rec": rec,
            "y_pred": y_pred_te, "y_prob": y_prob_te,
        }
    return results

with st.spinner("Training models for evaluation…"):
    results = train_models(X_train_res, y_train_res, X_test_sc, y_test.values)

st.title("📈 Model Evaluation")
st.caption("ROC-AUC curves, confusion matrices, and Precision/Recall/F1 for all four classifiers.")

# ── 1. Combined ROC Curve ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>4.1 ROC-AUC Curves — All Models</div>", unsafe_allow_html=True)

fig_roc = go.Figure()
fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                  line=dict(dash="dot", color="grey", width=1))
for name, r in results.items():
    fig_roc.add_trace(go.Scatter(
        x=r["fpr"], y=r["tpr"],
        name=f"{name} (AUC={r['roc_auc']:.3f})",
        mode="lines", line=dict(color=MODEL_COLORS[name], width=2.5),
    ))
fig_roc.update_layout(
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.8)"),
)
apply_layout(fig_roc, title="ROC-AUC Curves", height=420)
st.plotly_chart(fig_roc, use_container_width=True)

# AUC table
auc_df = pd.DataFrame([
    {"Model": n, "ROC-AUC": r["roc_auc"], "Test Accuracy": r["test_acc"]}
    for n, r in results.items()
]).sort_values("ROC-AUC", ascending=False)
auc_df["ROC-AUC"] = auc_df["ROC-AUC"].round(4)
auc_df["Test Accuracy"] = auc_df["Test Accuracy"].round(4)
st.dataframe(auc_df.style.background_gradient(cmap="Greens", subset=["ROC-AUC","Test Accuracy"]),
             use_container_width=True, hide_index=True)

# ── 2. Precision-Recall Curves ────────────────────────────────────────────────
st.markdown("<div class='section-header'>4.2 Precision-Recall Curves</div>", unsafe_allow_html=True)

fig_pr = go.Figure()
for name, r in results.items():
    fig_pr.add_trace(go.Scatter(
        x=r["rec"], y=r["prec"],
        name=name, mode="lines",
        line=dict(color=MODEL_COLORS[name], width=2.5),
    ))
fig_pr.update_layout(
    xaxis_title="Recall", yaxis_title="Precision",
    legend=dict(x=0.55, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
)
apply_layout(fig_pr, title="Precision-Recall Curves (focus on Churn class)", height=380)
st.plotly_chart(fig_pr, use_container_width=True)

# ── 3. Per-model Confusion Matrices ──────────────────────────────────────────
st.markdown("<div class='section-header'>4.3 Confusion Matrices</div>", unsafe_allow_html=True)

model_names = list(results.keys())
cm_cols = st.columns(2)
for idx, name in enumerate(model_names):
    with cm_cols[idx % 2]:
        r   = results[name]
        cm  = r["cm"]
        TN, FP, FN, TP = cm.ravel()
        color = MODEL_COLORS[name]

        st.markdown(f"<div class='model-tab-title'>{name}</div>", unsafe_allow_html=True)

        # KPI row
        kc1, kc2, kc3, kc4 = st.columns(4)
        for col_w, val, lbl, bg in [
            (kc1, TP, "True Positives", "#E8F5E9"),
            (kc2, TN, "True Negatives", "#E8F5E9"),
            (kc3, FP, "False Positives", "#FEEBEB"),
            (kc4, FN, "False Negatives", "#FEF3E2"),
        ]:
            col_w.markdown(f"""
            <div style='background:{bg};border-radius:8px;padding:10px;text-align:center;margin-bottom:8px'>
              <div style='font-size:22px;font-weight:700;color:#1A3C5E'>{val}</div>
              <div style='font-size:10px;color:#7A7066'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # Heatmap
        cm_df = pd.DataFrame(cm,
                              index=["Actual Retained","Actual Churned"],
                              columns=["Pred Retained","Pred Churned"])
        annot = [[f"TN={TN}", f"FP={FP}"], [f"FN={FN}", f"TP={TP}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred Retained","Pred Churned"],
            y=["Actual Retained","Actual Churned"],
            colorscale=[[0,"#EBF3FB"],[1, color]],
            text=annot, texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(t=20,b=20,l=20,r=20),
            height=220,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("---")

# ── 4. Precision / Recall / F1 grouped bar ─────────────────────────────────
st.markdown("<div class='section-header'>4.4 Precision / Recall / F1-Score Comparison</div>", unsafe_allow_html=True)

prf_rows = []
for name, r in results.items():
    rep = r["report"]
    for cls_key, cls_label in [("0","Retained (0)"),("1","Churned (1)")]:
        prf_rows += [
            {"Model":name,"Class":cls_label,"Metric":"Precision","Value":rep[cls_key]["precision"]},
            {"Model":name,"Class":cls_label,"Metric":"Recall",   "Value":rep[cls_key]["recall"]},
            {"Model":name,"Class":cls_label,"Metric":"F1-Score", "Value":rep[cls_key]["f1-score"]},
        ]

prf_df = pd.DataFrame(prf_rows)

for cls in ["Churned (1)","Retained (0)"]:
    sub = prf_df[prf_df["Class"] == cls]
    fig_prf = px.bar(sub, x="Model", y="Value", color="Metric", barmode="group",
                     color_discrete_map={"Precision":PALETTE["primary"],"Recall":PALETTE["accent"],"F1-Score":PALETTE["green"]},
                     text_auto=".3f", title=f"Precision / Recall / F1 — {cls}")
    fig_prf.update_traces(textposition="outside")
    fig_prf.update_layout(yaxis_range=[0,1.15])
    apply_layout(fig_prf, height=340)
    st.plotly_chart(fig_prf, use_container_width=True)

# ── 5. Train vs Test Accuracy ─────────────────────────────────────────────────
st.markdown("<div class='section-header'>4.5 Training vs Testing Accuracy</div>", unsafe_allow_html=True)

acc_rows = []
for name, r in results.items():
    acc_rows.append({"Model":name,"Split":"Training","Accuracy":r["train_acc"]})
    acc_rows.append({"Model":name,"Split":"Testing", "Accuracy":r["test_acc"]})
acc_df = pd.DataFrame(acc_rows)

fig_acc = px.bar(acc_df, x="Model", y="Accuracy", color="Split", barmode="group",
                 color_discrete_map={"Training":PALETTE["primary"],"Testing":PALETTE["accent"]},
                 text_auto=".3f")
fig_acc.update_traces(textposition="outside")
fig_acc.update_layout(yaxis_range=[0,1.15])
apply_layout(fig_acc, title="Train vs Test Accuracy — Overfitting Check", height=360)
st.plotly_chart(fig_acc, use_container_width=True)

# ── 6. Model Scorecard ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>4.6 Full Model Scorecard</div>", unsafe_allow_html=True)

scorecard = []
for name, r in results.items():
    rep = r["report"]
    scorecard.append({
        "Model":              name,
        "Train Acc":          round(r["train_acc"], 4),
        "Test Acc":           round(r["test_acc"],  4),
        "Overfit (Train-Test)": round(r["train_acc"]-r["test_acc"], 4),
        "Precision (0)":      round(rep["0"]["precision"], 4),
        "Recall (0)":         round(rep["0"]["recall"],    4),
        "F1 (0)":             round(rep["0"]["f1-score"],  4),
        "Precision (1)":      round(rep["1"]["precision"], 4),
        "Recall (1)":         round(rep["1"]["recall"],    4),
        "F1 (1)":             round(rep["1"]["f1-score"],  4),
        "ROC-AUC":            round(r["roc_auc"],          4),
        "Macro F1":           round(rep["macro avg"]["f1-score"], 4),
    })

sc_df = pd.DataFrame(scorecard)
st.dataframe(
    sc_df.style
    .background_gradient(cmap="Greens",  subset=["Test Acc","F1 (1)","ROC-AUC","Macro F1"])
    .background_gradient(cmap="Reds_r",  subset=["Overfit (Train-Test)"])
    .background_gradient(cmap="Oranges", subset=["Recall (1)"]),
    use_container_width=True, hide_index=True
)
st.caption("Class 0 = Retained | Class 1 = Churned (the minority / target class)")
