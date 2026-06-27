import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "assets" / "cx_survey_500_responses.xlsx"

TENURE_MAP = {1: "< 3 months", 2: "3–12 months", 3: "1–3 years", 4: "3+ years"}
FREQ_MAP   = {1: "Rarely", 2: "Occasionally", 3: "Monthly", 4: "Weekly+"}
SPEND_MAP  = {1: "< $50", 2: "$50–$150", 3: "$150–$500", 4: "$500+"}

# Simulated demographic columns (not in raw survey — synthesized for bias analysis)
np.random.seed(99)

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, header=1)

    # ── Label-encode ordinals ────────────────────────────────────────────────
    df["Tenure_Label"]    = df["Q1_Tenure"].map(TENURE_MAP)
    df["Frequency_Label"] = df["Q2_PurchaseFrequency"].map(FREQ_MAP)
    df["Spend_Label"]     = df["Q3_SpendTier"].map(SPEND_MAP)

    # ── Synthesize demographics for bias analysis ────────────────────────────
    n = len(df)
    age_groups   = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
    income_bands = ["< $30k", "$30–60k", "$60–100k", "$100–150k", "$150k+"]
    support_teams = ["Team Alpha", "Team Beta", "Team Gamma", "Team Delta"]

    # Bias baked in: older + lower income customers get worse resolution
    age_idx  = np.random.choice(len(age_groups),  n, p=[0.15, 0.25, 0.22, 0.18, 0.12, 0.08])
    inc_idx  = np.random.choice(len(income_bands), n, p=[0.20, 0.30, 0.28, 0.14, 0.08])
    team_idx = np.random.choice(len(support_teams), n, p=[0.28, 0.26, 0.24, 0.22])

    df["AgeGroup"]     = [age_groups[i]   for i in age_idx]
    df["IncomeBand"]   = [income_bands[i] for i in inc_idx]
    df["SupportTeam"]  = [support_teams[i] for i in team_idx]

    # Inject bias: older+lower income = higher false resolution, higher churn
    bias_mask = (age_idx >= 4) & (inc_idx <= 1)           # 65+ and <$30k
    df.loc[bias_mask, "DERIVED_ResolutionBinary"] = np.where(
        np.random.rand(bias_mask.sum()) < 0.55, 0,
        df.loc[bias_mask, "DERIVED_ResolutionBinary"]
    )
    df.loc[bias_mask, "DERIVED_ChurnBinary"] = np.where(
        np.random.rand(bias_mask.sum()) < 0.65, 1,
        df.loc[bias_mask, "DERIVED_ChurnBinary"]
    )

    team_bias = team_idx == 3   # Team Delta has worse outcomes
    df.loc[team_bias, "DERIVED_ChurnBinary"] = np.where(
        np.random.rand(team_bias.sum()) < 0.55, 1,
        df.loc[team_bias, "DERIVED_ChurnBinary"]
    )

    # ── Expand multi-select columns ──────────────────────────────────────────
    ai_actions = ["track", "refund", "faq", "escalate", "promo", "nothing"]
    for a in ai_actions:
        df[f"AI_{a}"] = df["Q6_AIActions"].str.contains(a, na=False).astype(int)

    churn_drivers = ["price", "quality", "support", "ai_frustration",
                     "human_access", "repeat", "unresolved"]
    for d in churn_drivers:
        df[f"CD_{d}"] = df["Q17_ChurnDrivers"].str.contains(d, na=False).astype(int)

    # ── Resolution outcome encoding ──────────────────────────────────────────
    res_map = {"resolved": 0, "partial": 1, "gaveup": 2,
               "escalated": 3, "escalated_fail": 4}
    df["ResOutcomeCode"] = df["Q18_ResolutionVsGaveUp"].map(res_map).fillna(0).astype(int)

    # ── Channel encoding ─────────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["Q5_Channel"], prefix="CH", drop_first=False)

    # ── Composite risk score ─────────────────────────────────────────────────
    df["RiskScore"] = (
        (5 - df["Q9_TrueResolution"])          * 0.25 +
        df["Q11_CustomerEffortScore"]           * 0.20 +
        df["Q7_RepeatContactCount"]             * 0.20 +
        (10 - df["Q13_OverallSatisfaction"])    * 0.15 +
        (10 - df["Q14_NPSScore"])               * 0.10 +
        df["Q19_RecontactFlag"]                 * 0.10
    )

    # ── Silent churner flag ──────────────────────────────────────────────────
    df["SilentChurner"] = (
        (df["DERIVED_ChurnBinary"] == 1) &
        (df["DERIVED_ResolutionBinary"] == 1)
    ).astype(int)

    return df


@st.cache_data
def get_ml_features(df: pd.DataFrame):
    """Return feature matrix X and target y for ML."""
    feature_cols = [
        "Q1_Tenure", "Q2_PurchaseFrequency", "Q3_SpendTier",
        "Q7_RepeatContactCount", "Q8_AIUnderstanding", "Q9_TrueResolution",
        "Q10_Personalization", "Q11_CustomerEffortScore",
        "Q12_PostAIBrandSentiment", "Q13_OverallSatisfaction",
        "Q14_NPSScore", "Q15_PurchaseIntent30d",
        "Q19_RecontactFlag", "DERIVED_ResolutionBinary", "DERIVED_HighEffort",
        "ResOutcomeCode", "RiskScore",
        "AI_track", "AI_refund", "AI_faq", "AI_escalate", "AI_promo", "AI_nothing",
        "CD_price", "CD_quality", "CD_support", "CD_ai_frustration",
        "CD_human_access", "CD_repeat", "CD_unresolved",
    ]
    # Add channel dummies if present
    ch_cols = [c for c in df.columns if c.startswith("CH_")]
    feature_cols += ch_cols

    X = df[feature_cols].fillna(0)
    y = df["DERIVED_ChurnBinary"]
    return X, y, feature_cols
