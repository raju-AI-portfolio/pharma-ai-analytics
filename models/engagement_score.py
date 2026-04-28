import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset.csv")

WEIGHTS = {
    "rep_visits":      0.30,
    "email_opens":     0.20,
    "webinar_att":     0.20,
    "clm_sessions":    0.15,
    "recency_score":   0.10,
    "script_momentum": 0.05,
}


def compute_recency_score(days_since):
    """More recent engagement = higher score. Exponential decay over 30 days."""
    return np.exp(-days_since / 30).round(4)


def compute_script_momentum(scripts):
    """Script volume relative to average — above average = positive momentum."""
    mean_s   = scripts.mean()
    momentum = (scripts - mean_s) / (mean_s + 1e-9)
    return np.clip(momentum, -1, 1).round(4)


def build_engagement_scores(df):
    df = df.copy()

    channel_features = ["rep_visits", "email_opens", "webinar_att", "clm_sessions"]
    scaler    = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[channel_features]),
        columns=channel_features,
        index=df.index,
    )

    df["recency_score"]   = compute_recency_score(df["days_since_eng"])
    df["script_momentum"] = MinMaxScaler().fit_transform(
        compute_script_momentum(df["scripts_monthly"]).values.reshape(-1, 1)
    ).flatten()

    score = np.zeros(len(df))
    for col, w in WEIGHTS.items():
        if col in df_scaled.columns:
            score += df_scaled[col] * w
        else:
            score += df[col] * w

    df["eng_score_v2"] = (score * 100).clip(0, 100).round(1)
    return df


def score_summary(df):
    group_col = "segment" if "segment" in df.columns else "tier"
    return (
        df.groupby(group_col)
        .agg(
            avg_score=("eng_score_v2", "mean"),
            median   =("eng_score_v2", "median"),
            pct_90   =("eng_score_v2", lambda x: x.quantile(0.9)),
            count    =("hcp_id",       "count"),
        )
        .round(2)
    )


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = build_engagement_scores(df)

    print("Engagement score distribution:")
    print(df["eng_score_v2"].describe().round(2))

    print("\nScore by tier:")
    print(score_summary(df).to_string())

    out = DATA_PATH.replace(".csv", "_scored.csv")
    df.to_csv(out, index=False)
    print(f"\nScored data → {out}")