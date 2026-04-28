import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "segmentation_model.pkl")

FEATURES = [
    "scripts_monthly", "eng_score", "rep_visits",
    "email_opens", "webinar_att", "clm_sessions",
]
SEGMENT_LABELS = ["Champions", "Risers", "Loyalists", "Lapsed"]


def label_segments(df, cluster_col="cluster"):
    avg_scripts = (
        df.groupby(cluster_col)["scripts_monthly"]
        .mean()
        .sort_values(ascending=False)
    )
    rank_map = {
        cluster_id: SEGMENT_LABELS[rank]
        for rank, cluster_id in enumerate(avg_scripts.index)
    }
    return df[cluster_col].map(rank_map)


def train_segmentation(n_clusters=4, random_state=42):
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find best k using silhouette score
    best_k, best_score = n_clusters, -1
    for k in range(2, 8):
        km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km_tmp.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k

    print(f"Best k by silhouette: {best_k} (score={best_score:.3f})")
    print(f"Using k={n_clusters} as per business requirement")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    df["segment"] = label_segments(df)

    summary = (
        df.groupby("segment")
        .agg(
            count       =("hcp_id",          "count"),
            avg_scripts =("scripts_monthly",  "mean"),
            avg_eng     =("eng_score",        "mean"),
            avg_roi     =("roi",              "mean"),
            avg_rep     =("rep_visits",       "mean"),
            avg_email   =("email_opens",      "mean"),
        )
        .round(2)
    )
    print("\nSegment summary:")
    print(summary.to_string())

    artifacts = {
        "kmeans":   km,
        "scaler":   scaler,
        "features": FEATURES,
        "summary":  summary.to_dict(),
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    out_path = DATA_PATH.replace(".csv", "_segmented.csv")
    df.to_csv(out_path, index=False)
    print(f"Segmented data → {out_path}")

    return artifacts


if __name__ == "__main__":
    train_segmentation()