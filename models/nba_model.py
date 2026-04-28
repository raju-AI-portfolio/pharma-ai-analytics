import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset_segmented.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nba_model.pkl")

CHANNELS = ["Rep Visit", "Email", "Webinar", "CLM", "Med Affairs"]

FEATURES = [
    "scripts_monthly", "eng_score", "rep_visits", "email_opens",
    "webinar_att", "clm_sessions", "days_since_eng",
    "roi", "total_spend", "years_practice",
]


def encode_specialty(df):
    spec_map = {
        "Oncologist": 0, "Haematologist": 1,
        "Internist": 2, "Pulmonologist": 3, "Cardiologist": 4,
    }
    df = df.copy()
    df["specialty_enc"] = df["specialty"].map(spec_map).fillna(2)
    return df


def derive_nba_label(df):
    conditions = [
        (df["tier"] == "Tier1") & (df["days_since_eng"] < 20) & (df["scripts_monthly"] > 12),
        (df["email_opens"] >= 2) & (df["rep_visits"] < 2),
        (df["webinar_att"] >= 1) & (df["tier"] == "Tier2"),
        (df["clm_sessions"] >= 2),
    ]
    choices = [0, 1, 2, 3]
    return pd.Series(np.select(conditions, choices, default=4), index=df.index)


def train_nba(random_state=42):
    df = pd.read_csv(DATA_PATH)
    df = encode_specialty(df)
    df["nba_label"] = derive_nba_label(df)

    feature_cols = FEATURES + ["specialty_enc"]
    X = df[feature_cols].fillna(0)
    y = df["nba_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CHANNELS))

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8).round(4).to_string())

    df["nba_channel"]    = model.predict(X).astype(int)
    df["nba_label_name"] = df["nba_channel"].map(dict(enumerate(CHANNELS)))
    proba                = model.predict_proba(X)
    for i, ch in enumerate(CHANNELS):
        df[f"proba_{ch.replace(' ','_')}"] = proba[:, i].round(3)
    df["nba_score"] = proba.max(axis=1).round(3) * 100

    artifacts = {
        "model":       model,
        "features":    feature_cols,
        "channels":    CHANNELS,
        "cv_accuracy": float(cv_scores.mean()),
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"\nNBA model saved → {MODEL_PATH}")

    out_path = DATA_PATH.replace("_segmented.csv", "_nba.csv")
    df.to_csv(out_path, index=False)
    print(f"NBA-enriched data → {out_path}")

    return artifacts


if __name__ == "__main__":
    train_nba()