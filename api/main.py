import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE, "../data")
MODELS_DIR = os.path.join(BASE, "../models")

def _data(name):  return os.path.join(DATA_DIR,   name)
def _model(name): return os.path.join(MODELS_DIR, name)

app = FastAPI(
    title="Pharma Omnichannel AI API",
    description="HCP segmentation, NBA, MMM and AI insights.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

hcp_df        = pd.DataFrame()
seg_artifacts = None
nba_artifacts = None
mmm_results   = {}

@app.on_event("startup")
def load_artifacts():
    global hcp_df, seg_artifacts, nba_artifacts, mmm_results

    for name in ["hcp_dataset_nba.csv", "hcp_dataset_segmented.csv", "hcp_dataset.csv"]:
        path = _data(name)
        if os.path.exists(path):
            hcp_df = pd.read_csv(path)
            print(f"Loaded data: {name} ({len(hcp_df)} rows)")
            break

    if os.path.exists(_model("segmentation_model.pkl")):
        seg_artifacts = joblib.load(_model("segmentation_model.pkl"))
        print("Loaded segmentation model")

    if os.path.exists(_model("nba_model.pkl")):
        nba_artifacts = joblib.load(_model("nba_model.pkl"))
        print("Loaded NBA model")

    if os.path.exists(_model("mmm_results.json")):
        with open(_model("mmm_results.json")) as f:
            mmm_results = json.load(f)
        print("Loaded MMM results")


# ── Schemas ──────────────────────────────────────────────────────────────────

class InsightRequest(BaseModel):
    question: str

class NBARequest(BaseModel):
    scripts_monthly: float = 8.0
    eng_score:       float = 50.0
    rep_visits:      float = 1.5
    email_opens:     float = 2.0
    webinar_att:     float = 0.5
    clm_sessions:    float = 0.5
    days_since_eng:  float = 30.0
    roi:             float = 10.0
    total_spend:     float = 25.0
    years_practice:  int   = 10
    specialty_enc:   int   = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Pharma Omnichannel AI API v1.0"}


@app.get("/kpis")
def get_kpis():
    if hcp_df.empty:
        raise HTTPException(404, "Data not loaded.")
    return {
        "total_hcps":      int(len(hcp_df)),
        "avg_eng_score":   round(float(hcp_df["eng_score"].mean()), 1),
        "avg_roi":         round(float(hcp_df["roi"].mean()), 2),
        "avg_script_lift": round(float(hcp_df["script_lift_pct"].mean()), 1),
        "total_revenue":   round(float(hcp_df["revenue"].sum()), 0),
        "total_spend":     round(float(hcp_df["total_spend"].sum()), 0),
        "tier_counts":     hcp_df["tier"].value_counts().to_dict(),
    }


@app.get("/hcps")
def get_hcps(
    limit:     int = 50,
    segment:   Optional[str] = None,
    tier:      Optional[str] = None,
    specialty: Optional[str] = None,
):
    if hcp_df.empty:
        raise HTTPException(404, "Data not loaded.")
    df = hcp_df.copy()
    if segment  and "segment" in df.columns: df = df[df["segment"]   == segment]
    if tier:                                  df = df[df["tier"]      == tier]
    if specialty:                             df = df[df["specialty"] == specialty]
    return df.head(limit).fillna(0).to_dict(orient="records")


@app.get("/hcps/{hcp_id}")
def get_hcp(hcp_id: str):
    if hcp_df.empty:
        raise HTTPException(404, "Data not loaded.")
    row = hcp_df[hcp_df["hcp_id"] == hcp_id]
    if row.empty:
        raise HTTPException(404, f"{hcp_id} not found.")
    return row.iloc[0].fillna(0).to_dict()


@app.get("/nba/{hcp_id}")
def get_nba(hcp_id: str):
    if hcp_df.empty:
        raise HTTPException(404, "Data not loaded.")
    row = hcp_df[hcp_df["hcp_id"] == hcp_id]
    if row.empty:
        raise HTTPException(404, f"{hcp_id} not found.")
    r = row.iloc[0]

    if "nba_label_name" in hcp_df.columns:
        channel = r.get("nba_label_name", "Email")
        score   = float(r.get("nba_score", 75.0))
    else:
        channel, score = rule_based_nba(r)

    return {
        "hcp_id":              hcp_id,
        "recommended_channel": channel,
        "confidence_score":    round(score, 1),
        "rationale":           nba_rationale(r, channel),
        "next_steps":          next_steps(channel),
    }


@app.get("/segments/summary")
def get_segments():
    if hcp_df.empty or "segment" not in hcp_df.columns:
        raise HTTPException(404, "Segmented data not loaded.")
    summary = (
        hcp_df.groupby("segment")
        .agg(
            count       =("hcp_id",         "count"),
            avg_scripts =("scripts_monthly", "mean"),
            avg_eng     =("eng_score",       "mean"),
            avg_roi     =("roi",             "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )
    return {"segments": summary}


@app.get("/mmm/results")
def get_mmm():
    if not mmm_results:
        raise HTTPException(404, "MMM results not found. Run mmm_model.py first.")
    return mmm_results


@app.get("/roi/channels")
def get_roi():
    if mmm_results and "roi_per_channel" in mmm_results:
        roi = mmm_results["roi_per_channel"]
        rec = mmm_results.get("recommendation", "")
    else:
        roi = {"Email": 64.4, "Webinar": 20.5, "CLM": 19.3, "Rep visits": 11.4}
        rec = "Run MMM model for precise attribution."
    return {
        "roi_by_channel": roi,
        "recommendation": rec,
        "blended_roi":    round(sum(roi.values()) / len(roi), 2),
    }


@app.post("/insights")
async def get_insights(req: InsightRequest):
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"answer": "Set ANTHROPIC_API_KEY environment variable to enable AI insights."}

        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 512,
            system     = (
                "You are an expert pharma omnichannel analytics AI for brand NEXAVIR (Oncology). "
                "You have 500 HCPs segmented into Champions (78), Risers (107), "
                "Loyalists (164), Lapsed (151). "
                "Channel ROI: Webinar 120x, CLM 166x, Rep visits 18x, Email negative. "
                "NBA model accuracy: 99%. MMM R-squared: 0.647. "
                "Answer in 3-4 sentences. Be data-driven and actionable."
            ),
            messages=[{"role": "user", "content": req.question}],
        )
        return {"answer": message.content[0].text}
    except Exception as e:
        return {"answer": f"AI insights error: {str(e)}"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def rule_based_nba(row):
    if str(row.get("tier")) == "Tier1" and float(row.get("days_since_eng", 999)) < 21:
        return "Rep Visit", 90.0
    if float(row.get("email_opens", 0)) >= 2:
        return "Email", 82.0
    if str(row.get("tier")) == "Tier2":
        return "Webinar", 76.0
    return "CLM", 70.0

def nba_rationale(row, channel):
    return (
        f"{row.get('tier')} HCP with {row.get('scripts_monthly', 0):.0f} scripts/mo, "
        f"engagement score {row.get('eng_score', 0):.0f}. "
        f"Last engaged {row.get('days_since_eng', 0):.0f} days ago. "
        f"'{channel}' has highest predicted response probability."
    )

def next_steps(channel):
    steps = {
        "Rep Visit":   ["Schedule within 7 days", "Bring clinical dossier", "Follow up with CLM"],
        "Email":       ["Send personalised e-detail", "Track open rate", "Follow up in 5 days"],
        "Webinar":     ["Invite to next KOL event", "Send pre-event teaser", "Post-event follow-up"],
        "CLM":         ["Schedule CLM via rep", "Use disease awareness deck", "Capture feedback"],
        "Med Affairs": ["Request MSL engagement", "Share HEOR data", "Arrange advisory board"],
    }
    return steps.get(channel, ["Contact HCP"])