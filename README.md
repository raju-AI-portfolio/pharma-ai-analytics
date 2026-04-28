# Pharma Omnichannel AI Analytics Platform

End-to-end AI system for pharmaceutical commercial analytics.

## Models Built
- **HCP Segmentation** — KMeans clustering (k=4) → Champions, Risers, Loyalists, Lapsed
- **Next-Best-Action** — XGBoost classifier, 99% CV accuracy
- **Marketing Mix Model** — OLS with adstock decay + Hill saturation, R²=0.647
- **Engagement Scorer** — Composite 0–100 index with recency weighting

## Key Results
| Metric | Value |
|--------|-------|
| HCPs analysed | 500 |
| NBA model accuracy | 99% |
| MMM R-squared | 0.647 |
| Highest ROI channel | CLM 165.9x |
| Unit tests | 19 passed |

## Tech Stack
Python · scikit-learn · XGBoost · statsmodels · FastAPI · Streamlit · Plotly · Anthropic Claude

## Quickstart
```bash
pip install -r requirements.txt
python data/generate_data.py
python models/segmentation.py
python models/nba_model.py
python models/mmm_model.py
uvicorn api.main:app --port 8000
streamlit run dashboard/app.py
```

## API Endpoints
| Endpoint | Description |
|----------|-------------|
| GET /kpis | Headline KPIs |
| GET /hcps | All HCP records |
| GET /nba/{id} | NBA recommendation |
| GET /segments/summary | Segment statistics |
| GET /mmm/results | MMM attribution |
| GET /roi/channels | Channel ROI |
| POST /insights | AI chat (Claude) |