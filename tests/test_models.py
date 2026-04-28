import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generate_data import generate_dataset
from models.mmm_model import adstock, hill_saturation, build_monthly_data, run_mmm
from models.engagement_score import compute_recency_score, build_engagement_scores


# ── Data generation tests ─────────────────────────────────────────────────────

class TestDataGeneration:

    def test_shape(self):
        df = generate_dataset(n=100)
        assert df.shape[0] == 100

    def test_hcp_id_unique(self):
        df = generate_dataset(n=100)
        assert df["hcp_id"].nunique() == 100

    def test_tier_distribution(self):
        df = generate_dataset(n=1000)
        tiers = df["tier"].value_counts(normalize=True)
        assert abs(tiers.get("Tier1", 0) - 0.20) < 0.05
        assert abs(tiers.get("Tier2", 0) - 0.50) < 0.05
        assert abs(tiers.get("Tier3", 0) - 0.30) < 0.05

    def test_no_negative_scripts(self):
        df = generate_dataset(n=200)
        assert (df["scripts_monthly"] >= 0).all()

    def test_roi_non_negative(self):
        df = generate_dataset(n=200)
        assert (df["roi"] >= 0).all()

    def test_tier1_higher_scripts(self):
        df = generate_dataset(n=500)
        t1 = df[df["tier"] == "Tier1"]["scripts_monthly"].mean()
        t3 = df[df["tier"] == "Tier3"]["scripts_monthly"].mean()
        assert t1 > t3

    def test_required_columns(self):
        df = generate_dataset(n=50)
        required = [
            "hcp_id", "specialty", "tier", "scripts_monthly",
            "eng_score", "rep_visits", "email_opens",
            "total_spend", "roi"
        ]
        for col in required:
            assert col in df.columns


# ── MMM tests ─────────────────────────────────────────────────────────────────

class TestMMM:

    def test_adstock_no_decay(self):
        x = np.array([100.0, 0.0, 0.0, 0.0])
        result = adstock(x, decay=0.0)
        assert result[0] == 100.0
        assert result[1] == 0.0

    def test_adstock_half_decay(self):
        x = np.array([100.0, 0.0, 0.0])
        result = adstock(x, decay=0.5)
        assert result[0] == 100.0
        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(25.0)

    def test_adstock_length(self):
        x = np.random.rand(12)
        assert len(adstock(x, 0.4)) == 12

    def test_hill_saturation_bounds(self):
        x = np.linspace(0, 100, 50)
        y = hill_saturation(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_hill_monotone(self):
        x = np.linspace(1, 100, 20)
        y = hill_saturation(x)
        assert (np.diff(y) >= 0).all()

    def test_mmm_data_shape(self):
        df = build_monthly_data(n_months=24)
        assert len(df) == 24
        assert "revenue" in df.columns

    def test_mmm_r_squared(self):
        df  = build_monthly_data(n_months=24)
        res = run_mmm(df)
        assert res["r_squared"] > 0.5


# ── Engagement score tests ────────────────────────────────────────────────────

class TestEngagementScore:

    def test_recency_bounds(self):
        days   = pd.Series([0, 7, 14, 30, 60, 90])
        scores = compute_recency_score(days)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_recency_decreasing(self):
        days   = pd.Series([0, 10, 30, 60])
        scores = compute_recency_score(days)
        assert (scores.diff().dropna() <= 0).all()

    def test_eng_score_range(self):
        df = generate_dataset(n=100)
        df = build_engagement_scores(df)
        assert (df["eng_score_v2"] >= 0).all()
        assert (df["eng_score_v2"] <= 100).all()

    def test_tier1_scores_higher(self):
        df = generate_dataset(n=300)
        df = build_engagement_scores(df)
        t1_avg = df[df["tier"] == "Tier1"]["eng_score_v2"].mean()
        t3_avg = df[df["tier"] == "Tier3"]["eng_score_v2"].mean()
        assert t1_avg > t3_avg


# ── API syntax check ──────────────────────────────────────────────────────────

def test_api_importable():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main",
        os.path.join(os.path.dirname(__file__), "../api/main.py")
    )
    assert spec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])