import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "mmm_results.json")


def adstock(x, decay=0.5):
    """Geometric adstock — captures carryover effect of spend."""
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result


def hill_saturation(x, alpha=2.0, gamma=0.5):
    """S-curve saturation — captures diminishing returns on spend."""
    x_norm = x / (x.max() + 1e-9)
    return x_norm ** alpha / (x_norm ** alpha + gamma ** alpha)


def build_monthly_data(n_months=24, seed=42):
    np.random.seed(seed)
    months = pd.date_range("2024-05-01", periods=n_months, freq="MS")

    rep_spend     = np.random.uniform(280, 380, n_months)
    email_spend   = np.random.uniform(40,  80,  n_months)
    webinar_spend = np.random.uniform(60,  120, n_months)
    clm_spend     = np.random.uniform(30,  60,  n_months)

    rep_sat     = hill_saturation(adstock(rep_spend,     decay=0.4))
    email_sat   = hill_saturation(adstock(email_spend,   decay=0.3))
    webinar_sat = hill_saturation(adstock(webinar_spend, decay=0.5))
    clm_sat     = hill_saturation(adstock(clm_spend,     decay=0.3))

    season = np.array([
        1.0, 1.02, 1.05, 1.08, 1.1, 1.1,
        1.08, 1.05, 1.0, 0.98, 0.97, 0.98,
        1.0, 1.02, 1.05, 1.08, 1.1, 1.1,
        1.08, 1.05, 1.0, 0.98, 0.97, 0.98,
    ])[:n_months]

    revenue = (
        18000
        + 12000 * rep_sat
        + 8000  * email_sat
        + 6500  * webinar_sat
        + 4500  * clm_sat
        + np.random.normal(0, 500, n_months)
    ) * season

    return pd.DataFrame({
        "month":           months,
        "revenue":         revenue.round(0),
        "rep_spend":       rep_spend.round(1),
        "email_spend":     email_spend.round(1),
        "webinar_spend":   webinar_spend.round(1),
        "clm_spend":       clm_spend.round(1),
        "total_spend":     (rep_spend+email_spend+webinar_spend+clm_spend).round(1),
        "rep_adstock":     rep_sat.round(4),
        "email_adstock":   email_sat.round(4),
        "webinar_adstock": webinar_sat.round(4),
        "clm_adstock":     clm_sat.round(4),
    })


def run_mmm(df):
    y = df["revenue"]
    X = sm.add_constant(df[[
        "rep_adstock", "email_adstock",
        "webinar_adstock", "clm_adstock"
    ]])

    model = sm.OLS(y, X).fit()
    print(model.summary())

    coefs      = model.params.drop("const")
    mean_ads   = df[["rep_adstock","email_adstock",
                      "webinar_adstock","clm_adstock"]].mean()
    contribution  = coefs * mean_ads
    total_contrib = contribution.sum()
    attribution   = (contribution / total_contrib * 100).round(1)

    channel_map = {
        "rep_adstock":     "Rep visits",
        "email_adstock":   "Email",
        "webinar_adstock": "Webinar",
        "clm_adstock":     "CLM",
    }
    spend_map = {
        "rep_adstock":     df["rep_spend"].mean(),
        "email_adstock":   df["email_spend"].mean(),
        "webinar_adstock": df["webinar_spend"].mean(),
        "clm_adstock":     df["clm_spend"].mean(),
    }

    roi = {
        channel_map[k]: round(float(contribution[k] / spend_map[k]), 2)
        for k in channel_map if spend_map[k] > 0
    }

    results = {
        "r_squared":       round(float(model.rsquared), 4),
        "adj_r_squared":   round(float(model.rsquared_adj), 4),
        "attribution_pct": {channel_map[k]: float(v) for k, v in attribution.items()},
        "roi_per_channel": roi,
        "coefficients":    {k: round(float(v), 2) for k, v in coefs.items()},
        "p_values":        {k: round(float(v), 4)
                            for k, v in model.pvalues.drop("const").items()},
        "monthly_revenue": [round(v, 0) for v in df["revenue"].tolist()],
        "months":          [str(m.date()) for m in df["month"]],
        "recommendation":  budget_recommendation(roi),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMMM results saved → {RESULTS_PATH}")
    print(f"\nROI per channel:  {roi}")
    print(f"Attribution %:    {results['attribution_pct']}")
    print(f"R-squared:        {results['r_squared']}")
    return results


def budget_recommendation(roi):
    sorted_roi = sorted(roi.items(), key=lambda x: x[1], reverse=True)
    best, worst = sorted_roi[0], sorted_roi[-1]
    return (
        f"Reallocate 10-15% of '{worst[0]}' budget (ROI {worst[1]:.1f}x) "
        f"to '{best[0]}' (ROI {best[1]:.1f}x) to improve blended ROI. "
        f"Rep visits remain anchor investment. "
        f"Email shows highest marginal ROI due to low cost per touch."
    )


def optimise_budget(total_budget, roi, min_pct=0.05, max_pct=0.60):
    channels = list(roi.keys())
    roi_vals = np.array([roi[c] for c in channels])

    def neg_revenue(w):
        return -np.dot(roi_vals, w * total_budget)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(min_pct, max_pct)] * len(channels)
    res = minimize(neg_revenue, np.ones(len(channels)) / len(channels),
                   method="SLSQP", bounds=bounds, constraints=constraints)

    print(f"\nOptimal budget allocation (total {total_budget:,.0f}K):")
    for ch, w in zip(channels, res.x):
        print(f"  {ch:15s}: {w*100:5.1f}%  budget: {w*total_budget:,.0f}K  "
              f"exp revenue: {w*total_budget*roi[ch]:,.0f}K")


if __name__ == "__main__":
    df  = build_monthly_data()
    res = run_mmm(df)
    optimise_budget(total_budget=600, roi=res["roi_per_channel"])