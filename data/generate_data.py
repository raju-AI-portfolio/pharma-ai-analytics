import pandas as pd
import numpy as np
import os

SEED = 42
N = 500
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "hcp_dataset.csv")

def tier_val(tier_arr, tier1, tier2, tier3):
    return np.where(tier_arr == "Tier1", tier1,
           np.where(tier_arr == "Tier2", tier2, tier3))

def generate_dataset(n=N, seed=SEED):
    np.random.seed(seed)

    hcp_ids   = [f"HCP{str(i).zfill(4)}" for i in range(1, n+1)]
    specialty = np.random.choice(
        ["Oncologist","Haematologist","Internist","Pulmonologist","Cardiologist"],
        n, p=[0.35, 0.20, 0.25, 0.10, 0.10]
    )
    region         = np.random.choice(["North","South","East","West","Central"], n)
    tiers          = np.random.choice(["Tier1","Tier2","Tier3"], n, p=[0.20,0.50,0.30])
    years_practice = np.random.randint(3, 36, n)

    base_scripts    = tier_val(tiers, 18, 8, 2)
    scripts_monthly = np.array([np.random.poisson(b) for b in base_scripts])

    rep_visits   = np.array([np.random.poisson(p)     for p in tier_val(tiers, 2.5, 1.5, 0.5)])
    email_opens  = np.array([np.random.binomial(8, p) for p in tier_val(tiers, 0.35, 0.25, 0.15)])
    webinar_att  = np.array([np.random.binomial(4, p) for p in tier_val(tiers, 0.40, 0.25, 0.10)])
    clm_sessions = np.array([np.random.binomial(3, p) for p in tier_val(tiers, 0.50, 0.30, 0.10)])

    eng_score = (rep_visits*15 + email_opens*8 + webinar_att*12 + clm_sessions*10).astype(float)
    eng_score = np.clip(eng_score + np.random.normal(0, 5, n), 0, 100).round(1)

    script_lift = (0.30*rep_visits + 0.15*email_opens + 0.20*webinar_att + 0.10*clm_sessions)
    script_lift = np.clip(script_lift + np.random.normal(0, 2, n), -10, 60).round(1)

    rep_spend     = (rep_visits   * np.random.uniform(8,  12, n)).round(1)
    email_spend   = (email_opens  * np.random.uniform(0.5, 1, n)).round(1)
    webinar_spend = (webinar_att  * np.random.uniform(3,   5, n)).round(1)
    clm_spend     = (clm_sessions * np.random.uniform(2,   4, n)).round(1)
    total_spend   = (rep_spend + email_spend + webinar_spend + clm_spend).round(1)

    revenue = (scripts_monthly * np.random.uniform(45, 55, n)).round(1)
    roi     = np.where(total_spend > 0, (revenue / total_spend).round(2), 0.0)

    days_since = tier_val(tiers, 14, 30, 90).astype(int) + np.random.randint(0, 15, n)

    return pd.DataFrame({
        "hcp_id": hcp_ids, "specialty": specialty, "region": region,
        "tier": tiers, "years_practice": years_practice,
        "scripts_monthly": scripts_monthly,
        "rep_visits": rep_visits, "email_opens": email_opens,
        "webinar_att": webinar_att, "clm_sessions": clm_sessions,
        "eng_score": eng_score, "script_lift_pct": script_lift,
        "rep_spend": rep_spend, "email_spend": email_spend,
        "webinar_spend": webinar_spend, "clm_spend": clm_spend,
        "total_spend": total_spend, "revenue": revenue,
        "roi": roi, "days_since_eng": days_since,
    })

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved → {OUTPUT_PATH}")
    print(f"Shape : {df.shape}")
    print(f"Tiers : {df['tier'].value_counts().to_dict()}")
    print(df.describe().round(2))