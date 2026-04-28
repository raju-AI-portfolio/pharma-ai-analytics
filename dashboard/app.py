import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Pharma Omnichannel AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset_nba.csv")
SEG_PATH  = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset_segmented.csv")
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data/hcp_dataset.csv")

SEGMENT_COLORS = {
    "Champions": "#378ADD",
    "Risers":    "#1D9E75",
    "Loyalists": "#7F77DD",
    "Lapsed":    "#D85A30",
}

@st.cache_data
def load_data():
    for path in [DATA_PATH, SEG_PATH, BASE_PATH]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("NEXAVIR Analytics")
st.sidebar.markdown("**Therapeutic Area:** Oncology")
st.sidebar.divider()

sel_tier = st.sidebar.multiselect(
    "Tier", ["Tier1","Tier2","Tier3"],
    default=["Tier1","Tier2","Tier3"]
)
sel_region = st.sidebar.multiselect(
    "Region", sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)
sel_spec = st.sidebar.multiselect(
    "Specialty", sorted(df["specialty"].unique()),
    default=sorted(df["specialty"].unique())
)

mask = (
    df["tier"].isin(sel_tier) &
    df["region"].isin(sel_region) &
    df["specialty"].isin(sel_spec)
)
fdf = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Pharma Omnichannel AI Analytics")
st.caption(
    f"Brand: NEXAVIR  |  Oncology  |  "
    f"HCPs: {len(fdf):,}  |  "
    f"Models: KMeans + XGBoost + OLS MMM"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total HCPs",      f"{len(fdf):,}")
k2.metric("Avg Engagement",  f"{fdf['eng_score'].mean():.1f}")
k3.metric("Blended ROI",     f"{fdf['roi'].mean():.1f}x")
k4.metric("Avg Script Lift", f"+{fdf['script_lift_pct'].mean():.1f}%")
k5.metric("Total Revenue",   f"₹{fdf['revenue'].sum():,.0f}K")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "HCP Segmentation",
    "Next-Best-Action",
    "Marketing Mix Model",
    "Regional Analysis",
    "Driver Analysis",
    "Data Explorer",
])

# ── Tab 1: Segmentation ───────────────────────────────────────────────────────
with t1:
    st.subheader("KMeans HCP Segmentation (k=4)")

    if "segment" not in fdf.columns:
        st.warning("Run models/segmentation.py to generate segments.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            seg_counts = fdf["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment","Count"]
            fig = px.pie(
                seg_counts, names="Segment", values="Count",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                hole=0.45, title="HCP distribution by segment"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_summary = fdf.groupby("segment").agg(
                HCPs        =("hcp_id",         "count"),
                Avg_Scripts =("scripts_monthly", "mean"),
                Avg_Eng     =("eng_score",       "mean"),
                Avg_ROI     =("roi",             "mean"),
            ).round(2).reset_index()
            st.dataframe(seg_summary, use_container_width=True, hide_index=True)

        fig2 = px.scatter(
            fdf, x="eng_score", y="scripts_monthly",
            color="segment", size="roi",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["hcp_id","tier","specialty"],
            labels={
                "eng_score":       "Engagement Score",
                "scripts_monthly": "Scripts / Month",
            },
            title="Engagement vs Scripts — coloured by segment",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "43% of HCPs are Lapsed (low engagement + low scripts). "
            "These represent the highest untapped revenue opportunity. "
            "Recommend a 4-touch reactivation sequence: "
            "Rep call → Email → Webinar → CLM follow-up."
        )

# ── Tab 2: NBA ────────────────────────────────────────────────────────────────
with t2:
    st.subheader("Next-Best-Action Recommendations")

    nba_col = "nba_label_name" if "nba_label_name" in fdf.columns else None

    if nba_col:
        col1, col2 = st.columns(2)
        with col1:
            dist = fdf[nba_col].value_counts().reset_index()
            dist.columns = ["Channel","Count"]
            fig = px.bar(
                dist, x="Channel", y="Count", color="Channel",
                title="NBA channel distribution across all HCPs"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "segment" in fdf.columns:
                heat = fdf.groupby(["segment", nba_col]).size().unstack(fill_value=0)
                fig2 = px.imshow(
                    heat, color_continuous_scale="Blues",
                    title="NBA channel by segment"
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 20 HCPs by NBA Score")
    score_col = "nba_score" if "nba_score" in fdf.columns else "eng_score"
    cols = ["hcp_id","specialty","tier","scripts_monthly",
            "eng_score", score_col]
    if nba_col: cols.append(nba_col)
    st.dataframe(
        fdf.nlargest(20, score_col)[cols].round(2),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: MMM ────────────────────────────────────────────────────────────────
with t3:
    st.subheader("Marketing Mix Model — Channel Attribution & ROI")

    channel_roi = {"Webinar": 119.6, "CLM": 165.9, "Rep visits": 18.1, "Email": -32.9}
    attribution = {"Rep visits": 38, "Email": 22, "Webinar": 18, "CLM": 12, "Med Affairs": 10}

    col1, col2 = st.columns(2)
    with col1:
        roi_df = pd.DataFrame({
            "Channel": list(channel_roi.keys()),
            "ROI (x)": list(channel_roi.values())
        })
        fig = px.bar(
            roi_df, x="ROI (x)", y="Channel", orientation="h",
            color="ROI (x)", color_continuous_scale="Teal",
            title="Channel ROI — revenue per ₹1 spend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        attr_df = pd.DataFrame({
            "Channel":       list(attribution.keys()),
            "Attribution %": list(attribution.values())
        })
        fig2 = px.pie(
            attr_df, names="Channel", values="Attribution %",
            title="Revenue attribution by channel"
        )
        st.plotly_chart(fig2, use_container_width=True)

    months  = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr"]
    scripts = [420,424,422,441,441,447,443,502,500,496,498,508]
    spend   = [86,86,98,95,100,92,100,98,108,113,117,109]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=months, y=scripts, name="Scripts/mo",
        line=dict(color="#378ADD", width=2.5)
    ))
    fig3.add_trace(go.Scatter(
        x=months, y=spend, name="Spend index",
        line=dict(color="#D85A30", width=2, dash="dot"),
        yaxis="y2"
    ))
    fig3.update_layout(
        title="12-month scripts vs spend trend",
        yaxis =dict(title="Scripts/mo"),
        yaxis2=dict(title="Spend index", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "MMM R² = 0.647. CLM delivers highest ROI (165.9x) followed by Webinar (119.6x). "
        "Email shows negative standalone ROI — effective only when sequenced after rep visits. "
        "Recommend shifting 15% of rep budget to CLM + Webinar."
    )

# ── Tab 4: Regional ──────────────────────────────────────────────────────────
with t4:
    st.subheader("Regional Performance")

    reg = fdf.groupby("region").agg(
        HCPs        =("hcp_id",         "count"),
        Avg_Scripts =("scripts_monthly", "mean"),
        Avg_Eng     =("eng_score",       "mean"),
        Total_Rev   =("revenue",         "sum"),
    ).round(1).reset_index()

    fig = px.bar(
        reg, x="region", y="Total_Rev",
        color="Avg_Eng", color_continuous_scale="Blues",
        title="Total revenue by region  (colour = avg engagement score)",
        labels={"Total_Rev": "Revenue (₹K)", "region": "Region"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(reg, use_container_width=True, hide_index=True)

# ── Tab 5: Driver Analysis ───────────────────────────────────────────────────
with t5:
    st.subheader("Channel → Script Lift: Pearson Correlations")

    channels = ["rep_visits","email_opens","webinar_att","clm_sessions"]
    corrs = {
        c: round(float(fdf[c].corr(fdf["script_lift_pct"])), 3)
        for c in channels
    }
    corr_df = pd.DataFrame({
        "Channel":   list(corrs.keys()),
        "Pearson r": list(corrs.values())
    })

    fig = px.bar(
        corr_df, x="Pearson r", y="Channel", orientation="h",
        color="Pearson r", color_continuous_scale="RdBu",
        range_color=[-0.3, 0.4],
        title="Correlation with script lift %"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key findings from your model:**
- Rep visits (r≈0.29) are the strongest single driver of script lift
- CLM sessions show meaningful digital impact
- Rep × CLM sequencing produces a 2× synergy effect
- Email ROI is channel-dependent — works best as a follow-up touch
    """)

# ── Tab 6: Data Explorer ─────────────────────────────────────────────────────
with t6:
    st.subheader("Raw Data Explorer")

    default_cols = ["hcp_id","specialty","tier","region",
                    "scripts_monthly","eng_score","roi"]
    if "segment"      in fdf.columns: default_cols.append("segment")
    if "nba_label_name" in fdf.columns: default_cols.append("nba_label_name")

    cols = st.multiselect("Columns to show", fdf.columns.tolist(),
                          default=default_cols)
    st.dataframe(fdf[cols].head(200), use_container_width=True, height=400)

    csv = fdf.to_csv(index=False).encode()
    st.download_button(
        "Download full dataset as CSV",
        csv, "hcp_data.csv", "text/csv"
    )