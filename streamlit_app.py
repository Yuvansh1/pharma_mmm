"""
Pharma MMM Dashboard - Streamlit UI
"""

import time

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Pharma MMM Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #e6edf3;
}

.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #58a6ff;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.channel-bar-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.channel-name {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 0.3rem;
}

.channel-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    width: 100%;
}

.channel-bar-fill {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
}

.channel-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #58a6ff;
    margin-top: 0.3rem;
}

.insight-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1d9;
}

.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb55;
}

.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #388bfd;
    color: white;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextInput"] input {
    background: #161b22;
    border: 1px solid #21262d;
    color: #e6edf3;
    border-radius: 6px;
}

.stSlider > div > div > div {
    background: #1f6feb;
}

hr {
    border-color: #21262d;
}

.stDataFrame {
    background: #161b22;
}
</style>
""",
    unsafe_allow_html=True,
)


def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=60)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint, payload=None):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload or {}, timeout=120)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


CHANNEL_LABELS = {
    "log_rep_visits": "Sales Rep Visits",
    "log_speaker_programs": "Speaker Programs",
    "log_samples_distributed": "Free Samples",
    "log_emails_sent": "Emails",
    "log_ad_clicks": "Ad Clicks",
    "log_samples_lag2": "Samples (2-wk lag)",
    "log_seasonality": "Seasonality",
}

CHANNEL_COLORS = {
    "log_rep_visits": "#58a6ff",
    "log_speaker_programs": "#3fb950",
    "log_samples_distributed": "#d2a8ff",
    "log_emails_sent": "#ffa657",
    "log_ad_clicks": "#f85149",
    "log_samples_lag2": "#a5d6ff",
    "log_seasonality": "#8b949e",
}

# Sidebar
with st.sidebar:
    st.markdown(
        "<h2 style='margin-top:0; font-size:1.3rem;'>💊 Pharma MMM</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8b949e; font-size:0.8rem;'>Marketing Mix Modeling Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    health = api_get("/health")
    if health:
        trained = health.get("model_trained", False)
        status_color = "#3fb950" if trained else "#f85149"
        status_text = "Model Ready" if trained else "Not Trained"
        st.markdown(
            f"<span class='status-pill' style='background:{status_color}22; color:{status_color}; border-color:{status_color}55;'>{status_text}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error("API offline — start the FastAPI server first.")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Channel Elasticities",
            "ROI Analysis",
            "Scenario Simulator",
            "LLM Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("Train Model"):
        with st.spinner("Training..."):
            result = api_post("/train")
            if result:
                st.success(f"Trained — Train R²: {result['train_metrics']['r2']}")
            else:
                st.error("Training failed.")


# --- Overview ---
if page == "Overview":
    st.markdown("## Marketing Mix Model")
    st.markdown(
        "<p style='color:#8b949e;'>Statistical attribution of Rx claims across pharma marketing channels.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    health = api_get("/health")
    if not health or not health.get("model_trained"):
        st.info("Train the model first using the sidebar button.")
    else:
        elast = api_get("/elasticities")
        roi = api_get("/roi")

        col1, col2, col3, col4 = st.columns(4)

        channel_elast = (
            {k: v for k, v in elast.items() if "seasonality" not in k and "log_" in k}
            if elast
            else {}
        )

        with col1:
            top = (
                max(channel_elast, key=lambda k: channel_elast[k])
                if channel_elast
                else "N/A"
            )
            st.markdown(
                f"""<div class='metric-card'>
                <div class='metric-value'>{CHANNEL_LABELS.get(top, top)}</div>
                <div class='metric-label'>Top Channel</div>
            </div>""",
                unsafe_allow_html=True,
            )

        with col2:
            total = len(
                [k for k in (elast or {}) if "log_" in k and "seasonality" not in k]
            )
            st.markdown(
                f"""<div class='metric-card'>
                <div class='metric-value'>{total}</div>
                <div class='metric-label'>Channels Tracked</div>
            </div>""",
                unsafe_allow_html=True,
            )

        with col3:
            best_roi = max(roi, key=lambda k: roi[k]) if roi else "N/A"
            best_val = round(roi[best_roi], 3) if roi and best_roi != "N/A" else "N/A"
            st.markdown(
                f"""<div class='metric-card'>
                <div class='metric-value'>{best_val}</div>
                <div class='metric-label'>Best ROI Score</div>
            </div>""",
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """<div class='metric-card'>
                <div class='metric-value'>OLS</div>
                <div class='metric-label'>Model Type</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Channel Elasticities")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.85rem;'>1% increase in channel activity → X% change in Rx claims</p>",
            unsafe_allow_html=True,
        )

        if channel_elast:
            sorted_elast = sorted(
                channel_elast.items(), key=lambda x: abs(x[1]), reverse=True
            )
            max_val = max(abs(v) for _, v in sorted_elast) or 1

            for k, v in sorted_elast:
                label = CHANNEL_LABELS.get(k, k)
                pct = int((abs(v) / max_val) * 100)
                color = "#3fb950" if v > 0 else "#f85149"
                st.markdown(
                    f"""<div class='channel-bar-wrap'>
                    <div class='channel-name'>{label}</div>
                    <div class='channel-bar-bg'>
                        <div class='channel-bar-fill' style='width:{pct}%; background:{color}88;'></div>
                    </div>
                    <div class='channel-value' style='color:{color};'>{v:+.4f}</div>
                </div>""",
                    unsafe_allow_html=True,
                )


# --- Channel Elasticities ---
elif page == "Channel Elasticities":
    st.markdown("## Channel Elasticities")
    st.markdown(
        "<p style='color:#8b949e;'>In a log-log model, each coefficient is a direct elasticity — how sensitive Rx claims are to each channel.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    elast = api_get("/elasticities")
    if not elast:
        st.info("Train the model first.")
    else:
        rows = []
        for k, v in elast.items():
            if "log_" in k:
                rows.append(
                    {
                        "Channel": CHANNEL_LABELS.get(k, k),
                        "Elasticity": round(v, 4),
                        "Direction": "Positive" if v > 0 else "Negative",
                        "Strength": (
                            "High"
                            if abs(v) > 0.1
                            else "Moderate" if abs(v) > 0.03 else "Low"
                        ),
                    }
                )

        df = pd.DataFrame(rows).sort_values("Elasticity", ascending=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            st.markdown("#### How to read this")
            st.markdown(
                """<div class='insight-box'>
                <b>Elasticity = 0.16</b> means a 1% increase in that channel leads to a 0.16% increase in Rx claims.<br><br>
                <b>Positive</b> channels drive Rx up.<br>
                <b>Negative</b> values may indicate multicollinearity with other channels.<br><br>
                Focus budget on <b>High</b> strength positive channels.
            </div>""",
                unsafe_allow_html=True,
            )


# --- ROI Analysis ---
elif page == "ROI Analysis":
    st.markdown("## ROI Analysis")
    st.markdown(
        "<p style='color:#8b949e;'>Revenue contribution relative to channel cost.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    roi = api_get("/roi")
    if not roi:
        st.info("Train the model first.")
    else:
        COSTS = {
            "log_rep_visits": 250000,
            "log_speaker_programs": 80000,
            "log_samples_distributed": 120000,
            "log_emails_sent": 15000,
            "log_ad_clicks": 40000,
        }

        rows = []
        for k, v in roi.items():
            rows.append(
                {
                    "Channel": CHANNEL_LABELS.get(k, k),
                    "ROI Score": round(v, 4),
                    "Budget (USD)": f"${COSTS.get(k, 0):,}",
                }
            )

        df = pd.DataFrame(rows).sort_values("ROI Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Budget Allocation")
        total = sum(COSTS.values())
        for k, cost in sorted(COSTS.items(), key=lambda x: x[1], reverse=True):
            label = CHANNEL_LABELS.get(k, k)
            pct = int((cost / total) * 100)
            st.markdown(
                f"""<div class='channel-bar-wrap'>
                <div class='channel-name'>{label} — ${cost:,}</div>
                <div class='channel-bar-bg'>
                    <div class='channel-bar-fill' style='width:{pct}%;'></div>
                </div>
                <div class='channel-value'>{pct}% of total budget</div>
            </div>""",
                unsafe_allow_html=True,
            )


# --- Scenario Simulator ---
elif page == "Scenario Simulator":
    st.markdown("## Scenario Simulator")
    st.markdown(
        "<p style='color:#8b949e;'>Predict Rx impact by adjusting channel budgets.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Adjust Channels")
        rep_mult = st.slider("Sales Rep Visits", 0.5, 2.0, 1.0, 0.05)
        speaker_mult = st.slider("Speaker Programs", 0.5, 2.0, 1.0, 0.05)
        samples_mult = st.slider("Free Samples", 0.5, 2.0, 1.0, 0.05)
        emails_mult = st.slider("Emails", 0.5, 2.0, 1.0, 0.05)
        ads_mult = st.slider("Ad Clicks", 0.5, 2.0, 1.0, 0.05)
        desc = st.text_input("Scenario name", "Custom scenario")

        if st.button("Run Simulation"):
            payload = {
                "adjustments": {
                    "log_rep_visits": rep_mult,
                    "log_speaker_programs": speaker_mult,
                    "log_samples_distributed": samples_mult,
                    "log_emails_sent": emails_mult,
                    "log_ad_clicks": ads_mult,
                },
                "description": desc,
            }
            with st.spinner("Simulating..."):
                result = api_post("/simulate", payload)

            if result:
                st.session_state["sim_result"] = result
            else:
                st.error("Simulation failed. Train the model first.")

    with col2:
        st.markdown("#### Results")
        if "sim_result" in st.session_state:
            r = st.session_state["sim_result"]
            delta = r.get("delta_pct", 0)
            color = "#3fb950" if delta >= 0 else "#f85149"
            arrow = "▲" if delta >= 0 else "▼"

            st.markdown(
                f"""<div class='metric-card' style='margin-bottom:1rem;'>
                <div class='metric-value' style='color:{color};'>{arrow} {abs(delta):.1f}%</div>
                <div class='metric-label'>Projected Rx Change</div>
            </div>""",
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.3rem;'>{int(r.get('base_rx_total', 0)):,}</div>
                    <div class='metric-label'>Baseline Rx</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.3rem; color:{color};'>{int(r.get('scenario_rx_total', 0)):,}</div>
                    <div class='metric-label'>Projected Rx</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            if r.get("llm_explanation"):
                st.markdown("#### LLM Analysis")
                st.markdown(
                    f"<div class='insight-box'>{r['llm_explanation']}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='insight-box' style='color:#8b949e;'>Adjust the sliders and run a simulation to see projected Rx impact.</div>",
                unsafe_allow_html=True,
            )


# --- LLM Insights ---
elif page == "LLM Insights":
    st.markdown("## LLM Insights")
    st.markdown(
        "<p style='color:#8b949e;'>Gemini-powered analysis of your MMM results.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Channel Insights")
        if st.button("Get Channel Insights"):
            with st.spinner("Asking Gemini..."):
                data = api_get("/insights")
            if data and data.get("llm_insights"):
                st.markdown(
                    f"<div class='insight-box'>{data['llm_insights']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No insights returned. Make sure GEMINI_API_KEY is set.")

    with col2:
        st.markdown("#### Budget Recommendations")
        if st.button("Get Budget Recommendation"):
            with st.spinner("Asking Gemini..."):
                data = api_get("/recommend")
            if data and data.get("llm_recommendation"):
                st.markdown(
                    f"<div class='insight-box'>{data['llm_recommendation']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    "No recommendation returned. Make sure GEMINI_API_KEY is set."
                )
